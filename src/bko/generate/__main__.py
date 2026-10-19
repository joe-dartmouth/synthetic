"""Command line for generation: `python -m bko.generate <definition> --log PATH`.

The command fetches UCI Heart Disease and the releases its field mappings name into the data root
if they are absent, splits the records, and reads the training split's gaps under the chosen
definition of `bko.gaps.definitions`. It asks the Anthropic Messages API for each count-flagged
cell's shortfall below `UNDERREPRESENTED_BELOW` training records (E1's count-based gap request),
at the chosen guidance level, and writes one JSON line per asked record to stdout: the call, the
cell, the record's position in the call, and its decoded `candidate` or its `undecoded` reason.
Each call's lines are written and flushed as it returns, so a failed run keeps the earlier ones.
Every call's request and response go to the call log at `--log`, which must not exist; the plan,
each call, the run's totals and the output's digest go to stderr through logging. Calls spend
money and a rerun's candidates differ. `docs/DEVELOPMENT.md` (Generation) is the usage home.

Exit codes: 0 the candidates are written; 1 an unexpected error, with its traceback; 2 the command
could not run, no credential, an API failure, or an existing log; 3 the data or a release
contradicts its pin or schema.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import pandas as pd

from bko.datasets.cells import UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.datasets.split import split
from bko.datasets.uci import load_uci
from bko.gaps.definitions import Definition, gaps
from bko.gaps.detect import count_gaps
from bko.gaps.observe import Cell, cell_key, record_cells
from bko.gaps.scope import load_scope
from bko.generate.api import default_send
from bko.generate.calls import Batch, generate
from bko.generate.decode import Undecoded
from bko.generate.errors import GenerationError
from bko.generate.guidance import guide
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Level, Send, Settings
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping


logger = logging.getLogger("bko.generate")

QUIET_LOGGERS = ("httpx2",)
"""The SDK's HTTP client logs each request at INFO; the call log already holds them."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        releases: The pinned release for each concept prefix.
        field_rows: The UCI field mappings.
        send: Builds the send every call goes through.
        settings: How each call is made; `--guidance` replaces its level.
        now: The clock the call log reads.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS
    send: Callable[[], Send] = default_send
    settings: Settings = SETTINGS
    now: Callable[[], datetime] = _utc_now


def main(
    argv: Sequence[str] | None = None,
    *,
    context: Context | None = None,
    out: TextIO | None = None,
) -> int:
    """Generate the request's candidates and return the exit code.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when omitted.
        context: The outside world; the real one when omitted.
        out: Where the candidates go; stdout when omitted.

    Returns:
        0 done, 2 the command could not run or a call got no response, 3 the data contradicts
        its pin or schema. Any other exception propagates, so the process exits 1.
    """
    context = context or Context()
    args = _parser(context.settings).parse_args(argv)
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        _run(args, data_dir, context, _Writer(out or sys.stdout))
    except DatasetError as exc:
        # The message is the diagnosis; a traceback would read as a crash in the loader.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except (GenerationError, OSError) as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    return 0


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr, and quiet the SDK's HTTP client."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.captureWarnings(True)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bko.generate", description=__doc__)
    parser.add_argument(
        "definition",
        choices=[definition.value for definition in Definition],
        help="the gap definition whose gaps guide each cell",
    )
    parser.add_argument(
        "--guidance",
        choices=[level.value for level in Level],
        default=settings.guidance.value,
        help="the guidance level; default: %(default)s",
    )
    parser.add_argument(
        "--log", type=Path, required=True, help="the call log to create; it must not exist"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    return parser


class _Writer:
    """Writes each batch's candidate lines to stdout as its call returns, and digests them."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._rows = 0
        self._digest = hashlib.sha256()

    def write(self, batch: Batch) -> None:
        text = "".join(_lines(batch))
        self._out.write(text)
        self._out.flush()
        self._rows += len(batch.candidates)
        self._digest.update(text.encode())

    def summary(self) -> None:
        logger.info("candidates rows=%d sha256=%s", self._rows, self._digest.hexdigest())


def _run(args: argparse.Namespace, data_dir: Path, context: Context, writer: _Writer) -> None:
    # Checked before the downloads so a rerun fails fast; `CallLog` still creates the file
    # exclusively, so a log made in between is never overwritten.
    if args.log.exists():
        raise FileExistsError(errno.EEXIST, "the call log exists", str(args.log))
    ensure(context.uci, data_dir, open_url=context.open_url)
    parts = split(load_uci(data_dir, source=context.uci))
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        parts.seed,
        len(parts.train),
        len(parts.test),
        parts.fingerprint,
    )
    ontologies, fragments = load_scope(
        {row.concept for row in context.field_rows},
        data_dir,
        releases=context.releases,
        open_url=context.open_url,
    )
    cells = record_cells(parts.train, context.field_rows, name=context.uci.name)
    definition = Definition(args.definition)
    found = gaps(definition, fragments, cells)
    quota = _quota(parts.train)
    guidance = guide(
        parts.train,
        cells,
        found,
        definition=definition,
        fragments=fragments,
        ontologies=ontologies,
        rows=context.field_rows,
    )
    settings = dataclasses.replace(context.settings, guidance=Level(args.guidance))
    log = CallLog(args.log, settings, now=context.now)
    try:
        with log:
            generate(
                quota,
                guidance,
                send=context.send(),
                log=log,
                settings=settings,
                on_batch=writer.write,
            )
    finally:
        # After the last write and the run's totals, whether or not every call returned.
        writer.summary()


def _quota(train: pd.DataFrame) -> dict[Cell, int]:
    """E1's count-based request: each flagged cell's shortfall below `UNDERREPRESENTED_BELOW`."""
    return {gap.cell: UNDERREPRESENTED_BELOW - gap.count for gap in count_gaps(cell_counts(train))}


def _lines(batch: Batch) -> list[str]:
    lines: list[str] = []
    for position, candidate in enumerate(batch.candidates):
        line: dict[str, object] = {
            "call": batch.index,
            "cell": cell_key(batch.cell),
            "position": position,
        }
        if isinstance(candidate, Undecoded):
            line["undecoded"] = candidate.reason.value
        else:
            line["candidate"] = candidate
        lines.append(json.dumps(line, sort_keys=True) + "\n")
    return lines


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
