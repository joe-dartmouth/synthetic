"""Command line for generation: `python -m bko.generate <definition> --log PATH`.

The command fetches UCI Heart Disease and the releases its field mappings name into the data root
if they are absent, splits the records, and reads the training split's gaps under the chosen
definition of `bko.gaps.definitions`. It asks the Anthropic Messages API for each count-flagged
cell's shortfall below `UNDERREPRESENTED_BELOW` training records (E1's count-based gap request),
or with `--request completeness` each cell holding a per-subgroup gap for
`UNDERREPRESENTED_BELOW` rows, at the chosen guidance level, and writes one JSON line per asked
record to stdout: the call, the cell, the record's position in the call, and its decoded
`candidate` or its `undecoded` reason. Each call's lines are written and flushed as it returns,
so a failed run keeps the earlier ones.
Every call's request and response go to the call log at `--log`, which must not exist; the plan,
each call, the run's totals and the output's digest go to stderr through logging. Calls spend
money and a rerun's candidates differ. `docs/DEVELOPMENT.md` (Generation) is the usage home.

Exit codes: 0 the candidates are written; 1 an unexpected error, with its traceback; 2 the command
could not run, no credential, an API failure, an existing log, a completeness request no cell holds
a gap for, or `--request completeness` with another definition; 3 the data or a release
contradicts its pin or schema.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TextIO

import pandas as pd

from bko.datasets.cells import UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.gaps.definitions import Definition, completeness_request, gaps
from bko.gaps.detect import count_gaps
from bko.gaps.observe import Cell, cell_key
from bko.gaps.scope import load_uci_scope
from bko.generate.api import HTTP_LOGGERS, default_send
from bko.generate.calls import Batch, generate
from bko.generate.candidates import render_lines
from bko.generate.errors import GenerationError
from bko.generate.guidance import guide
from bko.generate.log import CallLog, utc_now
from bko.generate.messages import SETTINGS, Level, Send, Settings
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping


logger = logging.getLogger("bko.generate")

QUIET_LOGGERS = HTTP_LOGGERS
"""The SDK's HTTP client logs each request at INFO; the call log already holds them."""


class Request(StrEnum):
    """The gap request the command fills."""

    E1 = "e1"
    """E1's count-based request: each count-flagged cell's shortfall below the threshold."""
    COMPLETENESS = "completeness"
    """The completeness request: the threshold for each cell holding a per-subgroup gap."""


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
    now: Callable[[], datetime] = utc_now


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
    parser = _parser(context.settings)
    args = parser.parse_args(argv)
    if args.request == Request.COMPLETENESS and args.definition != Definition.SUBGROUP:
        parser.error("--request completeness reads the per-subgroup gaps; use per-subgroup")
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
        "--request",
        choices=[request.value for request in Request],
        default=Request.E1.value,
        help="the gap request to fill; completeness needs per-subgroup; default: %(default)s",
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
        text = render_lines(batch.index, batch.cell, batch.candidates)
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
    scope = load_uci_scope(
        data_dir,
        uci=context.uci,
        releases=context.releases,
        field_rows=context.field_rows,
        open_url=context.open_url,
    )
    parts, ontologies, fragments, cells = (
        scope.parts,
        scope.ontologies,
        scope.fragments,
        scope.cells,
    )
    definition = Definition(args.definition)
    found = gaps(definition, fragments, cells)
    request = Request(args.request)
    if request is Request.COMPLETENESS:
        quota = completeness_request(fragments, cells)
    else:
        quota = _quota(parts.train)
    logger.info("request name=%s cells=%d rows=%d", request, len(quota), sum(quota.values()))
    for cell, rows in quota.items():
        logger.info("quota key=%s rows=%d", cell_key(cell), rows)
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


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
