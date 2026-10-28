"""Command line for generation: `python -m bko.generate <definition> --log PATH`, or `direct`.

The command fetches UCI Heart Disease and the releases its field mappings name into the data root
if they are absent, splits the records, and reads the training split's gaps under the chosen
definition of `bko.gaps.definitions`. It asks the Anthropic Messages API for each count-flagged
cell's shortfall below `UNDERREPRESENTED_BELOW` training records (E1's count-based gap request),
or with `--request completeness` each cell holding a per-subgroup gap for
`UNDERREPRESENTED_BELOW` rows, at the chosen guidance level. Inside the call log, before any
call, it writes the candidates header to stdout (`bko.generate.candidates`): the run's request,
definition, level, pins, split fingerprint and call plan. Then it writes one JSON line per asked
record: the call, the cell, the record's position in the call, and its decoded `candidate` or its
`undecoded` reason. Each call's lines are written and flushed as it returns, so a failed run
keeps the earlier ones.
Every call's request and response go to the call log at `--log`, which must not exist; the plan,
each call, the run's totals and the output's digest go to stderr through logging. Calls spend
money and a rerun's candidates differ. `--guidance` sets the level, default `summary`.

`python -m bko.generate direct --seed N --log PATH` makes no call: it fills the same request
from the fused BKB's concept shares through `bko.generate.direct`, writes a schema 2 header naming
the generator and seed, then one line per record, and the same bytes on every run with the same
seed and inputs. `--log` is its draw log, created readable by its owner only: a `draw` line per
record with the concepts drawn, their closure and the fields read from the label group.
`docs/DEVELOPMENT.md` (Generation) is the usage home.

Exit codes: 0 the candidates are written; 1 an unexpected error, with its traceback; 2 the command
could not run, no credential, an API failure, an existing log, a completeness request no cell holds
a gap for, `--request completeness` with another definition, or a draw no training record can
supply; 3 the data or a release contradicts its pin or schema.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pandas as pd

from bko.bkb.structure import BKB
from bko.datasets.cells import UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.gaps.definitions import Definition, completeness_request, gaps
from bko.gaps.detect import count_gaps
from bko.gaps.observe import Cell, Cells, cell_key
from bko.gaps.scope import load_uci_scope
from bko.generate.api import HTTP_LOGGERS, default_send
from bko.generate.calls import Batch, chunks, generate
from bko.generate.candidates import Generator, Header, Request, render_header, render_lines
from bko.generate.direct import draw_line, sample
from bko.generate.errors import GenerationError
from bko.generate.guidance import guide
from bko.generate.log import CallLog, utc_now
from bko.generate.messages import SETTINGS, Level, Send, Settings
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping


logger = logging.getLogger("bko.generate")

DIRECT = "direct"
"""The subcommand that draws from the fused BKB instead of calling the API."""

QUIET_LOGGERS = HTTP_LOGGERS
"""The SDK's HTTP client logs each request at INFO; the call log already holds them."""


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
        0 done, 2 the command could not run, a call got no response or a draw had no record,
        3 the data contradicts its pin or schema. Any other exception propagates, so the process
        exits 1.
    """
    context = context or Context()
    parser = _parser(context.settings)
    args = parser.parse_args(argv)
    if (
        args.command != DIRECT
        and args.request == Request.COMPLETENESS
        and args.command != Definition.SUBGROUP
    ):
        parser.error("--request completeness reads the per-subgroup gaps; use per-subgroup")
    run = _direct if args.command == DIRECT else _run
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        run(args, data_dir, context, _Writer(out or sys.stdout))
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
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--request",
        choices=[request.value for request in Request],
        default=Request.E1.value,
        help="the gap request to fill; completeness needs per-subgroup; default: %(default)s",
    )
    shared.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    parser = argparse.ArgumentParser(
        prog="python -m bko.generate",
        description=__doc__,
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        help="a gap definition whose gaps guide each cell's calls, or direct",
    )
    for definition in Definition:
        command = commands.add_parser(definition.value, parents=[shared])
        command.add_argument(
            "--guidance",
            choices=[level.value for level in Level],
            default=settings.guidance.value,
            help="the guidance level; default: %(default)s",
        )
        command.add_argument(
            "--log", type=Path, required=True, help="the call log to create; it must not exist"
        )
    direct = commands.add_parser(DIRECT, parents=[shared])
    direct.add_argument(
        "--seed", type=_seed, required=True, help="the sampler's seed, a non-negative integer"
    )
    direct.add_argument(
        "--log", type=Path, required=True, help="the draw log to create; it must not exist"
    )
    return parser


def _seed(text: str) -> int:
    seed = int(text)
    if seed < 0:
        msg = f"the seed must be a non-negative integer, not {seed}"
        raise argparse.ArgumentTypeError(msg)
    return seed


class _Writer:
    """Writes the header, then each batch's candidate lines as its call returns; digests both."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._rows = 0
        self._digest = hashlib.sha256()

    def header(self, text: str) -> None:
        """Write the header line, flushed and digested, counting no row."""
        self._out.write(text)
        self._out.flush()
        self._digest.update(text.encode())

    def write(self, batch: Batch) -> None:
        self.lines(batch.index, batch.cell, batch.candidates)

    def lines(self, index: int, cell: Cell, candidates: Sequence[object]) -> None:
        """Write one call's candidate lines, flushed and digested."""
        text = render_lines(index, cell, candidates)
        self._out.write(text)
        self._out.flush()
        self._rows += len(candidates)
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
    definition = Definition(args.command)
    found = gaps(definition, fragments, cells)
    request = Request(args.request)
    quota = _request(request, parts.train, fragments, cells)
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
    log = CallLog(args.log, settings, run="generate", now=context.now)
    try:
        with log:
            send = context.send()
            writer.header(
                render_header(
                    Header(
                        request=request,
                        definition=definition,
                        guidance=settings.guidance,
                        uci=context.uci.sha256,
                        fingerprint=parts.fingerprint,
                        releases=tuple(
                            (prefix, context.releases[prefix].sha256)
                            for prefix in sorted(ontologies)
                        ),
                        calls=tuple(
                            (cell_key(cell), rows)
                            for cell, rows in chunks(quota, settings.per_call)
                        ),
                    )
                )
            )
            generate(
                quota,
                guidance,
                send=send,
                log=log,
                settings=settings,
                on_batch=writer.write,
            )
    finally:
        # After the last write and the run's totals, whether or not every call returned.
        writer.summary()


def _direct(args: argparse.Namespace, data_dir: Path, context: Context, writer: _Writer) -> None:
    """Fill the request from the fused BKB's shares: header, then each planned call's records."""
    if args.log.exists():
        raise FileExistsError(errno.EEXIST, "the draw log exists", str(args.log))
    scope = load_uci_scope(
        data_dir,
        uci=context.uci,
        releases=context.releases,
        field_rows=context.field_rows,
        open_url=context.open_url,
    )
    request = Request(args.request)
    quota = _request(request, scope.parts.train, scope.fragments, scope.cells)
    drawn = sample(
        scope.parts.train,
        quota,
        rows=context.field_rows,
        ontologies=scope.ontologies,
        name=context.uci.name,
        seed=args.seed,
    )
    plan = chunks(quota, context.settings.per_call)
    # Owner-only and exclusive, as `CallLog` creates the call log.
    descriptor = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as log:
        writer.header(
            render_header(
                Header(
                    request=request,
                    definition=None,
                    guidance=None,
                    uci=context.uci.sha256,
                    fingerprint=scope.parts.fingerprint,
                    releases=tuple(
                        (prefix, context.releases[prefix].sha256)
                        for prefix in sorted(scope.ontologies)
                    ),
                    calls=tuple((cell_key(cell), rows) for cell, rows in plan),
                    generator=Generator.DIRECT,
                    seed=args.seed,
                )
            )
        )
        taken = dict.fromkeys(quota, 0)
        for index, (cell, rows) in enumerate(plan, start=1):
            draws = drawn[cell][taken[cell] : taken[cell] + rows]
            taken[cell] += rows
            log.write("".join(draw_line(index, p, d) for p, d in enumerate(draws)))
            writer.lines(index, cell, [dict(d.record) for d in draws])
    logger.info("draws seed=%d rows=%d log=%s", args.seed, sum(quota.values()), args.log)
    writer.summary()


def _request(
    request: Request, train: pd.DataFrame, fragments: Sequence[BKB], cells: Cells
) -> dict[Cell, int]:
    """The request's rows per cell, each logged."""
    if request is Request.COMPLETENESS:
        quota = completeness_request(fragments, cells)
    else:
        quota = _quota(train)
    logger.info("request name=%s cells=%d rows=%d", request, len(quota), sum(quota.values()))
    for cell, rows in quota.items():
        logger.info("quota key=%s rows=%d", cell_key(cell), rows)
    return quota


def _quota(train: pd.DataFrame) -> dict[Cell, int]:
    """E1's count-based request: each flagged cell's shortfall below `UNDERREPRESENTED_BELOW`."""
    return {gap.cell: UNDERREPRESENTED_BELOW - gap.count for gap in count_gaps(cell_counts(train))}


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
