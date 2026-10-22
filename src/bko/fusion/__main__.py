"""Command line for fusion: `python -m bko.fusion CANDIDATES`.

The command reads a candidates file `python -m bko.generate per-subgroup --request completeness`
wrote, fetches UCI Heart Disease and the releases its field mappings name into the data root if
they are absent, splits the records, and lists the training split's gaps under every definition
of `bko.gaps.definitions`. It refuses a file that does not answer the completeness request,
validates each requested cell's candidates, joins the accepted records (ADR
`research/gap-closure-design`), lists the gaps again, and writes one CSV to stdout: every (cell,
statement) pair listed before or after the join under each definition, with whether it was
listed before and after. The same bytes on every run over the same inputs, and nothing unless
the whole report is built. It makes no API call. `docs/DEVELOPMENT.md` (Gap closure) is the
usage home.

Exit codes: 0 the report is written; 1 an unexpected error, with its traceback; 2 the command
could not run; 3 the candidates file, the data or a release contradicts its contract, pin or
schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

import pandas as pd

from bko.datasets.cells import CELL_COLUMNS
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.fusion.closure import Closure, closure
from bko.fusion.errors import ClosureError
from bko.fusion.join import Joined, join
from bko.gaps.definitions import completeness_request
from bko.gaps.observe import Cell, Cells, cell_key, record_cells
from bko.gaps.report import STATEMENT_COLUMNS, Label, Row, cell_cells, labeller, statement_cells
from bko.gaps.scope import load_uci_scope
from bko.generate.candidates import CandidateLine, read
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.ontology.schema import FieldMapping
from bko.validate.checks import validate


logger = logging.getLogger("bko.fusion")

CLOSURE_COLUMNS: Final[Row] = ("definition", *STATEMENT_COLUMNS, *CELL_COLUMNS, "before", "after")


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        releases: The pinned release for each concept prefix.
        field_rows: The UCI field mappings.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS


def main(
    argv: Sequence[str] | None = None,
    *,
    context: Context | None = None,
    out: TextIO | None = None,
) -> int:
    """Write the closure report and return the exit code.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when omitted.
        context: The outside world; the real one when omitted.
        out: Where the report goes; stdout when omitted.

    Returns:
        0 done, 2 the command could not run, 3 an input contradicts its contract, pin or schema.
        Any other exception propagates, so the process exits 1.
    """
    args = _parser().parse_args(argv)
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        rows = _run(args.candidates, data_dir, Context() if context is None else context)
        text = _csv(rows)
        (out or sys.stdout).write(text)
    except DatasetError as exc:
        # The message is the diagnosis; a traceback would read as a crash in the loader.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("report rows=%d sha256=%s", len(rows) - 1, digest)
    return 0


def _csv(rows: Sequence[Row]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr, keeping stdout for the report."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.captureWarnings(True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bko.fusion", description=__doc__)
    parser.add_argument(
        "candidates",
        type=Path,
        help="the stdout of python -m bko.generate per-subgroup --request completeness",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    return parser


def _run(path: Path, data_dir: Path, context: Context) -> list[Row]:
    logger.info(
        "data_dir=%s pandas=%s scipy=%s",
        data_dir,
        importlib.metadata.version("pandas"),
        importlib.metadata.version("scipy"),
    )
    data = path.read_bytes()
    lines = read(data)
    digest = hashlib.sha256(data).hexdigest()
    logger.info("candidates path=%s lines=%d sha256=%s", path, len(lines), digest)
    scope = load_uci_scope(
        data_dir,
        uci=context.uci,
        releases=context.releases,
        field_rows=context.field_rows,
        open_url=context.open_url,
    )
    parts, ontologies, fragments = scope.parts, scope.ontologies, scope.fragments
    before = scope.cells
    request = completeness_request(fragments, before)
    candidates = _answer(lines, request, before)
    accepted: dict[Cell, pd.DataFrame] = {}
    for cell, items in candidates.items():
        logger.info("validate key=%s lines=%d", cell_key(cell), len(items))
        accepted[cell] = validate(items).accepted
    joined = join(parts.train, accepted, request)
    after = record_cells(joined.frame, context.field_rows, name=f"{context.uci.name}-joined")
    _log_join(candidates, accepted, joined, before, after)
    readings = closure(fragments, before, after)
    _log_closure(readings)
    return _report(readings, before, labeller(ontologies))


def _answer(
    lines: Sequence[CandidateLine], request: Mapping[Cell, int], cells: Cells
) -> dict[Cell, tuple[object, ...]]:
    """Each requested cell's candidates in file order; a file that does not answer is refused."""
    keys = {cell_key(cell): cell for cell in cells.each}
    found: dict[Cell, list[object]] = {cell: [] for cell in request}
    for line in lines:
        cell = keys.get(line.cell)
        if cell is None or cell not in found:
            msg = (
                f"call {line.call} answers {line.cell!r}, "
                "a cell the completeness request does not ask for"
            )
            raise ClosureError(msg)
        found[cell].append(line.candidate)
    short = [
        f"{cell_key(cell)} has {len(found[cell])} lines for {rows} rows"
        for cell, rows in request.items()
        if len(found[cell]) < rows
    ]
    if short:
        msg = f"the candidates do not answer the completeness request: {'; '.join(short)}"
        raise ClosureError(msg)
    for cell, rows in request.items():
        logger.info("request key=%s rows=%d lines=%d", cell_key(cell), rows, len(found[cell]))
    return {cell: tuple(items) for cell, items in found.items()}


def _log_join(
    candidates: Mapping[Cell, tuple[object, ...]],
    accepted: Mapping[Cell, pd.DataFrame],
    joined: Joined,
    before: Cells,
    after: Cells,
) -> None:
    for cell, items in candidates.items():
        held = set(before.each[cell].observed.held)
        newly = [concept for concept in after.each[cell].observed.held if concept not in held]
        logger.info(
            "join key=%s lines=%d passed=%d off_cell=%d past_quota=%d joined=%d newly_held=%s",
            cell_key(cell),
            len(items),
            len(accepted[cell]),
            joined.off_cell[cell],
            joined.past_quota[cell],
            joined.joined[cell],
            ",".join(newly) or "none",
        )
        if joined.off_cell[cell]:
            logger.warning(
                "off_cell key=%s dropped=%d: accepted records whose own values name another cell",
                cell_key(cell),
                joined.off_cell[cell],
            )
        if joined.past_quota[cell]:
            logger.warning(
                "past_quota key=%s dropped=%d: accepted on-cell records past the cell's rows",
                cell_key(cell),
                joined.past_quota[cell],
            )
        if not joined.joined[cell]:
            logger.warning(
                "unjoined key=%s: no record joined, so none of its gaps can close", cell_key(cell)
            )


def _log_closure(readings: Sequence[Closure]) -> None:
    for reading in readings:
        share = "undefined" if reading.share is None else f"{reading.share:.4f}"
        logger.info(
            "closure definition=%s before=%d after=%d closed=%d open=%d new=%d share=%s",
            reading.definition,
            len(reading.before),
            len(reading.after),
            len(reading.closed),
            len(reading.still_open),
            len(reading.opened),
            share,
        )


def _report(readings: Sequence[Closure], cells: Cells, label: Label) -> list[Row]:
    rows: list[Row] = [CLOSURE_COLUMNS]
    for reading in readings:
        listed_before, listed_after = set(reading.before), set(reading.after)
        rows.extend(
            (
                reading.definition.value,
                *statement_cells(gap.statement, label),
                *cell_cells(gap.cell),
                str(gap in listed_before),
                str(gap in listed_after),
            )
            for gap in reading.listed(cells.each)
        )
    return rows


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
