"""Command line for gap detection: `python -m bko.gaps uci` and `synthea`.

`uci` fetches UCI Heart Disease and the releases its field mappings name into the data root if
they are absent, splits the records, and lists the training split's gaps under each definition of
`bko.gaps.definitions`, as (cell, statement) pairs, then the cells holding fewer training records
than the threshold. `synthea` generates the population E2 reads if it is absent, fetches the
releases its code mappings name, and for each of E2's ablations lists every definition's gaps and
the count-based ones before and after the removal, marking the gaps the removed patients account
for. Each writes one CSV to stdout, the same bytes on every run over the same inputs, and nothing
unless the whole report is built; each input's identity, each cell's records, each flagged concept,
each list's length, each detector's recovery and the report's digest go to stderr through logging.
`docs/DEVELOPMENT.md` (Gap detection) is the usage home.

Exit codes: 0 the report is written; 1 an unexpected error, with its traceback; 2 the command could
not run; 3 the data or a release contradicts its pin or schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import logging
import sys
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from bko.bkb.structure import BKB
from bko.datasets.ablation import AblationSpec
from bko.datasets.cells import CELL_COLUMNS, UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import SYNTHEA_JAR, UCI_HEART_DISEASE, Source
from bko.datasets.split import split
from bko.datasets.synthea import load_population
from bko.datasets.synthea_run import Runner, SyntheaRun, default_java, ensure_population, run_java
from bko.datasets.uci import load_uci
from bko.gaps.definitions import Definition, gaps, under_represented
from bko.gaps.detect import Statement, count_gaps, ontology_fragments
from bko.gaps.errors import GapError
from bko.gaps.experiment import E2_ABLATIONS, E2_RUN, SYNTHEA, AblationRun, run_ablation
from bko.gaps.observe import Cell, Cells, Observed, cell_key, observe_population, record_cells
from bko.ontology.mapping import CODE_MAPPINGS, FIELD_MAPPINGS, RELEASES
from bko.ontology.releases import load_releases
from bko.ontology.schema import CodeMapping, FieldMapping
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.gaps")

type Row = tuple[str, ...]
type Label = Callable[[str], str]

STATEMENT_COLUMNS: Final[Row] = (
    "source",
    "antecedent",
    "antecedent_label",
    "consequent",
    "consequent_label",
)
UCI_COLUMNS: Final[Row] = ("detector", *STATEMENT_COLUMNS, *CELL_COLUMNS, "count")
CLOSED_DETECTORS: Final = frozenset({Definition.SUBGROUP.value, "count"})
"""The detectors that, under a removal of whole patients, put no gap new outside the truth and
lose none (`bko.gaps.experiment`); a reading of either that does is a defect."""
SYNTHEA_COLUMNS: Final[Row] = (
    "ablation",
    "detector",
    *STATEMENT_COLUMNS,
    *CELL_COLUMNS,
    "before",
    "after",
    "truth",
)


@dataclass(frozen=True, slots=True)
class Context:
    """What the commands reach outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        releases: The pinned release for each concept prefix.
        field_rows: The UCI field mappings.
        code_rows: The Synthea code mappings.
        synthea_jar: The pinned Synthea jar.
        resolve_java: Returns the Java to run Synthea with and its version line.
        runner: Runs Synthea's argv in a directory and returns its exit status.
        run: The Synthea population E2 reads.
        ablations: E2's ablations by name, in report order.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS
    code_rows: tuple[CodeMapping, ...] = CODE_MAPPINGS
    synthea_jar: Source = SYNTHEA_JAR
    resolve_java: Callable[[], tuple[Path, str]] = default_java
    runner: Runner = run_java
    run: SyntheaRun = E2_RUN
    ablations: Mapping[str, AblationSpec] = E2_ABLATIONS


def main(
    argv: Sequence[str] | None = None,
    *,
    context: Context | None = None,
    out: TextIO | None = None,
) -> int:
    """Run one subcommand and return its exit code.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when omitted.
        context: The outside world; the real one when omitted.
        out: Where the report goes; stdout when omitted.

    Returns:
        0 done, 2 the command could not run, 3 the data contradicts its pin or schema.
    """
    args = _parser().parse_args(argv)
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        command = {"uci": _uci, "synthea": _synthea}[args.command]
        logger.info(
            "data_dir=%s pandas=%s scipy=%s",
            data_dir,
            importlib.metadata.version("pandas"),
            importlib.metadata.version("scipy"),
        )
        rows = command(data_dir, Context() if context is None else context)
    except DatasetError as exc:
        # The message is the diagnosis; a traceback would read as a crash in the loader.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    text = buffer.getvalue()
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("gaps rows=%d sha256=%s", len(rows) - 1, digest)
    (out or sys.stdout).write(text)
    return 0


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr, keeping stdout for the report."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.captureWarnings(True)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    parser = argparse.ArgumentParser(prog="python -m bko.gaps", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "uci",
        parents=[common],
        help="list each definition's gaps and the count-based ones in the UCI training split",
    )
    commands.add_parser(
        "synthea",
        parents=[common],
        help=(
            "list each definition's gaps and the count-based ones before and after "
            "each of E2's ablations"
        ),
    )
    return parser


def _uci(data_dir: Path, context: Context) -> list[Row]:
    ensure(context.uci, data_dir, open_url=context.open_url)
    parts = split(load_uci(data_dir, source=context.uci))
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        parts.seed,
        len(parts.train),
        len(parts.test),
        parts.fingerprint,
    )
    concepts = {row.concept for row in context.field_rows}
    ontologies, fragments = _ontology(concepts, data_dir, context)
    cells = record_cells(parts.train, context.field_rows, name=context.uci.name)
    _log_observed(cells.whole)
    _log_cells(cells, f"dataset={cells.whole.name}")
    label = _labeller(ontologies)
    rows: list[Row] = [UCI_COLUMNS]
    for definition in Definition:
        found = gaps(definition, fragments, cells)
        logger.info(
            "bko definition=%s dataset=%s gaps=%d cells=%d",
            definition,
            cells.whole.name,
            len(found),
            len({gap.cell for gap in found}),
        )
        rows.extend(
            (definition.value, *_statement_cells(gap.statement, label), *_cell_cells(gap.cell), "")
            for gap in found
        )
    counted = count_gaps(cell_counts(parts.train))
    logger.info(
        "count dataset=%s flagged=%d below=%d",
        cells.whole.name,
        len(counted),
        UNDERREPRESENTED_BELOW,
    )
    rows.extend(
        ("count", "", "", "", "", "", cell.age_band, cell.sex, str(cell.target), str(cell.count))
        for cell in counted
    )
    return rows


def _synthea(data_dir: Path, context: Context) -> list[Row]:
    directory = ensure_population(
        context.run,
        data_dir,
        jar_source=context.synthea_jar,
        open_url=context.open_url,
        resolve_java=context.resolve_java,
        runner=context.runner,
    )
    population = load_population(directory, snapshot_date=context.run.snapshot_date)
    concepts = {row.concept for row in context.code_rows}
    ontologies, fragments = _ontology(concepts, data_dir, context)
    _log_observed(observe_population(population, context.code_rows, name=SYNTHEA))
    label = _labeller(ontologies)
    rows: list[Row] = [SYNTHEA_COLUMNS]
    for name, spec in context.ablations.items():
        run = run_ablation(name, population, spec, fragments, context.code_rows)
        logger.info(
            "ablation name=%s codes=%s age_band=%s sex=%s edges=%s removed=%d",
            name,
            ",".join(sorted(spec.codes)),
            spec.age_band,
            spec.sex,
            ",".join(str(edge) for edge in spec.edges),
            len(run.ablation.removed),
        )
        _log_cells(run.before_cells, f"ablation={name} when=before")
        _log_cells(run.after_cells, f"ablation={name} when=after")
        for definition in Definition:
            logger.info(
                "bko ablation=%s definition=%s before=%d after=%d listed=%d",
                name,
                definition,
                len(run.before[definition]),
                len(run.after[definition]),
                len(run.listed(definition)),
            )
        readings = [(definition.value, run.reading(definition)) for definition in Definition]
        for detector, reading in [*readings, ("count", run.count)]:
            logger.info(
                "recovery ablation=%s detector=%s truth=%d eligible=%d recovered=%d outside=%d "
                "new_outside=%d vanished=%d",
                name,
                detector,
                reading.truth,
                reading.eligible,
                reading.recovered,
                reading.outside,
                reading.new_outside,
                reading.vanished,
            )
            if detector in CLOSED_DETECTORS and (reading.new_outside or reading.vanished):
                logger.warning(
                    "invariant ablation=%s detector=%s: a removal of whole patients puts no gap "
                    "new outside the truth and loses none, so this reading is a defect",
                    name,
                    detector,
                )
        rows.extend(_ablation_rows(run, label))
    return rows


def _ablation_rows(run: AblationRun, label: Label) -> Iterator[Row]:
    truth = set(run.truth)
    for definition in Definition:
        before, after = set(run.before[definition]), set(run.after[definition])
        for gap in run.listed(definition):
            flags = _flags(gap, before, after, truth)
            yield (
                run.name,
                definition.value,
                *_statement_cells(gap.statement, label),
                *_cell_cells(gap.cell),
                *flags,
            )
    flagged_before = {gap.cell for gap in run.count_before}
    flagged_after = {gap.cell for gap in run.count_after}
    fell = set(run.count_truth)
    listed = flagged_before | flagged_after | fell
    grid = run.ablation.before[list(CELL_COLUMNS)].itertuples(index=False, name=None)
    for band, sex, target in grid:
        cell = (str(band), str(sex), int(target))
        if cell in listed:
            flags = _flags(cell, flagged_before, flagged_after, fell)
            yield (run.name, "count", "", "", "", "", "", cell[0], cell[1], str(cell[2]), *flags)


def _flags(
    item: object, before: Collection[object], after: Collection[object], truth: Collection[object]
) -> Row:
    return (str(item in before), str(item in after), str(item in truth))


def _ontology(
    concepts: Collection[str], data_dir: Path, context: Context
) -> tuple[dict[str, Ontology], tuple[BKB, ...]]:
    prefixes = sorted({concept.partition(":")[0] for concept in concepts})
    unpinned = [prefix for prefix in prefixes if prefix not in context.releases]
    if unpinned:
        msg = f"no pinned release for the prefix {', '.join(unpinned)}"
        raise GapError(msg)
    sources = {prefix: context.releases[prefix] for prefix in prefixes}
    ontologies = load_releases(sources, data_dir, open_url=context.open_url)
    fragments = ontology_fragments(ontologies, concepts)
    for fragment in fragments:
        logger.info(
            "scope release=%s classes=%d statements=%d",
            fragment.name,
            len(fragment.inodes),
            len(fragment.snodes),
        )
    return ontologies, fragments


def _log_observed(observed: Observed) -> None:
    logger.info(
        "observed dataset=%s records=%d held=%d of=%d %s",
        observed.name,
        observed.records,
        len(observed.held),
        len(observed.holders),
        " ".join(f"{concept}={count}" for concept, count in observed.holders.items()),
    )


def _log_cells(cells: Cells, where: str) -> None:
    for cell, records in cells.each.items():
        logger.info(
            "cell %s key=%s records=%d held=%d of=%d assumed=%s",
            where,
            cell_key(cell),
            records.observed.records,
            len(records.observed.held),
            len(records.observed.holders),
            ",".join(sorted(records.assumed)),
        )
    for flag in under_represented(cells):
        logger.info(
            "flagged %s key=%s concept=%s shown=%d records=%d reference_shown=%d "
            "reference_records=%d mass=%.4f",
            where,
            cell_key(flag.cell),
            flag.concept,
            flag.shown,
            flag.records,
            flag.reference_shown,
            flag.reference_records,
            flag.mass,
        )


def _cell_cells(cell: Cell) -> Row:
    """A cell's `CELL_COLUMNS` values, `target` blank where cells carry no label."""
    return tuple(str(part) for part in cell) + ("",) * (len(CELL_COLUMNS) - len(cell))


def _labeller(ontologies: Mapping[str, Ontology]) -> Label:
    def label(curie: str) -> str:
        return ontologies[curie.partition(":")[0]].terms[curie].name or ""

    return label


def _statement_cells(statement: Statement, label: Label) -> Row:
    return (
        statement.source,
        statement.antecedent,
        label(statement.antecedent),
        statement.consequent,
        label(statement.consequent),
    )


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
