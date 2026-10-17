"""Command line for gap detection: `python -m bko.gaps uci`.

`uci` fetches UCI Heart Disease and the releases its field mappings name into the data root if
they are absent, splits the records, and lists both kinds of gap in the training split: the
ontology statements its records leave ungrounded, then the cells holding fewer training records
than the threshold. It writes one CSV to stdout, the same bytes on every run over the same inputs,
and nothing unless the whole report is built; each input's identity, the BKO's size, each list's
length and the report's digest go to stderr through logging. `docs/DEVELOPMENT.md` (Gap
detection) is the usage home.

Exit codes: 0 the report is written; 1 an unexpected error, with its traceback; 2 the command could
not run; 3 the data or a release contradicts its pin or schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from bko.bkb.structure import BKB
from bko.datasets.cells import CELL_COLUMNS, UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.datasets.split import split
from bko.datasets.uci import load_uci
from bko.gaps.detect import Statement, count_gaps, detect, ontology_fragments
from bko.gaps.errors import GapError
from bko.gaps.observe import Observed, dataset_fragment, observe_records
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


@dataclass(frozen=True, slots=True)
class Context:
    """What the commands reach outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        releases: The pinned release for each concept prefix.
        field_rows: The UCI field mappings.
        code_rows: The Synthea code mappings.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    releases: Mapping[str, Source] = RELEASES
    field_rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS
    code_rows: tuple[CodeMapping, ...] = CODE_MAPPINGS


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
        command = {"uci": _uci}[args.command]
        rows = command(data_dir, context or Context())
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
    (out or sys.stdout).write(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("gaps rows=%d sha256=%s", len(rows) - 1, digest)
    return 0


def configure_logging() -> None:
    """Send INFO and higher log records to stderr, keeping stdout for the report."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


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
        "uci", parents=[common], help="list both kinds of gap in the UCI training split"
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
    observed = observe_records(parts.train, context.field_rows, name=context.uci.name)
    _log_observed(observed)
    gaps = detect(fragments, dataset_fragment(observed))
    logger.info("bko dataset=%s ungrounded=%d", observed.name, len(gaps))
    cells = count_gaps(cell_counts(parts.train))
    logger.info(
        "count dataset=%s flagged=%d below=%d", observed.name, len(cells), UNDERREPRESENTED_BELOW
    )
    label = _labeller(ontologies)
    rows: list[Row] = [UCI_COLUMNS]
    rows.extend(("bko", *_statement(gap, label), "", "", "", "") for gap in gaps)
    rows.extend(
        ("count", "", "", "", "", "", cell.age_band, cell.sex, str(cell.target), str(cell.count))
        for cell in cells
    )
    return rows


def _ontology(
    concepts: Collection[str], data_dir: Path, context: Context
) -> tuple[dict[str, Ontology], tuple[BKB, ...]]:
    prefixes = sorted({concept.partition(":")[0] for concept in concepts})
    unpinned = [prefix for prefix in prefixes if prefix not in context.releases]
    if unpinned:
        msg = f"no pinned release for the prefix {unpinned[0]}"
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


def _labeller(ontologies: Mapping[str, Ontology]) -> Label:
    def label(curie: str) -> str:
        return ontologies[curie.partition(":")[0]].terms[curie].name or ""

    return label


def _statement(statement: Statement, label: Label) -> Row:
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
