"""How far each automatic or permuted mapping table is from the hand rows.

`precision_recall` counts exact (field, code, concept) atoms; `hierarchy_hits` counts, apart from
them, the built atoms whose concept is a strict `is_a` ancestor or descendant of a hand concept of
the same state (ADR `research/automatic-mapping`, Decision 7). The `measures` command writes one
CSV line per table of `bko.ontology.auto_tables` against `DIABETES_FIELD_ROWS` or
`DIABETES_CODES_FIELD_ROWS`, reading the pinned releases for the hierarchy:

    python -m bko.ontology.auto_measures [--data-dir <dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Final, TextIO

from bko.core.logs import configure
from bko.datasets.diabetes import DIAGNOSES
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, open_https
from bko.ontology import auto_tables
from bko.ontology.automatic import NONE, state, unmovable
from bko.ontology.code_frames import DIABETES_CODES_FIELD_ROWS
from bko.ontology.frames import DIABETES_FIELD_ROWS
from bko.ontology.mapping import RELEASES
from bko.ontology.releases import load_releases
from bko.ontology.schema import Codes, FieldMapping, reads
from bko.ontology.terms import Ontology, ancestors


logger = logging.getLogger("bko.ontology.auto_measures")

type Atom = tuple[str, str, str]
"""(field, code, concept); a row that is not one `Codes` clause is one atom keyed by its clauses."""

COLUMNS: Final = (
    "table",
    "rows",
    "precision",
    "recall",
    "exact",
    "hierarchy",
    "unmovable",
    "judged",
    "accepted",
)
"""The `measures` CSV's header."""


def atoms(rows: Sequence[FieldMapping]) -> frozenset[Atom]:
    """Each row's atoms, as `precision_recall` counts them.

    A row of one `Codes` clause gives one atom per code; any other row gives one atom, keyed on
    each clause's type, field and what it reads (`bko.ontology.schema.reads`: its codes sorted),
    so two equal clauses key alike whatever order their codes were added in.
    """
    found: set[Atom] = set()
    for row in rows:
        match row.when:
            case (Codes() as clause,):
                found.update((clause.field, str(code), row.concept) for code in clause.codes)
            case _:
                key = json.dumps([[type(c).__name__, c.field, reads(c)] for c in row.when])
                found.add((key, "", row.concept))
    return frozenset(found)


def precision_recall(
    built: Sequence[FieldMapping], hand: Sequence[FieldMapping]
) -> tuple[Fraction | None, Fraction | None]:
    """The share of `built`'s atoms `hand` holds, and of `hand`'s that `built` holds.

    Each share is None when its denominator is empty.
    """
    mine, theirs = atoms(built), atoms(hand)
    common = len(mine & theirs)
    return (
        Fraction(common, len(mine)) if mine else None,
        Fraction(common, len(theirs)) if theirs else None,
    )


def hierarchy_hits(
    built: Sequence[FieldMapping], hand: Sequence[FieldMapping], ontologies: Mapping[str, Ontology]
) -> int:
    """How many of `built`'s atoms miss `hand` but sit on a hand concept's `is_a` line.

    Such an atom names a strict ancestor or descendant of a hand concept of the same
    (field, code).
    Ancestry is read within the release of the concepts' shared prefix, following only that
    prefix's parents; concepts of two prefixes, or of a prefix `ontologies` lacks, never hit.
    """
    theirs = atoms(hand)
    by_state: dict[tuple[str, str], set[str]] = {}
    for field, code, concept in theirs:
        by_state.setdefault((field, code), set()).add(concept)
    return sum(
        any(_related(concept, other, ontologies) for other in by_state.get((field, code), ()))
        for field, code, concept in atoms(built) - theirs
    )


def _related(one: str, other: str, ontologies: Mapping[str, Ontology]) -> bool:
    prefix = one.partition(":")[0]
    ontology = ontologies.get(prefix)
    if ontology is None or other.partition(":")[0] != prefix:
        return False
    return one in ancestors(ontology, other, prefix=prefix) or other in ancestors(
        ontology, one, prefix=prefix
    )


def judged(verdicts: Mapping[str, str], crossed: Sequence[FieldMapping]) -> tuple[int, int]:
    """How many of `verdicts`' states the judge was asked about, and how many it mapped.

    The asked states are every key but those `crossed`, the cross-reference rows, map.
    """
    skipped = {"{}={}".format(*state(row)) for row in crossed}
    asked = [concept for key, concept in verdicts.items() if key not in skipped]
    return len(asked), sum(concept != NONE for concept in asked)


def measures(ontologies: Mapping[str, Ontology]) -> list[dict[str, str]]:
    """One `COLUMNS` record per committed table of `bko.ontology.auto_tables`."""
    crossed = [row for row in auto_tables.AUTO_LEXICAL_CODES_ROWS if state(row)[0] in DIAGNOSES]
    local = {
        "diabetes-auto-local": judged(auto_tables.DIABETES_VERDICTS, ()),
        "diabetes-codes-auto-local": judged(auto_tables.CODES_VERDICTS, crossed),
    }
    tables = (
        ("diabetes-auto-lexical", auto_tables.AUTO_LEXICAL_DIABETES_ROWS, DIABETES_FIELD_ROWS),
        ("diabetes-auto-local", auto_tables.AUTO_LOCAL_DIABETES_ROWS, DIABETES_FIELD_ROWS),
        ("diabetes-permuted", auto_tables.PERMUTED_DIABETES_ROWS, DIABETES_FIELD_ROWS),
        (
            "diabetes-codes-auto-lexical",
            auto_tables.AUTO_LEXICAL_CODES_ROWS,
            DIABETES_CODES_FIELD_ROWS,
        ),
        ("diabetes-codes-auto-local", auto_tables.AUTO_LOCAL_CODES_ROWS, DIABETES_CODES_FIELD_ROWS),
        ("diabetes-codes-permuted", auto_tables.PERMUTED_CODES_ROWS, DIABETES_CODES_FIELD_ROWS),
    )
    records = []
    for name, built, hand in tables:
        precision, recall = precision_recall(built, hand)
        asked, accepted = local.get(name, (None, None))
        records.append(
            {
                "table": name,
                "rows": str(len(built)),
                "precision": str(precision),
                "recall": str(recall),
                "exact": str(len(atoms(built) & atoms(hand))),
                "hierarchy": str(hierarchy_hits(built, hand, ontologies)),
                "unmovable": str(unmovable(hand)) if name.endswith("-permuted") else "",
                "judged": "" if asked is None else str(asked),
                "accepted": "" if accepted is None else str(accepted),
            }
        )
    return records


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    open_url: UrlOpener = open_https,
) -> int:
    """Write `measures`' CSV to `out` (stdout when omitted), fetching the releases if absent.

    Returns:
        0 done; a `DatasetError`'s exit code when a release cannot be read; 2 another I/O failure.
    """
    parser = argparse.ArgumentParser(
        prog="python -m bko.ontology.auto_measures", description=__doc__
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = default_data_dir() if args.data_dir is None else args.data_dir.resolve()
    try:
        ontologies = load_releases(RELEASES, root, open_url=open_url)
    except DatasetError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    writer = csv.DictWriter(sys.stdout if out is None else out, COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(measures(ontologies))
    return 0


if __name__ == "__main__":
    configure()
    sys.exit(main())
