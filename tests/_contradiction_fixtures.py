"""Planted data contradictions: one small UCI-shaped training split per kind, built in memory.

Each kind is a way records can contradict the knowledge the BKO holds or the structure its stages
assume (ADR `research/data-contradiction-handling`):

- `narrow-without-broad`: records show an `is_a` child concept without its mapped parent.
- `child-share-above-parent`: a cell shows the child more often than the parent, which two rows
  map.
- `empty-cell`: a cell holds no record.
- `all-show`: every record shows every mapped concept.
- `code-overlap`: two concepts' codes on one field intersect, neither inside the other.

Every kind maps a nested pair on `restecg`, as UCI's own table does, and the `is_a` pair on `cp`
and `exang`, all under one invented root in one release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from bko.gaps.observe import Cell
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology, Term
from tests._validate_fixtures import planted
from tests.test_generate_direct import frame


NAME: Final = "uci-test"
"""The dataset's source name every planted split is read under."""

ROOT: Final = "HP:9000000"
BROAD: Final = "HP:9000001"
"""The parent of the `is_a` pair: `exang` 1."""
NARROW: Final = "HP:9000002"
"""The child of the `is_a` pair: `cp` 1."""
OUTER: Final = "HP:9000011"
"""The outer concept of the nested pair: `restecg` 1 or 2."""
INNER: Final = "HP:9000012"
"""The inner concept of the nested pair: `restecg` 2."""
OVERLAP_A: Final = "HP:9000021"
"""`ca` 1 or 2, under `code-overlap`."""
OVERLAP_B: Final = "HP:9000022"
"""`ca` 2 or 3, under `code-overlap`."""

KINDS: Final = (
    "narrow-without-broad",
    "child-share-above-parent",
    "empty-cell",
    "all-show",
    "code-overlap",
)

YW1: Final[Cell] = ("<55", "female", 1)
YW0: Final[Cell] = ("<55", "female", 0)
YM1: Final[Cell] = ("<55", "male", 1)
OM1: Final[Cell] = (">=55", "male", 1)
OM0: Final[Cell] = (">=55", "male", 0)

_AGE_SEX: Final[Mapping[tuple[str, str], tuple[int, int]]] = {
    ("<55", "female"): (40, 0),
    ("<55", "male"): (40, 1),
    (">=55", "male"): (63, 1),
}


@dataclass(frozen=True, slots=True, eq=False)
class Planted:
    """One kind's training split and the knowledge it is read against.

    Attributes:
        kind: The kind.
        frame: The training records, UCI's columns and dtypes, indexed from 1.
        rows: The field mappings.
        ontologies: The one release, by concept prefix.
    """

    kind: str
    frame: pd.DataFrame
    rows: tuple[FieldMapping, ...]
    ontologies: Mapping[str, Ontology]


def row(concept: str, field: str, *codes: int) -> FieldMapping:
    """A mapping row reading `field`'s `codes` as `concept`."""
    return FieldMapping(
        when=(Codes(field, frozenset(codes)),),
        concept=concept,
        label=concept,
        source="test",
        limits="test",
    )


BASE_ROWS: Final = (
    row(OUTER, "restecg", 1, 2),
    row(INNER, "restecg", 2),
    row(NARROW, "cp", 1),
    row(BROAD, "exang", 1),
)
"""The rows every kind maps."""


def ontologies() -> dict[str, Ontology]:
    """One release holding every planted concept: the `is_a` pair, the rest under the root."""
    parents = {
        ROOT: (),
        BROAD: (ROOT,),
        NARROW: (BROAD,),
        OUTER: (ROOT,),
        INNER: (ROOT,),
        OVERLAP_A: (ROOT,),
        OVERLAP_B: (ROOT,),
    }
    terms = {curie: Term(curie, curie, up, (), (), False, ()) for curie, up in parents.items()}
    return {"HP": Ontology("hpo", "test", terms)}


def _cell(cell: Cell, fields: Sequence[Mapping[str, int]]) -> list[dict[str, object]]:
    """One record per entry of `fields` in `cell`, `restecg` cycling 0, 1, 2 unless given."""
    age, sex = _AGE_SEX[str(cell[0]), str(cell[1])]
    return [
        planted(
            **{
                "age": age + index,
                "sex": sex,
                "target": int(cell[2]),
                "restecg": index % 3,
                "thal": 3,
                "thalach": 120 + 5 * index,
                **values,
            }
        )
        for index, values in enumerate(fields)
    ]


def _records(kind: str) -> list[dict[str, object]]:
    neither = {"cp": 4, "exang": 0}
    both = {"cp": 1, "exang": 1}
    match kind:
        case "narrow-without-broad":
            cells = {
                YW1: [{"cp": 1, "exang": 0}] * 6,
                YM1: [both] * 6,
                OM1: [{"cp": 4, "exang": 1}] * 6,
                YW0: [neither] * 6,
                OM0: [neither] * 6,
            }
        case "child-share-above-parent":
            cells = {
                YW1: [{"cp": 1, "exang": 0}, both],
                YM1: [{"cp": 1, "exang": 0, "thal": 7}] * 3 + [neither] * 3,
                OM1: [{"cp": 4, "exang": 1}] * 6,
                YW0: [neither] * 6,
                OM0: [neither] * 6,
            }
        case "empty-cell":
            cells = {
                YM1: [both] * 6,
                OM1: [{"cp": 4, "exang": 1}] * 6,
                YW0: [{"cp": 1, "exang": 0}] * 6,
                OM0: [neither] * 6,
            }
        case "all-show":
            shows = {"cp": 1, "exang": 1, "restecg": 2}
            cells = {cell: [shows] * 6 for cell in (YW1, YM1, OM1, YW0, OM0)}
        case "code-overlap":
            cycle = [{**(both if i % 2 else neither), "ca": i % 4} for i in range(6)]
            cells = dict.fromkeys((YW1, YM1, OM1, YW0, OM0), cycle)
        case _:
            msg = f"no planted kind {kind!r}; the kinds are {', '.join(KINDS)}"
            raise ValueError(msg)
    return [record for cell, fields in cells.items() for record in _cell(cell, fields)]


def _rows(kind: str) -> tuple[FieldMapping, ...]:
    if kind == "child-share-above-parent":
        return (*BASE_ROWS, row(BROAD, "thal", 7))
    if kind == "code-overlap":
        return (*BASE_ROWS, row(OVERLAP_A, "ca", 1, 2), row(OVERLAP_B, "ca", 2, 3))
    return BASE_ROWS


def plant(kind: str) -> Planted:
    """The training split, rows and release that plant `kind`.

    Raises:
        ValueError: If `kind` is not one of `KINDS`.
    """
    return Planted(kind, frame(_records(kind)), _rows(kind), ontologies())
