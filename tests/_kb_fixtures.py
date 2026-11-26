"""Offline inputs for the `bko-kb` tests: a toy frame every field of which a row codes, its rows,
two releases over them, a quota and a training frame.

The `MONDO:000000n` and `HP:000001n` ids stand for invented concepts; no stanza copies release
content.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import pandas as pd

from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.gaps.observe import Cell
from bko.ontology.schema import AtLeast, Codes, FieldMapping
from bko.ontology.terms import Ontology, Term


FRAME: Final = FrameSpec(
    name="kb-toy",
    features=("age", "sex", "a", "b", "c"),
    numeric=("age",),
    fractional=(),
    fraction_digits=0,
    categorical=("sex", "a", "b", "c"),
    codebook={
        "sex": {0: "female", 1: "male"},
        "a": {0: "none", 1: "mild", 2: "severe"},
        "b": {0: "no", 1: "yes"},
        "c": {0: "zero", 1: "one", 2: "two", 3: "three"},
    },
    descriptions={
        "age": "age",
        "sex": "sex",
        "a": "a grade",
        "b": "b flag",
        "c": "c level",
        "target": "the condition",
    },
    ranges=(RangeConstraint("age", "a", 20, 90, "toy", "toy"),),
    label_codes=(0, 1),
    label_meanings={0: "absent", 1: "present"},
    edges=(40, 60),
    study="toy",
    cohort="toy patient",
)
"""A frame every field of which a `Codes` clause reads, but `age` and `sex`."""


def kb_row(concept: str, *when: Codes | AtLeast) -> FieldMapping:
    return FieldMapping(when=when, concept=concept, label=concept, source="toy", limits="toy")


BROAD: Final = kb_row("MONDO:0000002", Codes("a", frozenset({1, 2})))
NARROW: Final = kb_row("MONDO:0000003", Codes("a", frozenset({2})))
JOINT: Final = kb_row("HP:0000010", Codes("b", frozenset({1})), Codes("c", frozenset({2, 3})))
ROWS: Final = (BROAD, NARROW, JOINT)
"""Two logic components: the `a` pair, `NARROW` is_a `BROAD`, and the `b`-`c` group alone."""


def _term(curie: str, *parents: str) -> Term:
    return Term(curie, curie, parents, (), (), False, ())


ONTOLOGIES: Final[Mapping[str, Ontology]] = {
    "MONDO": Ontology(
        "mondo",
        "test",
        {
            "MONDO:0000002": _term("MONDO:0000002"),
            "MONDO:0000003": _term("MONDO:0000003", "MONDO:0000002"),
        },
    ),
    "HP": Ontology("hpo", "test", {"HP:0000010": _term("HP:0000010")}),
}

YOUNG_1: Final[Cell] = ("<40", "female", 1)
MIDDLE_0: Final[Cell] = ("40-59", "male", 0)
OLD_1: Final[Cell] = (">=60", "female", 1)
QUOTA: Final[Mapping[Cell, int]] = {YOUNG_1: 40, MIDDLE_0: 40, OLD_1: 40}


def kb_train(ages: Sequence[int], shift: int) -> pd.DataFrame:
    """Training records at `ages`, their codes varied by `shift`, ids from 1."""
    records = [
        {"age": age, "sex": k % 2, "a": (k + shift) % 3, "b": k % 2, "c": (k * shift) % 4}
        | {"target": (k // 2) % 2}
        for k, age in enumerate(ages)
    ]
    frame = pd.DataFrame(records).astype(dict(FRAME.dtypes))
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="record_id")
    return frame
