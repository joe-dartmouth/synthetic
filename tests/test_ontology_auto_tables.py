"""The committed automatic and permuted tables, read with no data (`bko.ontology.auto_tables`).

Each table's rows hash to the digest ADR `research/automatic-mapping` states, each permuted table
moves every hand row's concept, the split verdict table's parts merge only when they agree, and
the ADR's reversal condition does not hold: no `auto-lexical` code row names another concept than
the hand code row of its state.
"""

from __future__ import annotations

import hashlib
import json
import types
from typing import Final

import pytest

from bko.ontology.auto_measures import precision_recall
from bko.ontology.auto_tables import (
    AUTO_LEXICAL_CODES_ROWS,
    AUTO_LEXICAL_DIABETES_ROWS,
    AUTO_LOCAL_CODES_ROWS,
    AUTO_LOCAL_DIABETES_ROWS,
    PERMUTED_CODES_ROWS,
    PERMUTED_DIABETES_ROWS,
    merged,
)
from bko.ontology.automatic import state, unmovable
from bko.ontology.code_frames import DIABETES_CODES_FIELD_ROWS
from bko.ontology.frames import DIABETES_FIELD_ROWS
from bko.ontology.schema import Codes, FieldMapping, rows_digest


DIGESTS: Final = {
    "auto-lexical diabetes": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "auto-lexical diabetes-codes": (
        "bf119a4d05428defc250c070b551d232f6a6d5f8a2421d84d01490fcd606b5c2"
    ),
    "auto-local diabetes": "7a52217e6d908c456cfa13d60811182fb047f16caf00e2eb1b295b596b0b16f0",
    "auto-local diabetes-codes": "ab632685d5bf2995b580bd09e071dbd863e79319a862d500ee95e0ab935b9109",
    "permuted diabetes": "b1a0cc61c9daaee98d7507d47ad41536ffd31f5678f8efa5c429937c5fde783b",
    "permuted diabetes-codes": "8f90fe5a70fe83467f7bad88d61754a74cdc4313cdfdc17b02328ca88eb6775c",
}
"""Each table's `rows_digest`, as ADR `research/automatic-mapping` states it."""

TABLES: Final[dict[str, tuple[FieldMapping, ...]]] = {
    "auto-lexical diabetes": AUTO_LEXICAL_DIABETES_ROWS,
    "auto-lexical diabetes-codes": AUTO_LEXICAL_CODES_ROWS,
    "auto-local diabetes": AUTO_LOCAL_DIABETES_ROWS,
    "auto-local diabetes-codes": AUTO_LOCAL_CODES_ROWS,
    "permuted diabetes": PERMUTED_DIABETES_ROWS,
    "permuted diabetes-codes": PERMUTED_CODES_ROWS,
}

PERMUTED: Final = {
    "diabetes": (PERMUTED_DIABETES_ROWS, DIABETES_FIELD_ROWS),
    "diabetes-codes": (PERMUTED_CODES_ROWS, DIABETES_CODES_FIELD_ROWS),
}


@pytest.mark.parametrize("table", sorted(TABLES))
def test_each_tables_rows_hash_to_their_stated_digest(table: str):
    assert rows_digest(TABLES[table]) == DIGESTS[table]


@pytest.mark.parametrize("dataset", sorted(PERMUTED))
def test_each_permuted_table_moves_every_movable_concept(dataset: str):
    shuffled, hand = PERMUTED[dataset]
    assert unmovable(hand) == 0
    assert [row.when for row in shuffled] == [row.when for row in hand]
    assert [
        row.concept
        for row, original in zip(shuffled, hand, strict=True)
        if row.concept == original.concept
    ] == []
    assert precision_recall(shuffled, hand) == (0, 0)


def test_no_lexical_code_row_names_another_concept_than_its_hand_row():
    hand: dict[tuple[str, int], set[str]] = {}
    for row in DIABETES_CODES_FIELD_ROWS:
        for clause in row.when:
            if isinstance(clause, Codes):
                for code in clause.codes:
                    hand.setdefault((clause.field, code), set()).add(row.concept)
    held = [row for row in AUTO_LEXICAL_CODES_ROWS if state(row) in hand]
    assert held
    assert [row for row in held if row.concept not in hand[state(row)]] == []
    assert precision_recall(AUTO_LEXICAL_CODES_ROWS, DIABETES_CODES_FIELD_ROWS)[0] == 1


def _part(name: str, part: str | None, verdicts: dict, labels: dict) -> types.ModuleType:
    module = types.ModuleType(name)
    named = {"DATASET": "toy"} if part is None else {"DATASET": "toy", "PART": part}
    vars(module).update(named, VERDICTS=verdicts, LABELS=labels)
    return module


def test_the_parts_merge_when_they_agree():
    low = _part("low", "low", {"a=1": "HP:1"}, {"HP:1": "one"})
    high = _part("high", "high", {"b=1": "HP:1", "b=2": "none"}, {"HP:1": "one"})
    verdicts, labels = merged((low, high), dataset="toy", parts=("low", "high"))
    assert verdicts == {"a=1": "HP:1", "b=1": "HP:1", "b=2": "none"}
    assert labels == {"HP:1": "one"}


@pytest.mark.parametrize(
    ("high", "parts", "dataset"),
    [
        (_part("high", "high", {"a=1": "none"}, {}), ("low", "high"), "toy"),
        (_part("high", "high", {"b=1": "HP:1"}, {"HP:1": "uno"}), ("low", "high"), "toy"),
        (_part("high", "low", {"b=1": "none"}, {}), ("low", "high"), "toy"),
        (_part("high", "high", {"b=1": "none"}, {}), ("low", "high"), "other"),
        (_part("high", None, {"b=1": "none"}, {}), ("low", "high"), "toy"),
    ],
)
def test_parts_that_overlap_relabel_or_misname_are_refused(high, parts, dataset):
    low = _part("low", "low", {"a=1": "HP:1"}, {"HP:1": "one"})
    with pytest.raises(ValueError, match=r"names dataset and part|repeats states"):
        merged((low, high), dataset=dataset, parts=parts)


def test_rows_digest_sorts_a_clauses_codes():
    def row(codes: frozenset[int]) -> FieldMapping:
        return FieldMapping(
            when=(Codes("a", codes),), concept="HP:1", label="x", source="s", limits="l"
        )

    assert list(frozenset({1024, 1})) == [1024, 1]
    document = [
        {
            "when": [["Codes", "a", [1, 1024]]],
            "concept": "HP:1",
            "label": "x",
            "source": "s",
            "limits": "l",
        }
    ]
    expected = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    assert rows_digest((row(frozenset({1024, 1})),)) == expected
    assert rows_digest((row(frozenset([16, 8])),)) == rows_digest((row(frozenset([8, 16])),))
