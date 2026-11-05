"""The rules each contradiction kind reads the rows and releases by, apart from any records."""

from __future__ import annotations

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.data import SYNTHEA_FIELD_ROWS
from bko.evaluate.diabetes import DIABETES_FIELD_ROWS
from bko.gaps.contradictions import KINDS, Finding, code_overlaps, is_a_pairs
from bko.gaps.shares import label_only, nested
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import AtLeast, Codes, FieldMapping
from bko.ontology.terms import Ontology, Term
from tests._contradiction_fixtures import KINDS as PLANTED_KINDS
from tests._contradiction_fixtures import row


def overlap(first: str, second: str, field: str, *codes: int) -> Finding:
    return Finding("code-overlap", concept=first, parent=second, field=field, codes=codes)


def mapped(concept: str, *when: Codes | AtLeast) -> FieldMapping:
    return FieldMapping(when=when, concept=concept, label=concept, source="test", limits="test")


def test_the_kinds_are_the_planted_kinds_in_order():
    assert KINDS == PLANTED_KINDS


def test_a_pair_nested_resolves_is_no_overlap_but_equal_codes_are():
    rows = (row("A", "f", 1, 2), row("B", "f", 2), row("C", "f", 1), row("C", "f", 2))
    assert nested(rows) == {"B": "A"}
    assert code_overlaps(rows) == (overlap("A", "C", "f", 1, 2), overlap("B", "C", "f", 2))


def test_a_concept_of_two_rows_overlaps_one_its_codes_hold():
    rows = (row("A", "f", 1), row("A", "g", 1), row("B", "f", 1, 2, 3))
    assert nested(rows) == {}
    assert code_overlaps(rows) == (overlap("A", "B", "f", 1),)


def test_codes_unite_over_rows_and_come_out_ascending():
    rows = (row("B", "f", 3, 1), row("A", "f", 9), row("A", "f", 3, 1, 4))
    assert code_overlaps(rows) == (overlap("A", "B", "f", 1, 3),)


def test_disjoint_codes_overlap_nowhere():
    assert code_overlaps((row("A", "f", 1), row("B", "f", 2), row("C", "g", 1))) == ()


def test_a_rows_other_clauses_are_ignored():
    rows = (
        mapped("A", AtLeast("g", 1.0), Codes("f", frozenset({1, 2}))),
        mapped("B", Codes("f", frozenset({2, 3}))),
        mapped("C", AtLeast("f", 0.0)),
    )
    assert code_overlaps(rows) == (overlap("A", "B", "f", 2),)


def test_a_rows_codes_clauses_on_one_field_intersect():
    rows = (
        mapped("A", Codes("f", frozenset({1, 2})), Codes("f", frozenset({2, 3}))),
        row("B", "f", 1),
        row("C", "f", 2),
    )
    assert code_overlaps(rows) == (overlap("A", "C", "f", 2),)


def test_overlaps_come_by_field_then_by_concept():
    rows = (row("D", "g", 1), row("C", "g", 1), row("B", "f", 1), row("A", "f", 1))
    assert code_overlaps(rows) == (overlap("A", "B", "f", 1), overlap("C", "D", "g", 1))


def test_rows_nesting_in_a_chain_are_refused():
    with pytest.raises(ValueError, match="only one level of nesting"):
        code_overlaps((row("A", "f", 1, 2, 3), row("B", "f", 1, 2), row("C", "f", 1)))


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            FIELD_MAPPINGS,
            (
                overlap("HP:0001681", "HP:0100749", "cp", 1),
                overlap("HP:0005145", "MONDO:0006715", "target", 1),
            ),
        ),
        (SYNTHEA_FIELD_ROWS, ()),
        (DIABETES_FIELD_ROWS, ()),
    ],
    ids=["uci", "synthea", "diabetes"],
)
def test_each_frames_table_overlaps_where_it_does(rows, expected):
    """UCI's `HP:0012250` row reads a threshold beside its codes; that clause is ignored."""
    assert code_overlaps(rows) == expected


def test_only_uci_holds_a_row_of_more_than_one_clause():
    tables = (FIELD_MAPPINGS, SYNTHEA_FIELD_ROWS, DIABETES_FIELD_ROWS)
    several = [r.concept for rows in tables for r in rows if len(r.when) > 1]
    assert several == ["HP:0012250"]


def test_a_cycle_of_is_a_pairs_no_concept_with_itself():
    terms = {
        "HP:1": Term("HP:1", "HP:1", ("HP:2",), (), (), False, ()),
        "HP:2": Term("HP:2", "HP:2", ("HP:1",), (), (), False, ()),
    }
    ontologies = {"HP": Ontology("hpo", "test", terms)}
    assert is_a_pairs(["HP:1", "HP:2"], ontologies) == (("HP:1", "HP:2"), ("HP:2", "HP:1"))


def test_a_row_with_no_clause_does_not_follow_the_label():
    rows = (mapped("A"), row("B", UCI_FRAME.label, 1), mapped("C"), row("C", UCI_FRAME.label, 1))
    assert label_only(rows, frame=UCI_FRAME) == frozenset({"B"})
