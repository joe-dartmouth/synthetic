"""`bko.ontology.auto_measures`: exact atoms, hierarchy hits and judged counts against hand rows."""

from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path

from bko.ontology import auto_measures
from bko.ontology.auto_measures import COLUMNS, atoms, hierarchy_hits, judged, precision_recall
from bko.ontology.schema import AtLeast, Codes, FieldMapping
from tests._automatic_fixtures import automatic_ontologies


def _row(when: tuple, concept: str) -> FieldMapping:
    return FieldMapping(when=when, concept=concept, label="x", source="s", limits="l")


def _codes(field: str, *codes: int) -> Codes:
    return Codes(field, frozenset(codes))


def test_a_clause_keys_on_its_sorted_codes_not_its_set_order():
    # 8 and 16 share a hash slot, so the two sets iterate, and repr, in insertion order.
    first, second = frozenset([8, 16]), frozenset([16, 8])
    assert repr(first) != repr(second)
    one = _row((Codes("a", first), AtLeast("b", 2.0)), "HP:1")
    other = _row((Codes("a", second), AtLeast("b", 2.0)), "HP:1")
    assert atoms((one,)) == atoms((other,))
    assert precision_recall((one,), (other,)) == (1, 1)
    ((key, code, concept),) = atoms((one,))
    assert (key, code, concept) == ('[["Codes", "a", [8, 16]], ["AtLeast", "b", 2.0]]', "", "HP:1")


def test_hierarchy_hits_count_an_ancestor_or_descendant_of_the_same_state(tmp_path: Path):
    ontologies = automatic_ontologies(tmp_path)
    hand = (
        _row((_codes("htn", 1),), "MONDO:0005044"),
        _row((_codes("htn", 2),), "MONDO:0001134"),
        _row((_codes("dm", 1),), "MONDO:0005015"),
    )
    built = (
        # A descendant of the hand concept of htn=1, and an ancestor of that of htn=2.
        _row((_codes("htn", 1),), "MONDO:0001134"),
        _row((_codes("htn", 2),), "MONDO:0005044"),
        # The right concept on another state, an exact match, and an unrelated concept.
        _row((_codes("dm", 2),), "MONDO:0001134"),
        _row((_codes("dm", 1),), "MONDO:0005015"),
        _row((_codes("htn", 3),), "HP:0000822"),
    )
    assert hierarchy_hits(built, hand, ontologies) == 2
    assert precision_recall(built, hand) == (Fraction(1, 5), Fraction(1, 3))
    # An exact match is not a hierarchy hit, and a prefix the releases lack never hits.
    assert hierarchy_hits(hand, hand, ontologies) == 0
    assert hierarchy_hits(built, hand, {}) == 0


def test_judged_counts_the_asked_states_and_the_mapped_ones():
    verdicts = {"diag=1": "MONDO:1", "diag=2": "none", "diag=3": "MONDO:3", "a=1": "none"}
    crossed = (_row((_codes("diag", 1),), "MONDO:1"),)
    assert judged(verdicts, crossed) == (3, 1)
    assert judged(verdicts, ()) == (4, 2)


def test_the_command_writes_one_line_per_table(tmp_path: Path, monkeypatch):
    ontologies = automatic_ontologies(tmp_path)
    monkeypatch.setattr(auto_measures, "load_releases", lambda *args, **kwargs: ontologies)
    out = io.StringIO()
    assert auto_measures.main(["--data-dir", str(tmp_path)], out=out) == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert [line.split(",")[0] for line in lines[1:]] == [
        "diabetes-auto-lexical",
        "diabetes-auto-local",
        "diabetes-permuted",
        "diabetes-codes-auto-lexical",
        "diabetes-codes-auto-local",
        "diabetes-codes-permuted",
    ]
