"""`ancestors` and `scope`: the classes a release's fragment holds for some mapped concepts."""

from __future__ import annotations

import pytest

from bko.ontology.errors import ScopeError
from bko.ontology.fragments import INDIVIDUAL, is_a_fragment, scope
from bko.ontology.obo import parse_obo
from bko.ontology.terms import Ontology, Term, ancestors
from tests._gaps_fixtures import LADDER_MONDO
from tests._ontology_fixtures import write_obo


def term(curie: str, *parents: str) -> Term:
    return Term(
        curie=curie,
        name=None,
        parents=parents,
        xrefs=(),
        matches=(),
        obsolete=False,
        replaced_by=(),
    )


DIAMOND = Ontology(
    name="mondo",
    version="test",
    terms={
        t.curie: t
        for t in (
            term("MONDO:1"),
            term("MONDO:2", "MONDO:1"),
            term("MONDO:3", "MONDO:1"),
            term("MONDO:4", "MONDO:2", "MONDO:3"),
            term("MONDO:5", "MONDO:4", "BFO:9"),
            term("BFO:9", "MONDO:6"),
            term("MONDO:6"),
        )
    },
)
"""A diamond under MONDO:4, and a BFO parent of MONDO:5 that re-enters the prefix at MONDO:6."""


def test_ancestors_follow_is_a_to_every_root_and_count_a_diamond_once():
    expected = {"MONDO:1", "MONDO:2", "MONDO:3", "MONDO:4", "BFO:9", "MONDO:6"}
    assert ancestors(DIAMOND, "MONDO:5") == expected


def test_ancestors_with_a_prefix_follow_only_parents_that_carry_it():
    expected = {"MONDO:1", "MONDO:2", "MONDO:3", "MONDO:4"}
    assert ancestors(DIAMOND, "MONDO:5", prefix="MONDO") == expected


def test_a_parent_the_release_lacks_ends_its_branch():
    lacking = Ontology(name="mondo", version="test", terms={"MONDO:1": term("MONDO:1", "MONDO:0")})
    assert ancestors(lacking, "MONDO:1") == {"MONDO:0"}


def test_scope_holds_the_concepts_with_the_prefix_and_their_own_prefix_ancestors():
    expected = {"MONDO:1", "MONDO:2", "MONDO:3", "MONDO:4", "MONDO:5"}
    assert scope(DIAMOND, {"MONDO:5", "HP:1"}, prefix="MONDO") == expected


def test_scope_is_empty_when_no_concept_carries_the_prefix():
    assert scope(DIAMOND, {"HP:1"}, prefix="MONDO") == frozenset()


def test_the_prefix_filter_is_what_lets_is_a_fragment_build(tmp_path):
    mondo = parse_obo(write_obo(tmp_path, LADDER_MONDO), name="mondo")
    unfiltered = {"MONDO:9000003", *ancestors(mondo, "MONDO:9000003")}
    with pytest.raises(ScopeError, match="BFO:0000016 is not a MONDO term"):
        is_a_fragment(mondo, unfiltered, prefix="MONDO")
    classes = scope(mondo, {"MONDO:9000003"}, prefix="MONDO")
    fragment = is_a_fragment(mondo, classes, prefix="MONDO")
    assert [(sorted(s.tail), s.head) for s in fragment.snodes] == [
        ([("MONDO:9000002", INDIVIDUAL)], ("MONDO:9000001", INDIVIDUAL)),
        ([("MONDO:9000003", INDIVIDUAL)], ("MONDO:9000002", INDIVIDUAL)),
    ]
