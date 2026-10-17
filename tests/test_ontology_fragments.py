"""`bko.ontology.fragments`: loaded releases as BKB fragments."""

from __future__ import annotations

from pathlib import Path

import pytest

from bko.bkb.serialize import render
from bko.bkb.structure import BKB, SNode
from bko.ontology.errors import ScopeError
from bko.ontology.fragments import INDIVIDUAL, exact_match_fragment, is_a_fragment
from bko.ontology.obo import parse_obo
from bko.ontology.terms import Ontology
from tests._ontology_fixtures import (
    HPO_LABELS,
    HPO_TEXT,
    MONDO_TEXT,
    SANTOS_DO_TEXT,
    SANTOS_MONDO_TEXT,
    write_obo,
)


pytestmark = pytest.mark.filterwarnings("error")

MONDO_SCOPE = (
    "MONDO:0006960",
    "MONDO:0001543",
    "MONDO:0001397",
    "MONDO:0002121",
    "MONDO:0002122",
    "MONDO:0021166",
)
DO_SCOPE = ("DOID:11446", "DOID:12528", "DOID:9473", "DOID:1188", "DOID:1802")


def load(directory: Path, text: str, name: str) -> Ontology:
    return parse_obo(write_obo(directory, text, f"{name}.obo"), name=name)


def edge(head: str, tail: str) -> SNode:
    return SNode((head, INDIVIDUAL), 1.0, frozenset({(tail, INDIVIDUAL)}))


def bridge(directory: Path, mondo_text: str = SANTOS_MONDO_TEXT) -> BKB:
    mondo, do = load(directory, mondo_text, "mondo"), load(directory, SANTOS_DO_TEXT, "do")
    return exact_match_fragment(
        mondo, MONDO_SCOPE, do, DO_SCOPE, subject_prefix="MONDO", target_prefix="DOID"
    )


def test_each_scoped_is_a_edge_is_one_certain_snode(tmp_path):
    mondo = load(tmp_path, SANTOS_MONDO_TEXT, "mondo")
    fragment = is_a_fragment(mondo, MONDO_SCOPE, prefix="MONDO")
    assert fragment.name == "mondo"
    assert fragment.inodes == tuple((curie, INDIVIDUAL) for curie in sorted(MONDO_SCOPE))
    assert fragment.snodes == (
        edge("MONDO:0006960", "MONDO:0001543"),
        edge("MONDO:0001397", "MONDO:0002121"),
        edge("MONDO:0002122", "MONDO:0002121"),
        edge("MONDO:0021166", "MONDO:0002122"),
        edge("MONDO:0001397", "MONDO:0006960"),
    )


def test_do_gives_its_four_edges(tmp_path):
    fragment = is_a_fragment(load(tmp_path, SANTOS_DO_TEXT, "do"), DO_SCOPE, prefix="DOID")
    assert fragment.snodes == (
        edge("DOID:12528", "DOID:11446"),
        edge("DOID:9473", "DOID:12528"),
        edge("DOID:1188", "DOID:1802"),
        edge("DOID:1802", "DOID:9473"),
    )


def test_the_same_converter_builds_an_hpo_fragment(tmp_path):
    text = HPO_TEXT.replace(
        "id: HP:0001681\nname: Angina pectoris\n",
        "id: HP:0001681\nname: Angina pectoris\nis_a: HP:0100749 ! Chest pain\n",
    )
    fragment = is_a_fragment(load(tmp_path, text, "hpo"), HPO_LABELS, prefix="HP")
    assert fragment.name == "hpo"
    assert len(fragment.inodes) == len(HPO_LABELS)
    assert fragment.snodes == (edge("HP:0100749", "HP:0001681"),)


def test_scope_order_does_not_change_the_fragment(tmp_path):
    mondo = load(tmp_path, SANTOS_MONDO_TEXT, "mondo")
    forward = is_a_fragment(mondo, MONDO_SCOPE, prefix="MONDO")
    backward = is_a_fragment(mondo, MONDO_SCOPE[::-1], prefix="MONDO", name="mondo")
    assert render(forward) == render(backward)


def test_two_scoped_children_make_the_fragment_non_mutex(tmp_path):
    mondo = load(tmp_path, SANTOS_MONDO_TEXT, "mondo")
    assert is_a_fragment(mondo, MONDO_SCOPE, prefix="MONDO").non_mutex_pair() == (1, 4)


@pytest.mark.parametrize(
    ("scope", "match"),
    [
        ((), "is empty"),
        (("HP:0000822",), "HP:0000822 is not a MONDO term"),
        (("MONDO:0000001",), "MONDO:0000001 is absent in mondo releases/2026-10-06"),
        (("MONDO:9999998",), "MONDO:9999998 is obsolete in mondo releases/2026-10-06"),
    ],
)
def test_a_scope_the_release_cannot_supply_is_refused(tmp_path, scope, match):
    with pytest.raises(ScopeError, match=match):
        is_a_fragment(load(tmp_path, MONDO_TEXT, "mondo"), scope, prefix="MONDO")


def test_the_bridge_links_each_exact_match_both_ways(tmp_path):
    fragment = bridge(tmp_path)
    assert fragment.name == "bridge"
    assert fragment.snodes == (
        edge("DOID:1188", "MONDO:0001397"),
        edge("MONDO:0001397", "DOID:1188"),
        edge("DOID:12528", "MONDO:0001543"),
        edge("MONDO:0001543", "DOID:12528"),
        edge("DOID:1802", "MONDO:0002121"),
        edge("MONDO:0002121", "DOID:1802"),
        edge("DOID:11446", "MONDO:0006960"),
        edge("MONDO:0006960", "DOID:11446"),
    )


@pytest.mark.parametrize(
    "replacement", ["property_value: skos:closeMatch DOID:1802", "xref: DOID:1802"]
)
def test_the_bridge_reads_only_exact_matches(tmp_path, replacement):
    text = SANTOS_MONDO_TEXT.replace("property_value: skos:exactMatch DOID:1802", replacement)
    heads = {snode.head[0] for snode in bridge(tmp_path, text).snodes}
    assert "DOID:1802" not in heads
    assert len(heads) == 6


def test_a_bridge_within_one_prefix_is_refused(tmp_path):
    mondo = load(tmp_path, SANTOS_MONDO_TEXT, "mondo")
    with pytest.raises(ScopeError, match="links two releases"):
        exact_match_fragment(
            mondo, MONDO_SCOPE, mondo, MONDO_SCOPE, subject_prefix="MONDO", target_prefix="MONDO"
        )


def test_a_repeated_exact_match_gives_one_pair_of_snodes(tmp_path):
    line = "property_value: skos:exactMatch DOID:1802"
    text = SANTOS_MONDO_TEXT.replace(line, f"{line}\n{line}")
    assert bridge(tmp_path, text).snodes == bridge(tmp_path).snodes


def test_the_bridge_ignores_the_targets_matches_and_out_of_scope_classes(tmp_path):
    mondo, do = load(tmp_path, SANTOS_MONDO_TEXT, "mondo"), load(tmp_path, SANTOS_DO_TEXT, "do")
    reverse = exact_match_fragment(
        do, DO_SCOPE, mondo, MONDO_SCOPE, subject_prefix="DOID", target_prefix="MONDO"
    )
    assert reverse.snodes == ()
    narrow = exact_match_fragment(
        mondo, MONDO_SCOPE, do, ("DOID:1188",), subject_prefix="MONDO", target_prefix="DOID"
    )
    assert {snode.head[0] for snode in narrow.snodes} == {"DOID:1188", "MONDO:0001397"}


def test_a_bad_target_scope_is_refused(tmp_path):
    mondo, do = load(tmp_path, SANTOS_MONDO_TEXT, "mondo"), load(tmp_path, SANTOS_DO_TEXT, "do")
    with pytest.raises(ScopeError, match="DOID:0000000 is absent in do"):
        exact_match_fragment(
            mondo, MONDO_SCOPE, do, ("DOID:0000000",), subject_prefix="MONDO", target_prefix="DOID"
        )
