"""The BKB structure: pybkb's API tests ported, and the rules pybkb did not check.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-16; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from bko.bkb.errors import StructureError
from bko.bkb.structure import BKB, SNode
from tests._bkb_fixtures import build, document_names, exported, manifest, simple_build


pytestmark = pytest.mark.filterwarnings("error")


def test_a_simple_build_equals_pybkbs_saved_bkb():
    assert simple_build() == exported("test_api_bkb1")
    assert exported("test_api_bkb1") == simple_build()


def test_a_simple_build_holds_its_snodes_in_order():
    assert simple_build().snodes == (
        SNode(("A", "1"), 0.003, frozenset()),
        SNode(("1", "1"), 0.9, frozenset({("A", "1")})),
        SNode(("1", "B"), 0.56, frozenset()),
        SNode(("A", "2"), 0.97, frozenset({("1", "1")})),
    )


@pytest.mark.parametrize(
    ("prob", "tail_subset", "count"),
    [(None, (), 2), (None, [("pH_Value", "Normal")], 1), (0.1, [("pH_Value", "Normal")], 1)],
)
def test_find_snodes_matches_pybkb_on_aquatic_eco(prob, tail_subset, count):
    found = exported("aquatic_eco").find_snodes(
        "ReproduceAbility", "Low", prob=prob, tail_subset=tail_subset
    )
    assert len(found) == count


def test_find_snodes_filters_on_a_zero_probability_and_a_tail():
    bkb = build("zero", [("A", "1", 0.0, []), ("A", "1", 0.5, [("B", "1")])])
    assert bkb.find_snodes("A", "1", prob=0.0) == [0]
    assert bkb.find_snodes("A", "1", tail_subset=[("B", "1")]) == [1]
    assert bkb.find_snodes("A", "1") == [0, 1]


def test_find_snodes_refuses_an_unknown_head():
    with pytest.raises(StructureError, match="not an I-node"):
        simple_build().find_snodes("A", "3")


def test_pybkbs_saved_bkb_is_mutex():
    assert exported("test_api_bkb1").is_mutex()


NON_MUTEX: dict[str, list[Any]] = {
    "two priors": [("1", "T", 0.3, []), ("1", "T", 0.4, [])],
    "a prior and a conditional": [("1", "T", 0.3, [("2", "T")]), ("1", "T", 0.4, [])],
    "tails on different components": [
        ("1", "T", 0.3, [("2", "T")]),
        ("1", "T", 0.4, [("3", "F")]),
    ],
    "tails that agree where they overlap": [
        ("1", "T", 0.3, [("2", "T")]),
        ("1", "T", 0.4, [("2", "T"), ("3", "F")]),
    ],
}


@pytest.mark.parametrize("snodes", NON_MUTEX.values(), ids=list(NON_MUTEX))
def test_snodes_on_one_head_that_can_hold_together_are_not_mutex(snodes):
    bkb = build("non-mutex", snodes)
    assert not bkb.is_mutex()
    assert bkb.non_mutex_pair() == (0, 1)


def test_non_mutex_pair_takes_heads_in_order_then_pairs_in_index_order():
    bkb = build(
        "order",
        [
            ("1", "T", 0.3, [("2", "T")]),
            ("3", "T", 0.5, []),
            ("3", "T", 0.5, []),
            ("1", "T", 0.4, [("2", "F")]),
            ("1", "T", 0.2, [("4", "T")]),
        ],
    )
    assert bkb.non_mutex_pair() == (0, 4)


def test_find_snodes_on_a_head_without_snodes_is_empty():
    bkb = build("bare", [("A", "1", 0.5, [])])
    bkb.add_inode("A", "2")
    assert bkb.find_snodes("A", "2") == []


def test_tails_that_disagree_on_a_shared_component_are_mutex():
    bkb = build(
        "mutex",
        [
            ("1", "T", 0.3, [("2", "T")]),
            ("1", "T", 0.4, [("2", "F")]),
            ("1", "F", 0.7, [("2", "T")]),
        ],
    )
    assert bkb.is_mutex()
    assert bkb.non_mutex_pair() is None


@pytest.mark.parametrize("name", document_names())
def test_is_mutex_agrees_with_pybkb_on_every_fixture(name):
    assert exported(name).is_mutex() is manifest()["mutex"][name]


def test_adding_an_inode_twice_changes_nothing():
    bkb = BKB("twice")
    for component, state in [("A", "1"), ("A", "2"), ("A", "1")]:
        bkb.add_inode(component, state)
    assert bkb.inodes == (("A", "1"), ("A", "2"))
    assert bkb.states("A") == ("1", "2")
    assert bkb.components() == ("A",)
    assert ("A", "2") in bkb
    assert BKB("empty").states("A") == ()


BAD_INODES: dict[str, tuple[Any, Any]] = {
    "an integer component": (1, "1"),
    "an integer state": ("A", 1),
    "a boolean state": ("A", True),
    "an empty source set": ("A", frozenset()),
    "a source set of integers": ("A", frozenset({1})),
}


@pytest.mark.parametrize("inode", BAD_INODES.values(), ids=list(BAD_INODES))
def test_add_inode_refuses(inode):
    bkb = BKB("bad")
    with pytest.raises(StructureError):
        bkb.add_inode(*inode)
    assert bkb.inodes == ()


BAD_SNODES: dict[str, tuple[Any, ...]] = {
    "an unknown head": ("C", "1", 0.5, []),
    "an unknown tail I-node": ("A", "1", 0.5, [("C", "1")]),
    "a negative probability": ("A", "1", -0.1, []),
    "a probability above one": ("A", "1", 1.1, []),
    "a NaN probability": ("A", "1", math.nan, []),
    "an infinite probability": ("A", "1", math.inf, []),
    "a boolean probability": ("A", "1", True, []),
    "a string probability": ("A", "1", "0.5", []),
    "a tail on the head itself": ("A", "1", 0.5, [("A", "1")]),
    "a tail on the head's component": ("A", "1", 0.5, [("A", "2")]),
    "a tail naming a component twice": ("A", "1", 0.5, [("B", "1"), ("B", "2")]),
}


@pytest.mark.parametrize("snode", BAD_SNODES.values(), ids=list(BAD_SNODES))
def test_add_snode_refuses(snode):
    bkb = BKB("bad")
    for component, state in [("A", "1"), ("A", "2"), ("B", "1"), ("B", "2")]:
        bkb.add_inode(component, state)
    with pytest.raises(StructureError):
        bkb.add_snode(*snode)
    assert bkb.snodes == ()


BASE: list[Any] = [("A", "1", 0.5, []), ("B", "1", 0.25, [("A", "1")])]


def test_equality_ignores_names_descriptions_and_order():
    other = BKB("other", "described")
    for component, state in [("B", "1"), ("A", "1")]:
        other.add_inode(component, state)
    other.add_snode("B", "1", 0.25, [("A", "1")])
    other.add_snode("A", "1", 0.5)
    assert build("base", BASE) == other
    assert other == build("base", BASE)


def extra_inode() -> BKB:
    bkb = build("extra", BASE)
    bkb.add_inode("A", "2")
    return bkb


UNEQUAL: dict[str, Callable[[], BKB]] = {
    "a probability one ulp apart": lambda: build(
        "ulp", [("A", "1", math.nextafter(0.5, 1.0), []), BASE[1]]
    ),
    "a different tail": lambda: build("tail", [BASE[0], ("B", "1", 0.25, [])]),
    "a repeated S-node": lambda: build("repeat", [*BASE, BASE[0]]),
    "an extra I-node": extra_inode,
}


@pytest.mark.parametrize("make", UNEQUAL.values(), ids=list(UNEQUAL))
def test_equality_tells_apart(make):
    assert build("base", BASE) != make()
    assert make() != build("base", BASE)


def test_a_bkb_is_unhashable_and_unequal_to_other_types():
    with pytest.raises(TypeError):
        hash(BKB("x"))
    assert BKB("x") != "x"


def test_repr_names_the_bkb_and_its_sizes():
    assert repr(simple_build()) == "BKB(name='simple', inodes=4, snodes=4)"
