"""`bko.bkb.inference`: instantiation from an anchor, and the product of one inference."""

from __future__ import annotations

import pytest

from bko.bkb.errors import QueryError
from bko.bkb.fusion import fuse
from bko.bkb.inference import derivation, instantiate, probability, reach
from bko.bkb.structure import BKB
from tests._bkb_fixtures import build


pytestmark = pytest.mark.filterwarnings("error")

A, B, C, D, E = ("A", "1"), ("B", "1"), ("C", "1"), ("D", "1"), ("E", "1")


def chain() -> BKB:
    return build("chain", [("B", "1", 1.0, [A]), ("C", "1", 0.5, [B])])


def test_each_pass_adds_every_snode_whose_antecedents_hold():
    assert instantiate(chain(), [A]) == ((0,), (1,))
    assert reach(chain(), [A]) == {A: None, B: 0, C: 1}


def test_an_snode_waits_for_its_whole_tail():
    bkb = build("both", [("B", "1", 1.0, [A]), ("C", "1", 1.0, [A]), ("D", "1", 1.0, [B, C])])
    assert instantiate(bkb, [A]) == ((0, 1), (2,))
    assert list(reach(bkb, [A])) == [A, B, C, D]
    half = build("half", [("D", "1", 1.0, [B, C])])
    assert instantiate(half, [B]) == ()
    assert reach(half, [B]) == {B: None}


def test_the_lowest_index_of_the_first_pass_is_credited():
    bkb = build("two", [("B", "1", 0.25, [A]), ("B", "1", 0.75, [A])])
    assert instantiate(bkb, [A]) == ((0, 1),)
    assert reach(bkb, [A]) == {A: None, B: 0}


def test_a_prior_on_a_class_joins_the_first_pass():
    bkb = build("prior", [("B", "1", 1.0, [A]), ("E", "1", 0.5, [])])
    assert instantiate(bkb, [A]) == ((0, 1),)
    assert reach(bkb, [A]) == {A: None, B: 0, E: 1}


def test_an_snode_on_a_held_head_still_joins_its_pass_and_a_cycle_ends():
    bkb = build("cycle", [("A", "1", 0.5, [B]), ("B", "1", 0.5, [A])])
    assert instantiate(bkb, [A]) == ((1,), (0,))
    assert reach(bkb, [A]) == {A: None, B: 1}


def test_sources_stand_on_their_priors_and_join_the_derivation():
    fused = fuse([chain(), build("other", [("C", "1", 1.0, [D])])], [1, 2])
    snodes = derivation(fused, reach(fused, [A]), C)
    assert [fused.snodes[index].prob for index in snodes] == [1.0, 1 / 3, 0.5, 1 / 3]
    assert repr(probability(fused, snodes)) == repr(1.0 * (1 / 3) * 0.5 * (1 / 3))


def test_the_anchor_needs_no_snode():
    assert derivation(chain(), reach(chain(), [A]), B) == (0,)
    assert repr(probability(chain(), ())) == "1.0"


@pytest.mark.parametrize(
    ("anchor", "match"),
    [
        ([], "the anchor is empty"),
        ([("Z", "1")], "is not an I-node"),
        ([("__Source__[B]", "s")], "is on a source component"),
    ],
)
def test_instantiation_refuses_a_bad_anchor(anchor, match):
    bkb = chain()
    bkb.add_inode("__Source__[B]", "s")
    with pytest.raises(QueryError, match=match):
        instantiate(bkb, anchor)
    with pytest.raises(QueryError, match=match):
        reach(bkb, anchor)


def test_an_unreached_goal_or_a_source_without_its_prior_is_refused():
    with pytest.raises(QueryError, match="is not reached"):
        derivation(chain(), {A: None}, C)
    bare = build("bare", [("B", "1", 1.0, [A, ("__Source__[B]", "s")])])
    with pytest.raises(QueryError, match="has 0 prior S-nodes"):
        derivation(bare, reach(bare, [A]), B)


def test_probability_refuses_an_unknown_index():
    with pytest.raises(QueryError, match="is not an S-node"):
        probability(chain(), [2])
    with pytest.raises(QueryError, match="is not an S-node"):
        probability(chain(), [-1])


def test_a_repeated_anchor_counts_once_in_the_order_given():
    assert list(reach(chain(), [B, A, B])) == [B, A, C]


def test_source_priors_never_join_a_pass():
    fused = fuse([chain()], [1])
    passes = instantiate(fused, [A])
    assert all(fused.snodes[index].tail for added in passes for index in added)
    assert len(passes) == 2


def test_a_source_with_two_priors_is_refused():
    source = ("__Source__[B]", "s")
    bkb = build("two", [("B", "1", 1.0, [A, source]), (*source, 0.5, []), (*source, 0.25, [])])
    with pytest.raises(QueryError, match="has 2 prior S-nodes"):
        derivation(bkb, reach(bkb, [A]), B)
