"""The conditional `is_a` draw of `bko.generate.hierarchy`: a broad concept given its narrow one.

The hand cell holds 10 records split 2/3/1/4 over (N and B, N only, B only, neither), so
`s(N) = 1/2` and `s(B) = 3/10` whenever its label group of 40 shows the same shares. Every value
is worked by hand from ADR `research/bkb-direct-generation`, Decision 12, at `GROUP_WEIGHT` 20.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Final

import pytest

from bko.gaps.observe import Observed
from bko.gaps.shares import GROUP_WEIGHT, SharePart
from bko.generate.hierarchy import broad_chance, conditional, pair_joint, pair_weights


NARROW: Final = "MONDO:9000002"
BROAD: Final = "MONDO:9000001"
PAIR: Final = (NARROW, BROAD)


def _observed(name: str, both: int, narrow_only: int, broad_only: int, neither: int) -> Observed:
    """Records split over (N and B, N only, B only, neither), observed with the pair counted."""
    return Observed(
        name=name,
        records=both + narrow_only + broad_only + neither,
        holders={BROAD: both + broad_only, NARROW: both + narrow_only},
        both={PAIR: both},
    )


CELL: Final = _observed("cell", 2, 3, 1, 4)


def _share(cell: Observed, group: Observed, concept: str) -> float:
    """The concept's share at the cell, as `bko.gaps.shares.shares` reads it."""
    part = SharePart(
        cell.records,
        cell.holders[concept] / cell.records,
        group.holders[concept] / group.records,
    )
    return part.share


def _close(actual: Mapping[tuple[bool, bool], float], hand: Mapping[tuple[bool, bool], Fraction]):
    """Each cell of `actual` within a few ulps of its hand value, and no other cell."""
    assert actual.keys() == hand.keys()
    for key, value in hand.items():
        assert math.isclose(actual[key], float(value), rel_tol=1e-15, abs_tol=1e-16), key


def test_the_weight_is_the_one_the_hand_values_read():
    assert GROUP_WEIGHT == 20


@pytest.mark.parametrize(
    ("group", "weights", "hand_conditional", "joint"),
    [
        (
            _observed("group", 8, 12, 4, 16),
            (Fraction(6), Fraction(15)),
            Fraction(2, 5),
            (Fraction(1, 5), Fraction(3, 10), Fraction(1, 10), Fraction(2, 5)),
        ),
        (
            _observed("group", 4, 16, 8, 12),
            (Fraction(4), Fraction(15)),
            Fraction(4, 15),
            (Fraction(2, 15), Fraction(11, 30), Fraction(1, 6), Fraction(1, 3)),
        ),
    ],
    ids=["cell-proportions", "fused"],
)
def test_pseudo_count_0_reproduces_the_pairs_training_joint(
    group, weights, hand_conditional, joint
):
    both, narrow = pair_weights(CELL, group, PAIR)
    assert (both, narrow) == weights
    found = conditional(both=both, narrow=narrow, pseudo=0.0)
    assert (found,) == (float(hand_conditional),)
    share_broad, share_narrow = _share(CELL, group, BROAD), _share(CELL, group, NARROW)
    assert (share_broad, share_narrow) == (0.3, 0.5)
    drawn = pair_joint(share_broad=share_broad, share_narrow=share_narrow, conditional=found)
    keys = ((True, True), (True, False), (False, True), (False, False))
    _close(drawn, dict(zip(keys, joint, strict=True)))
    assert math.isclose(drawn[True, True] + drawn[False, True], 0.3, rel_tol=1e-15)


def test_the_fused_joint_differs_from_the_product_bko_direct_draws():
    both, narrow = pair_weights(CELL, _observed("group", 4, 16, 8, 12), PAIR)
    drawn = pair_joint(
        share_broad=0.3,
        share_narrow=0.5,
        conditional=conditional(both=both, narrow=narrow, pseudo=0.0),
    )
    independent = float(Fraction(3, 20))
    assert not math.isclose(drawn[True, True], independent, rel_tol=1e-3)
    assert math.isclose(drawn[True, True], float(Fraction(2, 15)), rel_tol=1e-15)


def test_the_shrunk_conditional_is_the_hand_value():
    both, narrow = pair_weights(CELL, _observed("group", 8, 12, 4, 16), PAIR)
    assert {
        pseudo: conditional(both=both, narrow=narrow, pseudo=pseudo) for pseudo in (0.0, 20.0)
    } == {
        0.0: float(Fraction(2, 5)),
        20.0: float(Fraction(26, 35)),
    }


def test_the_closed_conditional_is_1_and_the_marginal_is_clamped():
    assert (conditional(both=Fraction(4), narrow=Fraction(15), pseudo=None),) == (1.0,)
    assert broad_chance(
        share_broad=0.3, share_narrow=0.5, conditional=1.0, narrow_present=False
    ) == (0.0, True)
    drawn = pair_joint(share_broad=0.3, share_narrow=0.5, conditional=1.0)
    assert drawn == {(True, True): 0.5, (True, False): 0.0, (False, True): 0.0, (False, False): 0.5}
    assert (drawn[True, True] + drawn[False, True],) == (0.5,)


def test_the_broad_marginal_is_kept_when_the_clamp_does_not_fire():
    chance, clamped = broad_chance(
        share_broad=float(Fraction(7, 10)),
        share_narrow=0.5,
        conditional=float(Fraction(26, 35)),
        narrow_present=False,
    )
    assert not clamped
    assert math.isclose(chance, float(Fraction(23, 35)), rel_tol=1e-15)
    given = broad_chance(
        share_broad=0.7, share_narrow=0.5, conditional=float(Fraction(26, 35)), narrow_present=True
    )
    assert given == (float(Fraction(26, 35)), False)
    drawn = pair_joint(share_broad=0.7, share_narrow=0.5, conditional=float(Fraction(26, 35)))
    assert math.isclose(drawn[True, True] + drawn[False, True], 0.7, rel_tol=1e-15)


def test_a_complement_above_1_is_clamped():
    assert broad_chance(
        share_broad=0.9, share_narrow=0.5, conditional=0.0, narrow_present=False
    ) == (1.0, True)


def test_a_pair_with_no_narrow_holder_leaves_the_broad_at_its_share():
    cell, group = _observed("cell", 0, 0, 3, 7), _observed("group", 0, 0, 12, 28)
    both, narrow = pair_weights(cell, group, PAIR)
    assert (both, narrow) == (Fraction(0), Fraction(0))
    share_broad, share_narrow = _share(cell, group, BROAD), _share(cell, group, NARROW)
    assert (share_narrow,) == (0.0,)
    for pseudo in (0.0, 20.0, None):
        found = conditional(both=both, narrow=narrow, pseudo=pseudo)
        assert (found,) == (1.0,)
        assert broad_chance(
            share_broad=share_broad,
            share_narrow=share_narrow,
            conditional=found,
            narrow_present=False,
        ) == (share_broad, False)


def test_a_source_with_no_narrow_holder_contributes_0():
    cell = _observed("cell", 0, 0, 3, 7)
    assert pair_weights(cell, _observed("group", 8, 12, 4, 16), PAIR) == (Fraction(4), Fraction(10))


def test_no_complement_is_read_when_every_record_shows_the_narrow_concept():
    with pytest.raises(ValueError, match="share 1"):
        broad_chance(share_broad=0.5, share_narrow=1.0, conditional=0.5, narrow_present=False)
    drawn = pair_joint(share_broad=0.5, share_narrow=1.0, conditional=0.5)
    assert drawn == {(True, True): 0.5, (True, False): 0.5, (False, True): 0.0, (False, False): 0.0}
