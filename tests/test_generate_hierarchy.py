"""The conditional `is_a` draw of `bko.generate.hierarchy`: a broad concept given its narrow one.

The hand cell holds 10 records split 2/3/1/4 over (N and B, N only, B only, neither), so
`s(N) = 1/2` and `s(B) = 3/10` whenever its label group of 40 shows the same shares. Every value
is worked by hand from ADR `research/bkb-direct-generation`, Decision 12, at `GROUP_WEIGHT` 20.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from fractions import Fraction
from typing import Final

import pandas as pd
import pytest

from bko.datasets.cells import cell_keys
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.added_contradictions import added_counts
from bko.gaps.observe import Cell, Observed, record_cells
from bko.gaps.shares import GROUP_WEIGHT, SharePart
from bko.generate.direct import Draw, _Plan, sample
from bko.generate.donors import _label_groups, cell_records
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
from bko.generate.hierarchy import (
    always_present,
    broad_chance,
    conditional,
    pair_joint,
    pair_weights,
)
from bko.generate.presence import Presence
from bko.ontology.schema import FieldMapping
from tests._contradiction_fixtures import (
    BASE_ROWS,
    INNER,
    NAME,
    OM1,
    OUTER,
    YM1,
    YW0,
    YW1,
    ontologies,
    plant,
    row,
)
from tests._contradiction_fixtures import BROAD as PLANTED_BROAD
from tests._contradiction_fixtures import NARROW as PLANTED_NARROW
from tests._contradiction_fixtures import ROOT as PLANTED_ROOT


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
    drawn = pair_joint(share_broad=share_broad, share_narrow=share_narrow, given=found)
    keys = ((True, True), (True, False), (False, True), (False, False))
    _close(drawn, dict(zip(keys, joint, strict=True)))
    assert math.isclose(drawn[True, True] + drawn[False, True], 0.3, rel_tol=1e-15)


def test_the_fused_joint_differs_from_the_product_bko_direct_draws():
    both, narrow = pair_weights(CELL, _observed("group", 4, 16, 8, 12), PAIR)
    drawn = pair_joint(
        share_broad=0.3,
        share_narrow=0.5,
        given=conditional(both=both, narrow=narrow, pseudo=0.0),
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
    assert broad_chance(share_broad=0.3, share_narrow=0.5, given=1.0, narrow_present=False) == (
        0.0,
        True,
    )
    drawn = pair_joint(share_broad=0.3, share_narrow=0.5, given=1.0)
    assert drawn == {(True, True): 0.5, (True, False): 0.0, (False, True): 0.0, (False, False): 0.5}
    assert (drawn[True, True] + drawn[False, True],) == (0.5,)


def test_the_broad_marginal_is_kept_when_the_clamp_does_not_fire():
    chance, clamped = broad_chance(
        share_broad=float(Fraction(7, 10)),
        share_narrow=0.5,
        given=float(Fraction(26, 35)),
        narrow_present=False,
    )
    assert not clamped
    assert math.isclose(chance, float(Fraction(23, 35)), rel_tol=1e-15)
    given = broad_chance(
        share_broad=0.7, share_narrow=0.5, given=float(Fraction(26, 35)), narrow_present=True
    )
    assert given == (float(Fraction(26, 35)), False)
    drawn = pair_joint(share_broad=0.7, share_narrow=0.5, given=float(Fraction(26, 35)))
    assert math.isclose(drawn[True, True] + drawn[False, True], 0.7, rel_tol=1e-15)


def test_a_complement_above_1_is_clamped():
    assert broad_chance(share_broad=0.9, share_narrow=0.5, given=0.0, narrow_present=False) == (
        1.0,
        True,
    )


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
            given=found,
            narrow_present=False,
        ) == (share_broad, False)


def test_a_source_with_no_narrow_holder_contributes_0():
    cell = _observed("cell", 0, 0, 3, 7)
    assert pair_weights(cell, _observed("group", 8, 12, 4, 16), PAIR) == (Fraction(4), Fraction(10))


def test_no_complement_is_read_when_every_record_shows_the_narrow_concept():
    with pytest.raises(ValueError, match="share 1"):
        broad_chance(share_broad=0.5, share_narrow=1.0, given=0.5, narrow_present=False)
    drawn = pair_joint(share_broad=0.5, share_narrow=1.0, given=0.5)
    assert drawn == {(True, True): 0.5, (True, False): 0.5, (False, True): 0.0, (False, False): 0.0}


def test_a_narrow_share_one_ulp_below_1_reads_no_complement():
    below = math.nextafter(1.0, 0.0)
    assert (below, always_present(below), always_present(0.9999999999999998)) == (
        1 - 2**-53,
        True,
        False,
    )
    with pytest.raises(ValueError, match="share 1"):
        broad_chance(share_broad=0.5, share_narrow=below, given=0.25, narrow_present=False)
    drawn = pair_joint(share_broad=0.5, share_narrow=below, given=0.25)
    assert (drawn[False, True], drawn[False, False]) == (2**-55, 3 * 2**-55)


def test_an_empty_label_group_is_refused():
    with pytest.raises(ValueError, match="label group group holds no record"):
        pair_weights(CELL, _observed("group", 0, 0, 0, 0), PAIR)


ROWS: Final = (
    row(OUTER, "restecg", 1, 2),
    row(PLANTED_NARROW, "cp", 1),
    row(PLANTED_BROAD, "exang", 1),
)
"""The planted `is_a` pair on `cp` and `exang`, beside one concept outside every pair and no
nested pair."""

QUOTA: Final = {YW1: 300, YM1: 300, OM1: 300, YW0: 300}

HIER_0: Final = DrawSpec(hierarchy=0.0)
HIER_W: Final = DrawSpec(hierarchy=float(GROUP_WEIGHT))
CLOSED: Final = DrawSpec(closed=True)


def _draw(
    spec: DrawSpec, rows: tuple[FieldMapping, ...] = ROWS, kind: str = "narrow-without-broad"
) -> dict[Cell, tuple[Draw, ...]]:
    planted = plant(kind)
    return sample(
        planted.frame,
        QUOTA,
        rows=rows,
        ontologies=planted.ontologies,
        name=NAME,
        seed=0,
        frame=UCI_FRAME,
        tag=(3, "gap"),
        spec=spec,
    )


def _narrow_without_broad(spec: DrawSpec) -> int:
    drawn = _draw(spec)
    added = pd.DataFrame([dict(d.record) for draws in drawn.values() for d in draws])
    found = added_counts(added, ROWS, ontologies(), frame=UCI_FRAME, asked=None, name="added")[
        "narrow-without-broad"
    ]
    assert found is not None
    return found


def test_the_closed_arm_yields_no_narrow_without_broad_row():
    assert _narrow_without_broad(CLOSED) == 0
    assert _narrow_without_broad(DEFAULT_SPEC) > 0
    assert _narrow_without_broad(HIER_0) > 0
    for draws in _draw(CLOSED).values():
        assert all(PLANTED_BROAD in d.concepts for d in draws if PLANTED_NARROW in d.concepts)


def test_narrow_without_broad_rows_do_not_rise_with_the_pseudo_count():
    counts = [_narrow_without_broad(spec) for spec in (HIER_0, HIER_W, CLOSED)]
    assert counts[0] > counts[1] > counts[2] == 0


def test_concepts_outside_a_pair_match_bko_direct_at_the_same_seed():
    direct, hier = _draw(DEFAULT_SPEC), _draw(HIER_W)
    differs = 0
    for cell, draws in direct.items():
        assert len(hier[cell]) == len(draws)
        for mine, theirs in zip(hier[cell], draws, strict=True):
            assert (OUTER in mine.concepts) == (OUTER in theirs.concepts)
            differs += (PLANTED_BROAD in mine.concepts) != (PLANTED_BROAD in theirs.concepts)
    assert differs > 0


def test_the_narrow_concept_draws_at_its_share_and_the_broad_at_its_parts():
    """At `YW1`, `m = 6 + 20 * 12/18`, `b = 20 * 6/18`, so `c_0 = 10/29`; hand worked."""
    planted = plant("narrow-without-broad")
    plan = _Plan(ROWS, planted.ontologies, UCI_FRAME, HIER_0)
    assert plan.pairs == ((PLANTED_NARROW, PLANTED_BROAD),)
    cells = record_cells(planted.frame, ROWS, name=NAME, edges=UCI_FRAME.edges, pairs=plan.pairs)
    keys = cell_keys(planted.frame, edges=UCI_FRAME.edges)
    groups = _label_groups(planted.frame, keys, {1}, ROWS, NAME, UCI_FRAME, plan.pairs)
    own = cell_records(planted.frame, keys, YW1, UCI_FRAME)
    sources = plan.sources(YW1, cells.each[YW1], groups[1], own)
    (part,) = sources.pairs.values()
    assert (part.given, part.pseudo, part.clamped) == (PLANTED_NARROW, 0.0, False)
    assert math.isclose(part.both, float(Fraction(20, 3)), rel_tol=1e-15)
    assert math.isclose(part.narrow, float(Fraction(58, 3)), rel_tol=1e-15)
    assert (part.conditional,) == (float(Fraction(10, 29)),)
    assert part.complement is not None
    assert math.isclose(part.complement, 1.0, rel_tol=1e-12)


def test_the_hier_line_counts_each_cells_pairs_and_clamps(caplog):
    caplog.set_level(logging.INFO, logger="bko.generate.hierarchy")
    _draw(CLOSED)
    assert [m for m in caplog.messages if m.startswith("hier ")] == [
        "hier holdout=3 fill=gap key=<55/female/1 pairs=1 clamped=1",
        "hier holdout=3 fill=gap key=<55/male/1 pairs=1 clamped=0",
        "hier holdout=3 fill=gap key=>=55/male/1 pairs=1 clamped=0",
        "hier holdout=3 fill=gap key=<55/female/0 pairs=1 clamped=0",
    ]


def test_a_broad_with_two_narrow_concepts_is_refused():
    rows = (*ROWS, row(PLANTED_ROOT, "thal", 7))
    with pytest.raises(ValueError, match=rf"cannot draw: {PLANTED_ROOT} each hold more than one"):
        _draw(HIER_0, rows)


def test_a_frame_with_a_nested_pair_is_refused():
    with pytest.raises(ValueError, match=rf"nested pairs {INNER} in {OUTER}$"):
        _draw(CLOSED, BASE_ROWS)
    assert _draw(DEFAULT_SPEC, BASE_ROWS)


def test_a_frame_with_no_pair_is_refused():
    with pytest.raises(ValueError, match="needs an is_a pair among the drawn concepts"):
        _draw(HIER_W, (row(OUTER, "restecg", 1, 2), row(PLANTED_BROAD, "exang", 1)))


@pytest.mark.parametrize(
    ("fields", "says"),
    [
        ({"hierarchy": 0}, "hierarchy 0 must be a finite non-negative float"),
        ({"hierarchy": -1.0}, "hierarchy -1.0 must be"),
        ({"hierarchy": math.inf}, "hierarchy inf must be"),
        ({"hierarchy": 0.0, "closed": True}, "closed sets the conditional to 1"),
        ({"tree": True, "hierarchy": 0.0}, "tree draws each tree field as one leaf"),
        ({"tree": True, "closed": True}, "tree draws each tree field as one leaf"),
        ({"hierarchy": 0.0, "mapped": False}, "hierarchy needs the mapping table"),
        ({"closed": True, "presence": Presence.LOGIC}, "closed needs the mapping table"),
    ],
    ids=["int", "negative", "infinite", "both", "tree-hier", "tree-closed", "unmapped", "logic"],
)
def test_a_spec_the_conditional_draw_cannot_read_is_refused(fields, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        DrawSpec(**fields)


def test_the_conditional_specs_log_only_their_fields():
    assert [spec.params() for spec in (HIER_0, HIER_W, CLOSED)] == [
        {"hierarchy": 0.0},
        {"hierarchy": 20.0},
        {"closed": True},
    ]
    assert [spec.tokens() for spec in (HIER_0, HIER_W, CLOSED)] == [
        " hierarchy=0.0",
        " hierarchy=20.0",
        " closed=true",
    ]
    assert (DEFAULT_SPEC.conditional, HIER_0.conditional, CLOSED.conditional) == (
        False,
        True,
        True,
    )
