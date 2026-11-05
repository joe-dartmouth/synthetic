"""Concept shares under each planted data contradiction: finite, in [0, 1], and nested."""

from __future__ import annotations

import itertools
import math

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps.observe import CellRecords, Observed, observe_records, record_cells
from bko.gaps.shares import nested, shares
from bko.generate.direct import sample
from tests._contradiction_fixtures import INNER, KINDS, NAME, OUTER, YM1, plant


def share(n_c: int, k_c: int, n_g: int, k_g: int) -> float:
    """One concept's share in a cell of `n_c` records, `k_c` showing it, in a group of `n_g`."""
    cell = CellRecords(Observed("cell", n_c, {"X": k_c}), 1, frozenset())
    return shares(cell, Observed("group", n_g, {"X": k_g}))["X"]


GRID = [
    (n_c, k_c, n_g, k_g)
    for n_c, n_g in itertools.product((0, 1, 2, 5, 19, 20, 21, 300), (1, 2, 9, 20, 41, 300))
    for k_c in sorted({0, n_c // 2, n_c})
    for k_g in sorted({0, 1, n_g // 3, n_g - 1, n_g})
]
"""Cell and group sizes about `GROUP_WEIGHT`, each with none, some and every record showing."""


def test_every_share_on_the_grid_is_finite_and_inside_0_and_1():
    for n_c, k_c, n_g, k_g in GRID:
        found = share(n_c, k_c, n_g, k_g)
        assert math.isfinite(found)
        assert 0 <= found <= 1, (n_c, k_c, n_g, k_g)


def test_an_empty_cells_share_on_the_grid_is_exactly_its_groups():
    for n_g, k_g in {(n_g, k_g) for _, _, n_g, k_g in GRID}:
        assert share(0, 0, n_g, k_g) == k_g / n_g, (n_g, k_g)


@pytest.mark.parametrize("kind", KINDS)
def test_each_kinds_shares_are_finite_and_inside_0_and_1(kind):
    planted = plant(kind)
    cells = record_cells(planted.frame, planted.rows, name=NAME, edges=UCI_FRAME.edges)
    for cell, records in cells.each.items():
        members = planted.frame.loc[planted.frame[UCI_FRAME.label].eq(cell[2])]
        group = observe_records(members, planted.rows, name=f"{NAME}/group")
        found = shares(records, group)
        assert set(found) == {row.concept for row in planted.rows}
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in found.values())
        if records.observed.records == 0:
            assert found == {c: group.holders[c] / group.records for c in found}, cell


@pytest.mark.parametrize("kind", KINDS)
def test_the_outer_concept_is_drawn_whenever_the_inner_is(kind):
    planted = plant(kind)
    assert nested(planted.rows) == {INNER: OUTER}
    drawn = sample(
        planted.frame,
        {YM1: 2000},
        rows=planted.rows,
        ontologies=planted.ontologies,
        name=NAME,
        seed=0,
        frame=UCI_FRAME,
    )
    draws = drawn[YM1]
    assert len(draws) == 2000
    assert all(OUTER in d.concepts for d in draws if INNER in d.concepts)
