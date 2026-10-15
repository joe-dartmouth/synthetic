"""The `bko.baselines.request` core: strata, water-filling and the request."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from bko.baselines.request import Method, Request, Strata, Stratum, plan, stratum_keys, water_fill
from tests._datasets_fixtures import records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

FLAGGED = {
    (50, 0, 1): 2,
    (60, 0, 1): 18,
    (50, 0, 0): 30,
    (50, 1, 0): 47,
    (50, 1, 1): 32,
    (60, 0, 0): 27,
    (60, 1, 0): 24,
    (60, 1, 1): 58,
}
"""The seed-0 training split's cell sizes, one age per band."""


def test_twenty_rows_fill_both_flagged_cells_to_twenty():
    request = plan(records(FLAGGED), 20)
    assert dict(request.quota) == {("<55", "female", 1): 18, (">=55", "female", 1): 2}
    assert request.rows == 20


def test_ties_go_to_the_earlier_stratum_in_grid_order():
    grid = Strata.CELL.grid()
    assert water_fill(dict.fromkeys(grid, 6), 5) == dict.fromkeys(grid[:5], 1)


@pytest.mark.parametrize("seed", range(25))
def test_water_fill_levels_the_smallest_strata(seed):
    """Independent of the loop: totals match, and no gainer ends above the lowest level plus one."""
    rng = np.random.default_rng(seed)
    grid = Strata.CELL.grid()
    counts = {stratum: int(rng.integers(0, 30)) for stratum in grid}
    counts[grid[0]] = max(counts[grid[0]], 1)
    rows = int(rng.integers(1, 80))
    gained = water_fill(counts, rows)
    assert sum(gained.values()) == rows
    assert all(counts[stratum] > 0 for stratum in gained)
    final = {stratum: n + gained.get(stratum, 0) for stratum, n in counts.items() if n > 0}
    assert max(final[stratum] for stratum in gained) <= min(final.values()) + 1


def test_an_empty_stratum_never_gains_rows():
    counts = dict.fromkeys(Strata.CELL.grid(), 5) | {("<55", "female", 0): 0}
    assert ("<55", "female", 0) not in water_fill(counts, 40)


@pytest.mark.parametrize(
    ("counts", "rows"),
    [(dict.fromkeys(Strata.LABEL.grid(), 0), 1), (dict.fromkeys(Strata.LABEL.grid(), 3), -1)],
    ids=["no records", "negative rows"],
)
def test_impossible_rows_are_refused(counts, rows):
    with pytest.raises(ValueError, match="cannot give out"):
        water_fill(counts, rows)


def test_label_strata_key_by_the_label_alone():
    assert dict(plan(records(FLAGGED), 18, strata=Strata.LABEL).quota) == {(1,): 18}


def test_stratum_keys_follow_the_cell_columns():
    frame = records({(54, 0, 1): 1, (55, 1, 0): 1})
    assert stratum_keys(frame, Strata.CELL) == [("<55", "female", 1), (">=55", "male", 0)]
    assert stratum_keys(frame, Strata.LABEL) == [(1,), (0,)]


@pytest.mark.parametrize(
    "quota",
    [
        {("<55", "female", 1): 0},
        {("<55", "female", 1): 1.5},
        {("<55", "female", 1): True},
        {("nonsense",): 1},
        {(1,): 1},
    ],
    ids=[
        "zero rows",
        "fractional rows",
        "boolean rows",
        "unknown stratum",
        "label key under cell strata",
    ],
)
def test_a_malformed_quota_is_refused(quota):
    with pytest.raises(ValueError, match="not a positive row count"):
        Request(quota=quota)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.0])
def test_a_seed_outside_numpys_range_is_refused(seed):
    with pytest.raises(ValueError, match="outside"):
        Request(seed=seed)


def test_the_quota_is_held_in_grid_order_and_apart_from_its_source():
    source: dict[Stratum, int] = {(">=55", "female", 1): 1, ("<55", "female", 1): 2}
    request = Request(quota=source)
    assert list(request.quota) == [("<55", "female", 1), (">=55", "female", 1)]
    assert isinstance(request.quota, MappingProxyType)
    source["<55", "male", 0] = 1
    assert request.rows == 3


def test_only_the_controls_add_no_rows():
    assert [method for method in Method if not method.adds_rows] == [
        Method.NONE,
        Method.CLASS_WEIGHT,
    ]
