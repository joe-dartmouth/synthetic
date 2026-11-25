"""The synthetic-share sweep's dropped rows: a share of each cell's training rows, by hash rank."""

from __future__ import annotations

import hashlib
import itertools
from fractions import Fraction

import pytest

from bko.datasets.cells import CELL_COLUMNS, cell_keys
from bko.datasets.split import dropped, dropped_count
from tests._frame_fixtures import TOY, TOY_TRAIN


SHARES = (Fraction(1, 50), Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(1))


@pytest.mark.parametrize(
    ("count", "share", "expected"),
    [
        (1, Fraction(1, 2), 1),
        (3, Fraction(1, 50), 0),
        (7, Fraction(1), 7),
        (5, Fraction(1, 10), 1),
        (4, Fraction(1, 10), 0),
        (0, Fraction(1, 2), 0),
    ],
)
def test_the_dropped_count_rounds_half_up_with_no_floor(count, share, expected):
    assert dropped_count(count, share) == expected


@pytest.mark.parametrize("share", [Fraction(0), Fraction(3, 2), 0.5])
def test_the_dropped_count_refuses_a_share_outside_the_unit_interval(share):
    with pytest.raises(ValueError, match=r"synthetic share .* must be a Fraction in \(0, 1\]"):
        dropped_count(10, share)


def test_the_dropped_rows_nest_across_shares():
    sets = [set(dropped(TOY_TRAIN, share, seed=3, edges=TOY.edges)) for share in SHARES]
    for smaller, larger in itertools.pairwise(sets):
        assert smaller <= larger
    assert len(sets[0]) < len(sets[-1])


def test_the_dropped_rows_are_the_lowest_ranked():
    share, seed = Fraction(1, 4), 7
    expected: set[int] = set()
    cells = cell_keys(TOY_TRAIN, edges=TOY.edges).groupby(list(CELL_COLUMNS)).groups
    for records in cells.values():
        ranked = sorted(
            records,
            key=lambda record: hashlib.sha256(f"7:synthetic-share:{record}".encode()).digest(),
        )
        expected |= set(ranked[: (2 * len(ranked) + 4) // 8])
    found = dropped(TOY_TRAIN, share, seed=seed, edges=TOY.edges)
    assert set(found) == expected
    assert found.is_unique


def test_the_ranking_is_not_the_subsamples():
    # The tag sets the order: under `train-fraction:` the same quarter would differ.
    share = Fraction(1, 4)
    found = set(dropped(TOY_TRAIN, share, seed=0, edges=TOY.edges))
    other: set[int] = set()
    cells = cell_keys(TOY_TRAIN, edges=TOY.edges).groupby(list(CELL_COLUMNS)).groups
    for records in cells.values():
        ranked = sorted(
            records,
            key=lambda record: hashlib.sha256(f"0:train-fraction:{record}".encode()).digest(),
        )
        other |= set(ranked[: (2 * len(ranked) + 4) // 8])
    assert found != other


def test_share_1_drops_every_row():
    found = dropped(TOY_TRAIN, Fraction(1), seed=0, edges=TOY.edges)
    assert sorted(found) == sorted(TOY_TRAIN.index)
