"""The synthetic-share sweep's dropped rows: a share of each cell's training rows, by hash rank."""

from __future__ import annotations

import hashlib
import itertools
from fractions import Fraction

import pytest

from bko.datasets.cells import CELL_COLUMNS, cell_keys
from bko.datasets.split import dropped, dropped_count, require_share, share_text, subsampled
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


@pytest.mark.parametrize("share", [Fraction(0), Fraction(3, 2), 0.5, 1, True])
def test_the_dropped_count_refuses_a_share_outside_the_unit_interval(share):
    with pytest.raises(ValueError, match=r"synthetic share .* must be a Fraction in \(0, 1\]"):
        dropped_count(10, share)
    with pytest.raises(ValueError, match=r"synthetic share .* must be a Fraction in \(0, 1\]"):
        dropped(TOY_TRAIN, share, seed=0, edges=TOY.edges)


def test_a_share_of_one_is_a_fraction():
    require_share(Fraction(1))
    assert share_text(Fraction(1)) == "1/1"
    assert share_text(Fraction(2, 100)) == "1/50"


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
    # At a quarter, each 20-row cell drops 5 rows and `subsampled` keeps 5: the same 5 in every
    # cell were the tag `train-fraction:`, and the split's own lowest 5 were it empty.
    share = Fraction(1, 4)
    found = set(dropped(TOY_TRAIN, share, seed=0, edges=TOY.edges))
    kept = set(subsampled(TOY_TRAIN, share, seed=0, edges=TOY.edges).index)
    cells = cell_keys(TOY_TRAIN, edges=TOY.edges).groupby(list(CELL_COLUMNS)).groups
    full = [set(records) for records in cells.values() if len(records) == 20]
    assert len(full) == 10
    for records in full:
        untagged = sorted(records, key=lambda r: hashlib.sha256(f"0:{r}".encode()).digest())[:5]
        assert len(found & records) == len(kept & records) == 5
        assert found & records != kept & records
        assert found & records != set(untagged)


def test_share_1_drops_every_row():
    found = dropped(TOY_TRAIN, Fraction(1), seed=0, edges=TOY.edges)
    assert sorted(found) == sorted(TOY_TRAIN.index)
