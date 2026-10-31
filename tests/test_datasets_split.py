"""`bko.datasets.split.subsampled`: a share of each cell's training rows, drawn by hash."""

from __future__ import annotations

import itertools
from fractions import Fraction

import pytest

from bko.datasets.cells import cell_counts
from bko.datasets.split import kept_count, split, subsampled
from bko.datasets.uci import UCI_FRAME
from tests._baselines_fixtures import full_records


RECORDS = full_records(20)
EDGES = UCI_FRAME.edges
TRAIN = split(RECORDS, seed=0, edges=EDGES).train


def counts(rows):
    return cell_counts(rows, edges=EDGES)["count"].tolist()


def test_kept_count_rounds_half_up_exactly():
    assert kept_count(10, Fraction(1, 4)) == 3
    # 0.7 * 45 is 31.499999999999996 in floats; the exact half rounds up.
    assert 0.7 * 45 < 31.5
    assert kept_count(45, Fraction(7, 10)) == 32


def test_kept_count_never_below_two():
    assert kept_count(3, Fraction(1, 4)) == 2


def test_a_one_row_cell_keeps_its_row():
    assert kept_count(1, Fraction(1, 4)) == 1


def test_fraction_one_is_the_identity():
    kept = subsampled(TRAIN, Fraction(1), seed=0, edges=EDGES)
    assert kept.index.equals(TRAIN.index)


def test_each_cell_keeps_its_kept_count():
    kept = subsampled(TRAIN, Fraction(1, 2), seed=0, edges=EDGES)
    assert counts(kept) == [kept_count(n, Fraction(1, 2)) if n else 0 for n in counts(TRAIN)]


def test_subsamples_nest_across_fractions():
    sizes = [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
    kept = [set(subsampled(TRAIN, f, seed=3, edges=EDGES).index) for f in sizes]
    assert all(small <= large for small, large in itertools.pairwise(kept))


def test_the_draw_differs_by_seed():
    first = subsampled(TRAIN, Fraction(1, 2), seed=0, edges=EDGES)
    second = subsampled(TRAIN, Fraction(1, 2), seed=1, edges=EDGES)
    assert set(first.index) != set(second.index)
    assert counts(first) == counts(second)


def test_the_test_rows_are_untouched():
    parts = split(RECORDS, seed=0, edges=EDGES)
    before = parts.test.copy()
    subsampled(parts.train, Fraction(1, 4), seed=0, edges=EDGES)
    assert parts.test.equals(before)
    assert set(subsampled(parts.train, Fraction(1, 4), seed=0, edges=EDGES).index) <= set(
        parts.train.index
    )


@pytest.mark.parametrize("fraction", [Fraction(0), Fraction(5, 4)])
def test_a_fraction_outside_zero_to_one_is_refused(fraction):
    with pytest.raises(ValueError, match="must lie in"):
        subsampled(TRAIN, fraction, seed=0, edges=EDGES)
