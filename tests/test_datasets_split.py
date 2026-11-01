"""`bko.datasets.split.subsampled`: a share of each cell's training rows, drawn by hash."""

from __future__ import annotations

import hashlib
import itertools
from fractions import Fraction

import pandas as pd
import pytest

from bko.datasets.cells import cell_counts, cell_keys
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
    with pytest.raises(ValueError, match="must be a Fraction in"):
        subsampled(TRAIN, fraction, seed=0, edges=EDGES)


def test_the_draw_is_per_cell_on_unequal_cells():
    sizes = {(44, 0): 1, (44, 1): 3, (63, 0): 10, (63, 1): 23}
    rows = []
    for (age, sex), size in sizes.items():
        rows += [{"age": age, "sex": sex, "target": 1}] * size
    frame = pd.DataFrame(rows)
    kept = subsampled(frame, Fraction(1, 4), seed=0, edges=EDGES)
    found = sorted(cell_counts(kept, edges=EDGES)["count"].tolist())
    assert [n for n in found if n] == sorted(kept_count(n, Fraction(1, 4)) for n in sizes.values())


def test_the_subsample_ranks_apart_from_the_split():
    """The draw's own tag keeps it from being the first rows of the split's ranking."""
    for seed in range(5):
        kept = set(subsampled(TRAIN, Fraction(1, 4), seed=seed, edges=EDGES).index)
        ranked = sorted(TRAIN.index, key=lambda r: hashlib.sha256(f"{seed}:{r}".encode()).digest())
        cells = cell_keys(TRAIN, edges=EDGES)
        firsts = set()
        for _, group in cells.groupby(list(cells.columns)):
            members = [r for r in ranked if r in set(group.index)]
            firsts |= set(members[: kept_count(len(members), Fraction(1, 4))])
        assert kept != firsts


@pytest.mark.parametrize("fraction", [0.25, "1/4"])
def test_a_fraction_that_is_not_a_fraction_is_refused(fraction):
    with pytest.raises(ValueError, match="must be a Fraction"):
        subsampled(TRAIN, fraction, seed=0, edges=EDGES)


def test_a_duplicate_index_is_refused():
    with pytest.raises(ValueError, match="unique record index"):
        subsampled(pd.concat([TRAIN, TRAIN.iloc[:1]]), Fraction(1, 2), seed=0, edges=EDGES)
