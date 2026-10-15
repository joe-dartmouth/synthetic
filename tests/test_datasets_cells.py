"""The cell grid, the underrepresentation rule and the cell-stratified split."""

from __future__ import annotations

import collections
import hashlib
import warnings

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import (
    AGE_BAND_EDGES,
    COUNT_COLUMNS,
    UNDERREPRESENTED_BELOW,
    age_band_labels,
    cell_counts,
    cell_keys,
)
from bko.datasets.errors import ContractError
from bko.datasets.split import SPLIT_SEED, held_out_count, split
from tests._datasets_fixtures import records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

EDGE = AGE_BAND_EDGES[0]
YOUNG = EDGE - 1
OLD = EDGE
GRID = [(band, sex, t) for band in age_band_labels() for sex in ("female", "male") for t in (0, 1)]


def test_the_edge_age_starts_the_older_band():
    keys = cell_keys(records({(EDGE - 1, 0, 0): 1, (EDGE, 0, 0): 1}))
    assert list(keys["age_band"]) == list(age_band_labels())


def test_band_labels_for_several_edges():
    assert age_band_labels((40, 50, 60)) == ("<40", "40-49", "50-59", ">=60")


@pytest.mark.parametrize("edges", [(), (60, 40), (40, 40)])
def test_empty_or_unsorted_edges_are_refused(edges):
    with pytest.raises(ValueError, match="strictly increasing"):
        age_band_labels(edges)


def test_the_grid_lists_every_cell_including_empty_ones():
    counts = cell_counts(records({(YOUNG, 1, 0): 3}))
    assert list(counts.columns) == list(COUNT_COLUMNS)
    assert list(zip(counts["age_band"], counts["sex"], counts["target"], strict=True)) == GRID
    assert counts["count"].sum() == 3
    empty = counts.loc[counts["count"] == 0]
    assert len(empty) == len(GRID) - 1
    assert empty["underrepresented"].all()


def test_the_rule_is_fewer_than_twenty():
    """The 2-pager's figure, pinned once; every other case derives from the constant."""
    counts = cell_counts(records({(YOUNG, 0, 0): 19, (YOUNG, 0, 1): 20}))
    assert list(counts["underrepresented"][:2]) == [True, False]


def test_a_cell_at_the_threshold_is_not_underrepresented_and_one_below_is():
    below = UNDERREPRESENTED_BELOW
    counts = cell_counts(records({(OLD, 1, 0): below - 1, (OLD, 1, 1): below}))
    assert bool(counts.loc[counts["count"] == below - 1, "underrepresented"].item())
    assert not bool(counts.loc[counts["count"] == below, "underrepresented"].item())


def test_a_given_threshold_replaces_the_default():
    counts = cell_counts(records({(OLD, 1, 0): 4, (OLD, 1, 1): 5}), below=5)
    assert counts.loc[counts["count"] > 0, "underrepresented"].tolist() == [True, False]


def test_given_edges_band_the_ages_and_the_grid():
    """An independent answer for edges (40, 60): each boundary age starts the older band."""
    frame = records({(39, 0, 0): 1, (40, 0, 0): 2, (59, 0, 0): 3, (60, 0, 0): 4})
    counts = cell_counts(frame, edges=(40, 60))
    female_healthy = counts.loc[(counts["sex"] == "female") & (counts["target"] == 0)]
    assert female_healthy[["age_band", "count"]].values.tolist() == [
        ["<40", 1],
        ["40-59", 5],
        [">=60", 4],
    ]
    assert len(counts) == 12


@pytest.mark.parametrize("age", [float("nan"), float("inf")])
def test_a_missing_or_infinite_age_is_refused(age):
    frame = records({(YOUNG, 0, 0): 2}).astype({"age": "float64"})
    frame.loc[2, "age"] = age
    with pytest.raises(ContractError, match="age of record 2 is"):
        cell_keys(frame)


def test_a_refusal_names_one_value_even_with_a_duplicate_index():
    frame = records({(YOUNG, 0, 0): 2})
    frame.index = pd.Index([1, 1], name="record_id")
    frame["sex"] = [5, 0]
    with pytest.raises(ContractError, match=r"^sex of record 1 is 5, not in \[0, 1\]$"):
        cell_keys(frame)


def test_counts_do_not_depend_on_row_order():
    frame = records({(YOUNG, 0, 0): 4, (OLD, 1, 1): 7, (OLD, 0, 1): 2})
    shuffled = frame.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(cell_counts(frame), cell_counts(shuffled))


@pytest.mark.parametrize("seed", range(25))
def test_counts_match_a_counter_oracle(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    frame = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n),
            "sex": rng.integers(0, 2, n),
            "target": rng.integers(0, 2, n),
        }
    )
    oracle = collections.Counter(
        (f"<{EDGE}" if age < EDGE else f">={EDGE}", ("female", "male")[sex], target)
        for age, sex, target in zip(frame["age"], frame["sex"], frame["target"], strict=True)
    )
    counts = cell_counts(frame)
    got = dict(
        zip(
            zip(counts["age_band"], counts["sex"], counts["target"], strict=True),
            counts["count"],
            strict=True,
        )
    )
    assert got == {cell: oracle.get(cell, 0) for cell in GRID}


@pytest.mark.parametrize(("column", "value"), [("sex", 2), ("target", 2)])
def test_a_sex_or_label_outside_its_codes_is_refused(column, value):
    frame = records({(YOUNG, 0, 0): 2})
    frame.loc[2, column] = value
    with pytest.raises(ContractError, match=f"{column} of record 2"):
        cell_keys(frame)


@pytest.mark.parametrize(
    ("size", "held"),
    [(1, 0), (2, 0), (3, 1), (5, 1), (7, 1), (8, 2), (23, 5), (24, 5), (25, 5), (72, 14)],
)
def test_held_out_count_is_one_fifth_rounded_half_up(size, held):
    assert held_out_count(size) == held


SPEC = {
    (YOUNG, 0, 0): 1,
    (YOUNG, 0, 1): 2,
    (YOUNG, 1, 0): 3,
    (OLD, 0, 0): 24,
    (OLD, 0, 1): 25,
    (OLD, 1, 1): 7,
}


def test_split_partitions_the_records():
    frame = records(SPEC)
    parts = split(frame)
    assert set(parts.train.index).isdisjoint(parts.test.index)
    assert sorted([*parts.train.index, *parts.test.index]) == list(frame.index)


def test_each_cell_holds_out_its_rounded_fifth():
    frame = records(SPEC)
    parts = split(frame)
    full = cell_counts(frame).set_index(["age_band", "sex", "target"])["count"]
    held = cell_counts(parts.test).set_index(["age_band", "sex", "target"])["count"]
    assert held.to_dict() == {cell: held_out_count(n) for cell, n in full.items()}


def test_training_counts_are_the_same_for_every_seed():
    frame = records(SPEC)
    baseline = cell_counts(split(frame, seed=0).train)
    for seed in range(1, 10):
        pd.testing.assert_frame_equal(cell_counts(split(frame, seed=seed).train), baseline)


def test_the_same_seed_gives_the_same_split():
    frame = records(SPEC)
    assert list(split(frame, seed=3).test.index) == list(split(frame, seed=3).test.index)
    assert split(frame, seed=3).seed == 3


def test_split_stratifies_on_the_given_edges():
    frame = records({(39, 0, 0): 10, (45, 0, 0): 10})
    parts = split(frame, edges=(40,))
    held = cell_counts(parts.test, edges=(40,))
    assert held.loc[held["count"] > 0, "count"].tolist() == [2, 2]


def test_the_parts_can_be_changed_without_a_warning_or_touching_the_frame():
    frame = records(SPEC)
    parts = split(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parts.train["age"] = 0
        parts.test["age"] = 0
    assert (frame["age"] != 0).all()


def test_a_different_seed_gives_a_different_split():
    """The positive control: without it a split that ignored its seed would pass the test above."""
    frame = records(SPEC)
    tests = {tuple(split(frame, seed=seed).test.index) for seed in range(10)}
    assert len(tests) > 1


def test_split_does_not_depend_on_row_order():
    frame = records(SPEC)
    shuffled = frame.sample(frac=1, random_state=7)
    assert sorted(split(frame).test.index) == sorted(split(shuffled).test.index)


def test_the_split_is_pinned():
    """Golden ids for one 20-record cell at seed 0, computed with hashlib on 2026-10-14."""
    parts = split(records({(YOUNG, 0, 0): 20}), seed=SPLIT_SEED)
    assert sorted(parts.test.index) == [7, 15, 18, 20]
    assert parts.fingerprint == hashlib.sha256(b"7,15,18,20").hexdigest()


def test_a_duplicate_record_id_is_refused():
    frame = records({(YOUNG, 0, 0): 3})
    frame.index = pd.Index([1, 1, 2], name="record_id")
    with pytest.raises(ValueError, match="unique record index"):
        split(frame)
