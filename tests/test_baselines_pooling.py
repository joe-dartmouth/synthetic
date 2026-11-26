"""The pooled retained-data methods: `row-resample`, `marginals` and `eb-marginals`."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest

from bko.baselines.categories import categorise
from bko.baselines.errors import BaselineError
from bko.baselines.pooling import (
    KAPPA_GRID,
    ROW_RESAMPLE_STREAM,
    cell_weight,
    eb_marginals,
    fit_kappa,
    marginals,
    row_resample,
)
from bko.baselines.records import check
from bko.baselines.request import Request, Strata, Stratum
from bko.datasets.cells import LABEL
from bko.datasets.uci import UCI_FRAME
from bko.gaps.shares import GROUP_WEIGHT
from tests._baselines_fixtures import full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

FRAME = dataclasses.replace(UCI_FRAME, edges=(50, 60))
"""UCI's frame banded `<50`, `50-59` and `>=60`, so the fixture's ages 44, 54, 55 and 63 make
every label group span three bands."""

TRAIN = full_records()

QUOTA: dict[Stratum, int] = {
    ("<50", "female", 1): 3,
    ("50-59", "male", 1): 4,
    (">=60", "female", 0): 2,
}
"""Rows to add in three cells, one per band."""

RUNNERS = (row_resample, marginals, eb_marginals)
IDS = ("row-resample", "marginals", "eb-marginals")


def banded(train: pd.DataFrame, band: str, label: int) -> pd.DataFrame:
    """The label group's records in `band`, banded by hand for `FRAME`."""
    age = train["age"]
    inside = {"<50": age < 50, "50-59": (age >= 50) & (age < 60), ">=60": age >= 60}[band]
    return train.loc[inside & (train[LABEL] == label)]


@pytest.mark.parametrize("runner", RUNNERS, ids=IDS)
def test_every_added_row_passes_check(runner):
    request = Request(quota=QUOTA, frame=FRAME)
    check(runner(TRAIN, request), TRAIN, request)


@pytest.mark.parametrize("runner", RUNNERS, ids=IDS)
def test_an_empty_cell_reads_the_banded_label_group_only(runner, caplog):
    cell = ("<50", "female", 1)
    emptied = TRAIN.loc[~((TRAIN["age"] < 50) & (TRAIN["sex"] == 0) & (TRAIN[LABEL] == 1))]
    request = Request(quota={cell: 12}, frame=FRAME)
    with caplog.at_level(logging.INFO, logger="bko.baselines.pooling"):
        added = runner(emptied, request).added
    group = banded(emptied, "<50", 1)
    for name in FRAME.columns:
        if name != "sex":
            assert set(added[name]) <= set(group[name]), name
    assert (added["sex"] == 0).all()
    assert "pool holdout=-1 fill=none key=<50/female/1 own=0 banded=7" in caplog.messages


def test_a_full_cell_weighs_its_own_rows_at_n_over_n_plus_20():
    assert GROUP_WEIGHT == 20
    assert cell_weight(19, 20.0) == 19 / 39  # ruff: ignore[float-equality-comparison]
    assert cell_weight(19, float(GROUP_WEIGHT)) == 19 / 39  # ruff: ignore[float-equality-comparison]
    assert cell_weight(0, 20.0) == 0.0  # ruff: ignore[float-equality-comparison]


def test_row_resample_copies_whole_rows():
    request = Request(quota=QUOTA, frame=FRAME)
    added = row_resample(TRAIN, request).added
    start = 0
    for (band, _, label), rows in request.quota.items():
        part = added.iloc[start : start + rows].drop(columns="sex")
        start += rows
        held = banded(TRAIN, str(band), int(label)).drop(columns="sex")
        held_rows = {tuple(row) for row in held.itertuples(index=False)}
        assert {tuple(row) for row in part.itertuples(index=False)} <= held_rows


def test_row_resample_follows_its_stream():
    cell = ("50-59", "male", 1)
    request = Request(quota={cell: 6}, seed=7, frame=FRAME)
    added = row_resample(TRAIN, request).added.reset_index(drop=True)
    group = banded(TRAIN, "50-59", 1)
    own = group.loc[group["sex"] == 1]
    weight = len(own) / (len(own) + GROUP_WEIGHT)
    rng = np.random.default_rng([7, ROW_RESAMPLE_STREAM])
    expected = []
    for _ in range(6):
        pool = own if rng.random() < weight else group
        expected.append(pool.iloc[[int(rng.integers(len(pool)))]])
    frame = pd.concat(expected).reset_index(drop=True).assign(sex=1)
    pd.testing.assert_frame_equal(added, frame.loc[:, list(FRAME.columns)])


def test_marginals_breaks_a_planted_pair():
    planted = TRAIN.assign(chol=TRAIN["trestbps"])
    request = Request(quota={("50-59", "male", 1): 30}, frame=FRAME)
    added = marginals(planted, request).added
    assert (added["chol"] != added["trestbps"]).any()


def test_fit_kappa_pools_fully_when_cells_agree():
    prior = {"a": 0.5, "b": 0.5}
    columns = [[({"a": 5, "b": 5}, prior), ({"a": 3, "b": 3}, prior)]]
    assert fit_kappa(columns) == KAPPA_GRID[-1]


def test_fit_kappa_pools_nothing_when_cells_are_disjoint():
    columns = [
        [
            ({"a": 10}, {"a": 0.01, "b": 0.99}),
            ({"b": 10}, {"a": 0.99, "b": 0.01}),
        ]
    ]
    assert fit_kappa(columns) == KAPPA_GRID[0]


def test_the_kappa_grid_is_101_log_spaced_values_from_a_tenth_to_ten_thousand():
    assert len(KAPPA_GRID) == 101
    assert KAPPA_GRID[0] == pytest.approx(0.1)
    assert KAPPA_GRID[-1] == pytest.approx(10000.0)
    assert np.allclose(np.diff(np.log(KAPPA_GRID)), np.log(10.0) / 20)


def test_a_one_category_column_does_not_move_kappa():
    base = [
        [
            ({"a": 6, "b": 4}, {"a": 0.4, "b": 0.6}),
            ({"a": 2, "b": 8}, {"a": 0.6, "b": 0.4}),
        ]
    ]
    flat = [({"x": 7}, {"x": 1.0}), ({"x": 3}, {"x": 1.0})]
    assert fit_kappa([*base, flat]) == fit_kappa(base)


def test_a_zero_heavy_count_keeps_its_values_as_categories():
    values = pd.Series([0] * 340 + [value for value in range(1, 12) for _ in range(6)])
    assert values.nunique() == 12
    categories = categorise(values, values, ordered=True)
    assert categories.nunique() == 12
    assert values.groupby(categories).nunique().eq(1).all()


def test_a_wide_column_falls_into_ten_decile_bins():
    values = pd.Series(np.arange(200.0))
    categories = categorise(values, values, ordered=True)
    assert categories.nunique() == 10
    assert categories.value_counts().eq(20).all()


def test_an_ordered_column_merges_its_upper_tail():
    values = pd.Series([1] * 10 + [2] * 6 + [3] * 2 + [4] * 1)
    categories = categorise(values, values, ordered=True)
    assert categories.nunique() == 2
    assert categories.value_counts().min() >= 5
    assert len(set(categories[values >= 2])) == 1


def test_an_unordered_column_folds_rare_codes_into_other():
    values = pd.Series([0] * 8 + [1] * 5 + [2] * 3 + [3] * 1)
    categories = categorise(values, values, ordered=False)
    assert categories.tolist() == [0] * 8 + [1] * 5 + ["other"] * 4


@pytest.mark.parametrize("runner", RUNNERS, ids=IDS)
def test_a_request_by_label_is_refused(runner):
    request = Request(quota={(1,): 3}, strata=Strata.LABEL, frame=FRAME)
    with pytest.raises(BaselineError, match="keyed by cell"):
        runner(TRAIN, request)


def test_eb_marginals_logs_one_kappa_from_its_grid(caplog):
    request = Request(quota=QUOTA, frame=FRAME)
    with caplog.at_level(logging.INFO, logger="bko.baselines.pooling"):
        baseline = eb_marginals(TRAIN, request)
    lines = [line for line in caplog.messages if line.startswith("eb ")]
    assert lines == [f"eb holdout=-1 fill=none kappa={baseline.params['kappa']}"]
    assert baseline.params["kappa"] in KAPPA_GRID


@pytest.mark.parametrize("runner", RUNNERS, ids=IDS)
def test_each_method_is_deterministic_at_a_seed(runner):
    request = Request(quota=QUOTA, seed=3, frame=FRAME)
    pd.testing.assert_frame_equal(runner(TRAIN, request).added, runner(TRAIN, request).added)


@pytest.mark.parametrize("runner", RUNNERS, ids=IDS)
def test_a_retry_seed_changes_the_fill(runner):
    first = runner(TRAIN, Request(quota=QUOTA, seed=3, frame=FRAME)).added
    retry = runner(TRAIN, Request(quota=QUOTA, seed=4, frame=FRAME)).added
    assert not first.equals(retry)
