"""The Gaussian copula and sequential CART: the contract, the seed, and other frames."""

from __future__ import annotations

import functools
import logging
import re
import types

import numpy as np
import pandas as pd
import pytest
import torch

import bko.baselines.classical as classical_module
from bko.baselines.classical import _predictors, cart, gaussian_copula
from bko.baselines.errors import ShortfallError
from bko.baselines.fills import CURRENT_FILL, Fill
from bko.baselines.records import unit_weights
from bko.baselines.request import Request, Strata, Stratum
from bko.baselines.synthesizers import fill
from bko.datasets.uci import UCI_FRAME
from tests._baselines_fixtures import EDGE, QUOTA, assert_contract, full_records
from tests._frame_fixtures import SHORT, TOY, TOY_TRAIN


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TRAIN = full_records()


def _run(method, ambient: int, seed: int, train: pd.DataFrame = TRAIN, quota=QUOTA):
    np.random.seed(ambient)
    torch.manual_seed(ambient)
    return method(train, Request(quota=quota, seed=seed, frame=UCI_FRAME))


@pytest.fixture(scope="module")
def copula_runs():
    return _run(gaussian_copula, 1, 3), _run(gaussian_copula, 2, 4)


@pytest.fixture(scope="module")
def cart_runs():
    return _run(cart, 1, 3), _run(cart, 2, 3), _run(cart, 1, 4)


def test_the_copula_meets_the_contract(copula_runs):
    assert_contract(copula_runs[0].added, TRAIN, QUOTA)
    pd.testing.assert_series_equal(copula_runs[0].weights, unit_weights(TRAIN))


def test_the_copula_ignores_the_seed(copula_runs):
    """SDV fixes the copula's sampling seed, so neither the request's seed nor the ambient
    generators move the rows."""
    pd.testing.assert_frame_equal(copula_runs[0].added, copula_runs[1].added)


def test_cart_meets_the_contract(cart_runs):
    assert_contract(cart_runs[0].added, TRAIN, QUOTA)
    pd.testing.assert_series_equal(cart_runs[0].weights, unit_weights(TRAIN))


def test_cart_rows_depend_on_the_seed_alone(cart_runs):
    pd.testing.assert_frame_equal(cart_runs[0].added, cart_runs[1].added)


def test_cart_differs_at_another_seed(cart_runs):
    assert not cart_runs[0].added.equals(cart_runs[2].added)


def test_cart_restores_the_ambient_rngs():
    np.random.seed(11)
    torch.manual_seed(11)
    cart(TRAIN, Request(quota=QUOTA, seed=3, frame=UCI_FRAME))
    after = (np.random.random(), torch.rand(1).item())
    np.random.seed(11)
    torch.manual_seed(11)
    assert after == (np.random.random(), torch.rand(1).item())


def test_cart_copies_values_from_same_leaf_donors():
    """`thal` follows `cp` exactly in training (cp 1 -> 3, cp 4 -> 7, others 6), with enough rows
    per value for pure leaves; a donor drawn from the whole split would break the mapping."""
    train = full_records(per_cell=30)
    train["thal"] = train["cp"].map({1: 3, 4: 7}).fillna(6).astype(train["thal"].dtype)
    added = cart(train, Request(quota=QUOTA, seed=0, frame=UCI_FRAME)).added
    expected = added["cp"].map({1: 3, 4: 7}).fillna(6)
    assert (added["thal"] == expected).all()


@pytest.mark.parametrize("method", [gaussian_copula, cart], ids=["copula", "cart"])
def test_each_method_fills_the_toy_frame(method):
    quota: dict[Stratum, int] = dict.fromkeys(SHORT, 3)
    request = Request(quota=quota, strata=Strata.CELL, seed=0, frame=TOY)
    added = method(TOY_TRAIN, request).added
    assert list(added.columns) == list(TOY.columns)
    assert len(added) == sum(quota.values())


def test_cart_records_synthpops_settings_and_visit_order(cart_runs):
    assert cart_runs[0].params == {
        "seed": 3,
        "visit": list(UCI_FRAME.columns),
        "min_samples_leaf": 5,
        "min_samples_split": 15,
        "max_depth": 30,
        "complexity": 1e-8,
    }


def test_cart_one_hot_codes_a_predictor_of_three_or_more_values():
    rows = TRAIN.iloc[:6]
    predictors = _predictors(rows, ["sex", "cp"], UCI_FRAME)
    assert predictors.shape == (6, 1 + len(UCI_FRAME.codebook["cp"]))
    assert (predictors[:, 1:].sum(axis=1) == 1).all()


def test_cart_draws_the_first_column_from_its_training_values():
    """Nine in ten `<55` rows are aged 44; drawing the first column from the distinct ages instead
    of the training values would give 44 about half the `<55` rows."""
    train = full_records(per_cell=30)
    young = train.index[train["age"] == EDGE - 1]
    train.loc[young.difference(young[::10]), "age"] = 44
    quota: dict[Stratum, int] = {(f"<{EDGE}", "female", 1): 40, (f"<{EDGE}", "male", 0): 40}
    ages = cart(train, Request(quota=quota, seed=0, frame=UCI_FRAME)).added["age"]
    assert (ages == 44).mean() > 0.75


def test_cart_copies_measured_values_from_training_rows(cart_runs):
    added = cart_runs[0].added
    for field in ("trestbps", "chol", "thalach", "oldpeak"):
        assert set(added[field]) <= set(TRAIN[field])


@pytest.mark.parametrize(
    ("method", "name"),
    [(gaussian_copula, "gaussian-copula"), (cart, "cart")],
    ids=["copula", "cart"],
)
def test_each_classical_method_logs_a_fit_line_in_ctgans_format(method, name, caplog):
    quota: dict[Stratum, int] = dict.fromkeys(SHORT, 3)
    request = Request(quota=quota, strata=Strata.CELL, seed=0, frame=TOY)
    token = CURRENT_FILL.set(Fill(3, "replica"))
    try:
        with caplog.at_level(logging.INFO):
            method(TOY_TRAIN, request)
    finally:
        CURRENT_FILL.reset(token)
    fits = [r.getMessage() for r in caplog.records if r.name == classical_module.logger.name]
    fits = [m for m in fits if m.startswith("fit ")]
    assert len(fits) == 1
    assert re.fullmatch(rf"fit holdout=3 fill=replica method={name} seconds=\d+\.\d", fits[0])


def _ticking(monkeypatch) -> dict[str, float]:
    """Make the module's clock a counter that only the test's own steps advance."""
    clock = {"now": 0.0}
    monkeypatch.setattr(
        classical_module, "time", types.SimpleNamespace(perf_counter=lambda: clock["now"])
    )
    return clock


def _fit_seconds(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == classical_module.logger.name and r.getMessage().startswith("fit ")
    ]


def test_the_copula_fit_line_spans_its_fit_alone(monkeypatch, caplog):
    clock = _ticking(monkeypatch)
    original = classical_module.GaussianCopulaSynthesizer.fit

    def slow_fit(self, data):
        clock["now"] += 7.5
        return original(self, data)

    monkeypatch.setattr(classical_module.GaussianCopulaSynthesizer, "fit", slow_fit)
    request = Request(quota=dict.fromkeys(SHORT, 3), strata=Strata.CELL, seed=0, frame=TOY)
    with caplog.at_level(logging.INFO):
        gaussian_copula(TOY_TRAIN, request)
    assert _fit_seconds(caplog) == ["fit holdout=-1 fill=none method=gaussian-copula seconds=7.5"]


def test_the_cart_fit_line_spans_every_columns_tree_and_leaf_assignment(monkeypatch, caplog):
    clock = _ticking(monkeypatch)
    tree, predictors = classical_module._tree, classical_module._predictors

    def slow_tree(*args, **kwargs):
        clock["now"] += 10.0
        return tree(*args, **kwargs)

    def slow_predictors(*args, **kwargs):
        clock["now"] += 100.0
        return predictors(*args, **kwargs)

    monkeypatch.setattr(classical_module, "_tree", slow_tree)
    monkeypatch.setattr(classical_module, "_predictors", slow_predictors)
    request = Request(quota=dict.fromkeys(SHORT, 3), strata=Strata.CELL, seed=0, frame=TOY)
    with caplog.at_level(logging.INFO):
        cart(TOY_TRAIN, request)
    # Per later column: the tree's 10, its own predictors' 100, and the leaf assignment's 100.
    later = len(TOY.columns) - 1
    expected = f"fit holdout=-1 fill=none method=cart seconds={210.0 * later:.1f}"
    assert _fit_seconds(caplog) == [expected]


@pytest.mark.parametrize("method", [gaussian_copula, cart], ids=["copula", "cart"])
def test_a_classical_or_network_shortfall_holds_its_baseline(monkeypatch, method):
    monkeypatch.setattr(classical_module, "fill", functools.partial(fill, batch=10, limit=10))
    quota: dict[Stratum, int] = {(f">={EDGE}", "female", 1): 10}
    with pytest.raises(ShortfallError) as caught:
        method(TRAIN, Request(quota=quota, seed=3, frame=UCI_FRAME))
    short = caught.value
    assert short.baseline is not None
    assert set(short.short) == {f">={EDGE}/female/1"}
    assert 0 < short.short[f">={EDGE}/female/1"] <= 10
    pd.testing.assert_frame_equal(short.baseline.added, short.added)
    pd.testing.assert_series_equal(short.baseline.weights, unit_weights(TRAIN))
    assert short.baseline.params["short"] == dict(short.short)
    assert len(short.baseline.params) > 1
