"""The classifiers: codebook features, weights to the model only, no silent non-convergence, and a
boosting model that no seed or thread count moves."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

from bko.baselines.methods import augment
from bko.baselines.records import training_set
from bko.baselines.request import Method, Request
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME, complete_records, parse_cleveland
from bko.evaluate.classifier import (
    FIXED_EPOCHS,
    MLP_EPOCHS,
    MODELS,
    PANEL,
    Model,
    build,
    build_et,
    build_hgb,
    build_mlp,
    build_nb,
    build_rf,
    probabilities,
)
from bko.evaluate.context import Context
from bko.evaluate.errors import EvaluationError
from bko.evaluate.run import _fitted
from tests._baselines_fixtures import UNBALANCED_TEXT
from tests._evaluate_fixtures import FAST_METHODS, FLAGGED, GRADED, Spy


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

UNBALANCED = split(complete_records(parse_cleveland(UNBALANCED_TEXT)), edges=UCI_FRAME.edges).train
TRAIN = split(GRADED, seed=0, edges=UCI_FRAME.edges).train


def test_class_weights_reach_the_regression_row_by_row():
    spy = Spy()
    baseline = augment(UNBALANCED, Method.CLASS_WEIGHT, Request(frame=UCI_FRAME))
    shuffled = training_set(UNBALANCED, baseline, frame=UCI_FRAME).sample(frac=1.0, random_state=0)
    probabilities(shuffled, UNBALANCED, factory=spy, frame=UCI_FRAME)
    labels, weights = spy.models[0].seen
    assert weights == pytest.approx(np.where(labels == 0, 36 / 24, 36 / 48))


def test_added_rows_follow_the_real_rows_at_weight_one():
    request = Request(quota={FLAGGED[0]: 3}, seed=0, frame=UCI_FRAME)
    baseline = augment(TRAIN, Method.SMOTE, request, methods=FAST_METHODS)
    spy = Spy()
    probabilities(
        training_set(TRAIN, baseline, frame=UCI_FRAME), TRAIN, factory=spy, frame=UCI_FRAME
    )
    labels, weights = spy.models[0].seen
    assert len(weights) == len(TRAIN) + 3
    assert weights[-3:].tolist() == [1.0] * 3
    assert labels[-3:].tolist() == [1] * 3


def test_every_code_has_a_column_whatever_the_training_rows_hold():
    narrow = TRAIN[TRAIN["restecg"] != 1].assign(weight=1.0)
    test = TRAIN[TRAIN["restecg"] == 1]
    assert not test.empty
    probability = probabilities(narrow, test, frame=UCI_FRAME)
    assert probability.between(0, 1).all()
    assert probability.index.equals(test.index)


def test_a_fit_that_does_not_converge_stops_the_run():
    def stingy(frame):
        pipeline = build(frame)
        pipeline.set_params(model__max_iter=1)
        return pipeline

    with pytest.raises(EvaluationError, match="did not converge"):
        probabilities(TRAIN.assign(weight=1.0), TRAIN, factory=stingy, frame=UCI_FRAME)


def test_probabilities_are_pinned():
    probability = probabilities(TRAIN.assign(weight=1.0), GRADED, frame=UCI_FRAME)
    np.testing.assert_allclose(
        probability.to_numpy()[[0, 7, 23, 41]],
        [0.2743870822873947, 0.3788422046931614, 0.3911697315854433, 0.38625833787281055],
        rtol=1e-12,
    )


HGB_PARAMS = {
    "categorical_features": None,
    "class_weight": None,
    "early_stopping": False,
    "interaction_cst": None,
    "l2_regularization": 0.0,
    "learning_rate": 0.1,
    "loss": "log_loss",
    "max_bins": 255,
    "max_depth": None,
    "max_features": 1.0,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 1,
    "monotonic_cst": None,
    "n_iter_no_change": 10,
    "random_state": 0,
    "scoring": "loss",
    "tol": 1e-07,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": False,
}


def _hgb(training, test=GRADED, **params):
    def factory(frame):
        pipeline = build_hgb(frame)
        pipeline.set_params(**{f"model__{name}": value for name, value in params.items()})
        return pipeline

    return probabilities(training, test, factory=factory, frame=UCI_FRAME).to_numpy()


def test_models_name_the_regression_and_the_boosting_model():
    assert list(MODELS) == [Model.LR, Model.HGB]
    assert [model.value for model in MODELS] == ["lr", "hgb"]
    assert MODELS[Model.LR] is build
    assert MODELS[Model.HGB] is build_hgb


def test_the_boosting_model_settings_are_pinned():
    assert build_hgb(frame=UCI_FRAME).named_steps["model"].get_params() == HGB_PARAMS


def test_a_boosting_leaf_is_smaller_than_the_gap_fills_smallest_cell():
    assert build_hgb(frame=UCI_FRAME).named_steps["model"].min_samples_leaf < 2


def test_two_boosting_fits_score_the_same():
    training = TRAIN.assign(weight=1.0)
    np.testing.assert_array_equal(_hgb(training), _hgb(training))


def test_the_boosting_models_seed_moves_nothing():
    training = TRAIN.assign(weight=1.0)
    np.testing.assert_array_equal(_hgb(training), _hgb(training, random_state=12345))


def test_weights_reach_the_boosting_model():
    weights = np.where(TRAIN[UCI_FRAME.label].to_numpy() == 1, 3.0, 1.0)
    weighted = _hgb(TRAIN.assign(weight=weights))
    assert not np.array_equal(weighted, _hgb(TRAIN.assign(weight=1.0)))


def test_the_boosting_model_scores_the_same_on_four_threads_as_on_one():
    """Determinism only: `test_probabilities_fits_and_predicts_on_one_openmp_thread` guards the
    limit. The training rows are `TRAIN` 20 times, measured fields jittered, so each histogram
    pass has thousands of rows to split among the threads."""
    features = list(UCI_FRAME.features)
    numeric = list(UCI_FRAME.numeric)
    rng = np.random.default_rng(0)
    many = pd.concat([TRAIN] * 20, ignore_index=True)
    many[numeric] += rng.normal(0.0, 0.5, size=(len(many), len(numeric)))

    def fitted(threads):
        with threadpool_limits(limits=threads, user_api="openmp"):
            pipeline = build_hgb(frame=UCI_FRAME).fit(many.loc[:, features], many[UCI_FRAME.label])
            return pipeline.predict_proba(GRADED.loc[:, features])

    np.testing.assert_array_equal(fitted(4), fitted(1))


class ThreadReading(ClassifierMixin, BaseEstimator):
    """An estimator that records the OpenMP thread counts it sees while it fits and predicts."""

    def fit(self, features, labels, sample_weight=None):
        self.classes_ = np.unique(labels)
        self.fit_threads_ = _openmp_threads()
        return self

    def predict_proba(self, features):
        self.predict_threads_ = _openmp_threads()
        return np.full((len(features), 2), 0.5)


def _openmp_threads():
    return [pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "openmp"]


def test_probabilities_fits_and_predicts_on_one_openmp_thread():
    spy = ThreadReading()

    def factory(frame):
        pipeline = build_hgb(frame)
        pipeline.set_params(model=spy)
        return pipeline

    probabilities(TRAIN.assign(weight=1.0), GRADED, factory=factory, frame=UCI_FRAME)
    assert spy.fit_threads_
    assert spy.fit_threads_ == [1] * len(spy.fit_threads_)
    assert spy.predict_threads_ == [1] * len(spy.predict_threads_)


def test_boosting_probabilities_are_pinned():
    probability = probabilities(
        TRAIN.assign(weight=1.0), GRADED, factory=build_hgb, frame=UCI_FRAME
    )
    np.testing.assert_allclose(
        probability.to_numpy()[[3, 21, 33, 45]],
        [0.29854498130049056, 0.9832056845220146, 0.02037211227150123, 0.9018548910132792],
        rtol=1e-12,
    )


NEW = (Model.RF, Model.ET, Model.MLP, Model.NB)
"""The panel's models beyond the regression and the boosting model."""


def _panel(model, training, test=GRADED, on_fit=None):
    """`model`'s probabilities, with every thread pool at one thread, as `panel` runs it."""
    with threadpool_limits(limits=1):
        return probabilities(
            training, test, factory=PANEL[model], frame=UCI_FRAME, model=model, on_fit=on_fit
        )


def test_models_default_stays_lr_hgb():
    assert list(MODELS) == [Model.LR, Model.HGB]
    assert Context().models is MODELS


def test_the_panel_holds_every_model_in_order():
    assert list(PANEL) == [Model.LR, Model.HGB, Model.RF, Model.ET, Model.MLP, Model.NB]
    assert [PANEL[model] for model in PANEL] == [
        build,
        build_hgb,
        build_rf,
        build_et,
        build_mlp,
        build_nb,
    ]
    assert frozenset({Model.MLP}) == FIXED_EPOCHS


@pytest.mark.parametrize("model", NEW, ids=str)
def test_each_panel_model_fits_and_scores_a_toy_frame(model):
    probability = _panel(model, TRAIN.assign(weight=1.0))
    assert probability.index.equals(GRADED.index)
    assert probability.between(0, 1).all()
    assert probability.nunique() > 1


@pytest.mark.parametrize("model", NEW, ids=str)
def test_weights_reach_the_panel_model(model):
    weights = np.where(TRAIN[UCI_FRAME.label].to_numpy() == 1, 3.0, 1.0)
    weighted = _panel(model, TRAIN.assign(weight=weights)).to_numpy()
    assert not np.array_equal(weighted, _panel(model, TRAIN.assign(weight=1.0)).to_numpy())


RF_PARAMS = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "class_weight": None,
    "criterion": "gini",
    "max_depth": None,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": None,
    "n_estimators": 500,
    "n_jobs": 1,
    "oob_score": False,
    "random_state": 0,
    "verbose": 0,
    "warm_start": False,
}
ET_PARAMS = {**RF_PARAMS, "bootstrap": False}
MLP_PARAMS = {
    "activation": "relu",
    "alpha": 0.0001,
    "batch_size": 200,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "early_stopping": False,
    "epsilon": 1e-08,
    "hidden_layer_sizes": (100,),
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "max_fun": 15000,
    "max_iter": 15,
    "momentum": 0.9,
    "n_iter_no_change": 15,
    "nesterovs_momentum": True,
    "power_t": 0.5,
    "random_state": 0,
    "shuffle": True,
    "solver": "adam",
    "tol": 0.0,
    "validation_fraction": 0.1,
    "verbose": False,
    "warm_start": False,
}
NB_PARAMS = {
    "alpha": 1.0,
    "class_prior": None,
    "fit_prior": True,
    "force_alpha": True,
    "min_categories": [10] * len(UCI_FRAME.numeric)
    + [len(UCI_FRAME.codebook[name]) for name in UCI_FRAME.categorical],
}
BINS_PARAMS = {
    "dtype": None,
    "encode": "ordinal",
    "n_bins": 10,
    "quantile_method": "averaged_inverted_cdf",
    "random_state": None,
    "strategy": "uniform",
    "subsample": None,
}


def test_the_panel_model_settings_are_pinned():
    def params(factory):
        return factory(UCI_FRAME).named_steps["model"].get_params()

    assert params(build_rf) == RF_PARAMS
    assert params(build_et) == ET_PARAMS
    assert params(build_mlp) == MLP_PARAMS
    assert params(build_nb) == NB_PARAMS
    features = build_nb(UCI_FRAME).named_steps["features"]
    steps = {name: step for name, step, _ in features.transformers}
    bins, coded = steps["measured"], steps["coded"]
    assert bins.get_params() == BINS_PARAMS
    assert coded.categories == [sorted(UCI_FRAME.codebook[name]) for name in UCI_FRAME.categorical]


PANEL_PINS = {
    Model.RF: ([0.458, 0.4, 0.682, 0.49], 1e-12),
    Model.ET: ([0.464, 0.45, 0.76, 0.678], 1e-12),
    Model.MLP: (
        [0.48444663084702183, 0.41153020555138764, 0.46569538881794315, 0.46459778073016217],
        1e-9,
    ),
    Model.NB: (
        [0.9142451751415901, 0.24014533082267764, 0.6787750870458605, 0.2656454350095559],
        1e-12,
    ),
}
"""Each panel model's P(disease) of `GRADED` records 3, 6, 21 and 24, which seed 0 holds out, and
its relative tolerance. `_panel` fits under `threadpool_limits(limits=1)`, which bounds every pool
threadpoolctl detects, so the perceptron's matrix products sum in one order; the tolerance of 1e-9
leaves room for another BLAS build's rounding."""


@pytest.mark.parametrize("model", NEW, ids=str)
def test_panel_probabilities_are_pinned(model):
    expected, rtol = PANEL_PINS[model]
    probability = _panel(model, TRAIN.assign(weight=1.0))
    assert not GRADED.index[[3, 6, 21, 24]].isin(TRAIN.index).any()
    np.testing.assert_allclose(probability.to_numpy()[[3, 6, 21, 24]], expected, rtol=rtol)


def test_the_mlp_trains_exactly_its_epochs_and_records_not_raises():
    fits: list[tuple[int, bool]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        probability = _panel(
            Model.MLP, TRAIN.assign(weight=1.0), on_fit=lambda *fit: fits.append(fit)
        )
    assert probability.between(0, 1).all()
    assert fits == [(MLP_EPOCHS, False)]
    assert MLP_EPOCHS == 15


@pytest.mark.parametrize("tol", [0.0, 1e9], ids=["pinned", "no-epoch-improves"])
def test_no_loss_curve_can_stop_the_mlp_early(tol):
    """At a `tol` no loss beats, every epoch counts as not improving; it still trains 15, since
    `n_iter_no_change` is the epoch count (at scikit-learn's default of 10 it stops at 12)."""
    pipeline = build_mlp(UCI_FRAME).set_params(model__tol=tol)
    features = list(UCI_FRAME.features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(TRAIN.loc[:, features], TRAIN[UCI_FRAME.label])
    assert pipeline.named_steps["model"].n_iter_ == MLP_EPOCHS


class Fixed(ClassifierMixin, BaseEstimator):
    """A stand-in fixed-epoch model: it warns `warning` while it fits, then raises if `fail`."""

    def __init__(self, warning=None, fail=False):
        self.warning = warning
        self.fail = fail

    def fit(self, features, labels, sample_weight=None):
        if self.warning is not None:
            warnings.warn("stand-in", self.warning, stacklevel=2)
        if self.fail:
            msg = "the fit failed"
            raise RuntimeError(msg)
        self.classes_ = np.unique(labels)
        self.n_iter_ = 7
        return self

    def predict_proba(self, features):
        return np.full((len(features), 2), 0.5)


def _fixed(**kwargs):
    def factory(frame):
        return build_mlp(frame).set_params(model=Fixed(**kwargs))

    return factory


def test_a_fixed_epoch_fit_that_converges_reports_it():
    fits: list[tuple[int, bool]] = []
    probabilities(
        TRAIN.assign(weight=1.0),
        GRADED,
        factory=_fixed(),
        frame=UCI_FRAME,
        model=Model.MLP,
        on_fit=lambda *fit: fits.append(fit),
    )
    assert fits == [(7, True)]


def test_a_fixed_epoch_fits_other_warning_is_reissued():
    with pytest.warns(UserWarning, match="stand-in"):
        probabilities(
            TRAIN.assign(weight=1.0),
            GRADED,
            factory=_fixed(warning=UserWarning),
            frame=UCI_FRAME,
            model=Model.MLP,
        )


def test_a_failing_fixed_epoch_fit_still_reissues_its_warnings():
    with pytest.warns(UserWarning, match="stand-in"), pytest.raises(RuntimeError, match="failed"):
        probabilities(
            TRAIN.assign(weight=1.0),
            GRADED,
            factory=_fixed(warning=UserWarning, fail=True),
            frame=UCI_FRAME,
            model=Model.MLP,
        )


def _stingy(frame):
    pipeline = build(frame)
    pipeline.set_params(model__max_iter=1)
    return pipeline


@pytest.mark.parametrize("model", [Model.LR, Model.HGB], ids=str)
def test_a_default_models_non_convergence_keeps_the_default_message(model):
    with pytest.raises(EvaluationError) as raised:
        probabilities(
            TRAIN.assign(weight=1.0), TRAIN, factory=_stingy, frame=UCI_FRAME, model=model
        )
    assert str(raised.value) == "the classifier did not converge in 1000 iterations"


def test_a_panel_models_non_convergence_names_the_model_holdout_and_arm():
    with pytest.raises(EvaluationError) as raised:
        _fitted(
            3,
            "smote",
            TRAIN.assign(weight=1.0),
            TRAIN,
            factory=_stingy,
            frame=UCI_FRAME,
            model=Model.RF,
        )
    assert str(raised.value) == (
        "holdout 3's smote arm: the rf classifier did not converge in 1 iterations"
    )


def test_the_harness_logs_a_fixed_epoch_fit_with_its_holdout_and_arm(caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate.run")
    _fitted(
        2,
        "none",
        TRAIN.assign(weight=1.0),
        GRADED,
        factory=_fixed(),
        frame=UCI_FRAME,
        model=Model.MLP,
    )
    assert caplog.messages == ["fit holdout=2 arm=none model=mlp n_iter=7 converged=true"]


def test_a_test_category_no_training_row_holds_still_scores():
    training = TRAIN[TRAIN["sex"] == 0].assign(weight=1.0)
    test = GRADED[GRADED["sex"] == 1]
    assert len(training)
    assert len(test)
    probability = _panel(Model.NB, training, test)
    assert probability.between(0, 1).all()
    assert probability.notna().all()


def test_a_constant_column_bins_and_scores():
    constant = TRAIN.assign(weight=1.0, chol=240)
    probability = _panel(Model.NB, constant)
    assert probability.between(0, 1).all()
    assert probability.notna().all()
