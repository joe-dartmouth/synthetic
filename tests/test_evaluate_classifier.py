"""The classifiers: codebook features, weights to the model only, no silent non-convergence, and a
boosting model that no seed or thread count moves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from threadpoolctl import threadpool_info, threadpool_limits

from bko.baselines.methods import augment
from bko.baselines.records import training_set
from bko.baselines.request import Method, Request
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME, complete_records, parse_cleveland
from bko.evaluate.classifier import MODELS, Model, build, build_hgb, probabilities
from bko.evaluate.errors import EvaluationError
from tests._baselines_fixtures import UNBALANCED_TEXT
from tests._evaluate_fixtures import FAST_METHODS, FLAGGED, GRADED, Spy


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

UNBALANCED = split(complete_records(parse_cleveland(UNBALANCED_TEXT)), edges=UCI_FRAME.edges).train
TRAIN = split(GRADED, seed=0, edges=UCI_FRAME.edges).train


def test_class_weights_reach_the_regression_row_by_row():
    spy = Spy()
    baseline = augment(UNBALANCED, Method.CLASS_WEIGHT, Request())
    shuffled = training_set(UNBALANCED, baseline).sample(frac=1.0, random_state=0)
    probabilities(shuffled, UNBALANCED, factory=spy, frame=UCI_FRAME)
    labels, weights = spy.models[0].seen
    assert weights == pytest.approx(np.where(labels == 0, 36 / 24, 36 / 48))


def test_added_rows_follow_the_real_rows_at_weight_one():
    request = Request(quota={FLAGGED[0]: 3}, seed=0)
    baseline = augment(TRAIN, Method.SMOTE, request, methods=FAST_METHODS)
    spy = Spy()
    probabilities(training_set(TRAIN, baseline), TRAIN, factory=spy, frame=UCI_FRAME)
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
