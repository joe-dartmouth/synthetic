"""The classifier: codebook features, weights to the regression only, no silent non-convergence."""

from __future__ import annotations

import numpy as np
import pytest

from bko.baselines.methods import augment
from bko.baselines.records import training_set
from bko.baselines.request import Method, Request
from bko.datasets.split import split
from bko.datasets.uci import complete_records, parse_cleveland
from bko.evaluate.classifier import build, probabilities
from bko.evaluate.errors import EvaluationError
from tests._baselines_fixtures import UNBALANCED_TEXT
from tests._evaluate_fixtures import FAST_METHODS, FLAGGED, GRADED, Spy


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

UNBALANCED = split(complete_records(parse_cleveland(UNBALANCED_TEXT))).train
TRAIN = split(GRADED, seed=0).train


def test_class_weights_reach_the_regression_row_by_row():
    spy = Spy()
    baseline = augment(UNBALANCED, Method.CLASS_WEIGHT, Request())
    shuffled = training_set(UNBALANCED, baseline).sample(frac=1.0, random_state=0)
    probabilities(shuffled, UNBALANCED, factory=spy)
    labels, weights = spy.models[0].seen
    assert weights == pytest.approx(np.where(labels == 0, 36 / 24, 36 / 48))


def test_added_rows_follow_the_real_rows_at_weight_one():
    request = Request(quota={FLAGGED[0]: 3}, seed=0)
    baseline = augment(TRAIN, Method.SMOTE, request, methods=FAST_METHODS)
    spy = Spy()
    probabilities(training_set(TRAIN, baseline), TRAIN, factory=spy)
    labels, weights = spy.models[0].seen
    assert len(weights) == len(TRAIN) + 3
    assert weights[-3:].tolist() == [1.0] * 3
    assert labels[-3:].tolist() == [1] * 3


def test_every_code_has_a_column_whatever_the_training_rows_hold():
    narrow = TRAIN[TRAIN["restecg"] != 1].assign(weight=1.0)
    test = TRAIN[TRAIN["restecg"] == 1]
    assert not test.empty
    probability = probabilities(narrow, test)
    assert probability.between(0, 1).all()
    assert probability.index.equals(test.index)


def test_a_fit_that_does_not_converge_stops_the_run():
    def stingy(frame):
        pipeline = build(frame)
        pipeline.set_params(model__max_iter=1)
        return pipeline

    with pytest.raises(EvaluationError, match="did not converge"):
        probabilities(TRAIN.assign(weight=1.0), TRAIN, factory=stingy)


def test_probabilities_are_pinned():
    probability = probabilities(TRAIN.assign(weight=1.0), GRADED)
    np.testing.assert_allclose(
        probability.to_numpy()[[0, 7, 23, 41]],
        [0.2743870822873947, 0.3788422046931614, 0.3911697315854433, 0.38625833787281055],
        rtol=1e-12,
    )
