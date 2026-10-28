"""Recall, precision and AUC-PR per scope, and NaN wherever a scope cannot define one."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import AGE_BAND_EDGES, LABEL
from bko.evaluate.scores import Measure, Scope
from bko.evaluate.utility import utility
from tests._baselines_fixtures import EDGE


EDGES = AGE_BAND_EDGES
"""The band edges the fixture's cells are named by."""


pytestmark = [
    pytest.mark.filterwarnings("error::FutureWarning"),
    pytest.mark.filterwarnings("error::UserWarning"),
]

FLAGGED = ((f">={EDGE}", "female", 1),)
"""Women 55 and over: 2 positives and 2 negatives below."""

PROBABILITY = [0.9, 0.4, 0.6, 0.1, 0.2, 0.3, 0.8, 0.7]


def records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [60, 60, 60, 60, 40, 40, 40, 40],
            "sex": [0, 0, 0, 0, 0, 0, 1, 1],
            LABEL: [1, 1, 0, 0, 0, 0, 1, 0],
        }
    )


def test_each_measure_in_each_scope_is_hand_computed():
    test = records()
    scores = utility(test, pd.Series(PROBABILITY, index=test.index), FLAGGED, edges=EDGES)
    assert scores[Measure.RECALL, Scope.ALL] == pytest.approx(2 / 3)
    assert scores[Measure.PRECISION, Scope.ALL] == pytest.approx(1 / 2)
    assert scores[Measure.AUC_PR, Scope.ALL] == pytest.approx(13 / 15)
    assert scores[Measure.RECALL, Scope.FLAGGED] == pytest.approx(1 / 2)
    assert scores[Measure.PRECISION, Scope.FLAGGED] == pytest.approx(1 / 2)
    assert scores[Measure.AUC_PR, Scope.FLAGGED] == pytest.approx(5 / 6)


def test_a_probability_of_one_half_is_called_positive():
    test = records()
    probability = pd.Series([0.5] + [0.0] * 7, index=test.index)
    scores = utility(test, probability, FLAGGED, edges=EDGES)
    assert scores[Measure.RECALL, Scope.FLAGGED] == pytest.approx(0.5)


def test_a_scope_without_positives_is_nan_not_zero():
    test = records()
    scores = utility(
        test, pd.Series(0.9, index=test.index), ((f"<{EDGE}", "female", 1),), edges=EDGES
    )
    assert math.isnan(scores[Measure.RECALL, Scope.FLAGGED])
    assert math.isnan(scores[Measure.AUC_PR, Scope.FLAGGED])
    assert scores[Measure.PRECISION, Scope.FLAGGED] == pytest.approx(0.0)


def test_precision_without_a_positive_call_is_nan():
    test = records()
    scores = utility(test, pd.Series(0.1, index=test.index), FLAGGED, edges=EDGES)
    assert math.isnan(scores[Measure.PRECISION, Scope.ALL])
    assert scores[Measure.RECALL, Scope.ALL] == pytest.approx(0.0)


def test_probabilities_are_matched_to_records_by_index():
    test = records()
    probability = pd.Series(PROBABILITY, index=test.index)
    shuffled = probability.iloc[np.array([7, 2, 5, 0, 3, 6, 1, 4])]
    assert utility(test, shuffled, FLAGGED, edges=EDGES) == pytest.approx(
        utility(test, probability, FLAGGED, edges=EDGES)
    )
