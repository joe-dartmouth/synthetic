"""`Measure.AUC_ROC`: ROC AUC per scope, written only for a run that flags a cell by name."""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest

from bko.evaluate.classifier import Model
from bko.evaluate.experiment import Experiment
from bko.evaluate.run import ARMS, score_holdout
from bko.evaluate.scores import Measure, Scope, Source, suffixed
from bko.evaluate.utility import utility
from tests._frame_fixtures import TOY, TOY_PARTS


SIX = pd.DataFrame(
    {
        "age": [50, 50, 50, 50, 35, 35],
        "sex": [1, 1, 1, 1, 0, 0],
        "level": [1.0] * 6,
        "target": [0, 0, 1, 1, 0, 1],
    },
    index=pd.RangeIndex(1, 7, name="record_id"),
).astype(dict(TOY.dtypes))
"""Four test rows in the `45-69` male subgroup, two `<45` women."""

PROBABILITY = pd.Series([0.1, 0.4, 0.3, 0.8, 0.35, 0.9], index=SIX.index)
"""Over all six rows, 7 of the 9 positive-negative pairs rank the positive higher; over the four
male rows, 3 of 4."""

MALE = (("45-69", "male", 1),)

TOY_RUN = Experiment(
    holdout_seeds=(0, 1),
    arms=("none", "smote"),
    candidate="smote",
    comparators=("none",),
    replica_arms=(),
)

AUGMENTED = {Measure.RECALL, Measure.PRECISION, Measure.AUC_PR}
"""The measures every run writes for each model's augmented fit."""


def test_auc_roc_is_the_hand_value():
    scores = utility(SIX, PROBABILITY, MALE, edges=TOY.edges, roc=True)
    assert scores[Measure.AUC_ROC, Scope.ALL] == pytest.approx(7 / 9)
    assert scores[Measure.AUC_ROC, Scope.FLAGGED] == pytest.approx(3 / 4)
    keys = list(scores)
    assert keys.index((Measure.AUC_ROC, Scope.ALL)) == keys.index((Measure.AUC_PR, Scope.ALL)) + 1


def test_auc_roc_is_nan_on_one_class():
    one = SIX.assign(target=[0, 0, 0, 0, 0, 1])
    scores = utility(one, PROBABILITY, MALE, edges=TOY.edges, roc=True)
    assert math.isnan(scores[Measure.AUC_ROC, Scope.FLAGGED])
    assert scores[Measure.AUC_ROC, Scope.ALL] == pytest.approx(1.0)
    plain = utility(one, PROBABILITY, MALE, edges=TOY.edges)
    assert {measure for measure, _ in plain} == AUGMENTED


def _augmented(experiment: Experiment) -> list[tuple[str, Measure, Scope]]:
    arms = {name: ARMS[name] for name in experiment.arms}
    scores = score_holdout(TOY_PARTS, experiment, arms, frame=TOY)
    return [(s.arm, s.measure, s.scope) for s in scores if s.source is Source.AUGMENTED]


def test_a_run_without_a_flagged_cell_writes_no_auc_roc_row():
    found = _augmented(TOY_RUN)
    expected = {suffixed(m, model) for m in AUGMENTED for model in (Model.LR, Model.HGB)}
    assert {measure for _, measure, _ in found} == expected


def test_a_run_with_a_flagged_cell_writes_it_on_both_scopes():
    named = dataclasses.replace(TOY_RUN, fill_to=26, flagged_cell=("<45", "male", 1))
    found = _augmented(named)
    for arm in named.arms:
        for measure in (Measure.AUC_ROC, Measure.AUC_ROC_HGB):
            assert [scope for a, m, scope in found if (a, m) == (arm, measure)] == [
                Scope.ALL,
                Scope.FLAGGED,
            ]


@pytest.mark.parametrize("model", list(Model))
def test_every_model_has_an_auc_roc_measure(model):
    expected = "auc-roc" if model is Model.LR else f"auc-roc-{model.value}"
    assert suffixed(Measure.AUC_ROC, model).value == expected


def test_auc_roc_sits_after_auc_pr_and_before_the_calibration_measures():
    scores = utility(SIX, PROBABILITY, MALE, edges=TOY.edges, roc=True, calibration=True)
    for scope in Scope:
        assert [m for m, s in scores if s is scope] == [
            Measure.RECALL,
            Measure.PRECISION,
            Measure.AUC_PR,
            Measure.AUC_ROC,
            Measure.BRIER,
            Measure.CALIBRATION,
        ]
