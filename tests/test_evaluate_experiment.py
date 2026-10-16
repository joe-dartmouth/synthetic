"""E1's configuration and the requests it gives every arm."""

from __future__ import annotations

import dataclasses

import pytest

from bko.baselines.request import Method, Strata, plan
from bko.datasets.split import split
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1, METHOD_ARMS, flagged, gap_request, replica_request
from tests._baselines_fixtures import full_records
from tests._evaluate_fixtures import FLAGGED, GRADED


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_e1_judges_bko_against_every_method():
    assert E1.candidate == "bko"
    assert E1.comparators == E1.arms == METHOD_ARMS
    assert set(E1.replica_arms) == {method.value for method in Method if method.adds_rows}
    assert len(set(E1.holdout_seeds)) == len(E1.holdout_seeds) >= 2


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_gap_request_fills_each_flagged_cell_to_the_threshold(seed):
    train = split(GRADED, seed=seed).train
    request = gap_request(train, E1, seed=seed)
    assert dict(request.quota) == {FLAGGED[0]: 18, FLAGGED[1]: 2}
    assert request.strata is Strata.CELL
    assert request.seed == seed
    assert flagged(train, E1) == FLAGGED


def test_the_gap_request_is_what_the_baselines_command_asks_for():
    train = split(GRADED, seed=0).train
    assert dict(gap_request(train, E1, seed=0).quota) == dict(plan(train, 20).quota)


def test_a_cell_at_the_threshold_is_not_filled():
    train = split(GRADED, seed=0).train
    lower = dataclasses.replace(E1, fill_to=18)
    assert dict(gap_request(train, lower, seed=0).quota) == {FLAGGED[0]: 16}


def test_nothing_to_fill_is_refused():
    train = split(full_records(26), seed=0).train
    with pytest.raises(EvaluationError, match="nothing to fill"):
        gap_request(train, E1, seed=0)


def test_the_replica_asks_for_every_cells_training_count():
    train = split(GRADED, seed=0).train
    request = replica_request(train, seed=3)
    assert request.rows == len(train)
    assert request.seed == 3
    assert request.quota[FLAGGED[0]] == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"holdout_seeds": (0,)}, "2 or more"),
        ({"holdout_seeds": (0, 0)}, "2 or more"),
        ({"holdout_seeds": (0, -1)}, "must lie in"),
        ({"arms": ("none", "none")}, "twice"),
        ({"replica_arms": ("bko",)}, "not arms"),
        ({"candidate": "smote"}, "its comparators"),
        ({"attempts": 0}, "at least one attempt"),
        ({"confidence": 1.0}, "strictly between"),
    ],
    ids=[
        "one seed",
        "repeated seed",
        "negative seed",
        "repeated arm",
        "stray arm",
        "self",
        "no attempt",
        "certain",
    ],
)
def test_a_configuration_that_cannot_run_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(E1, **change)
