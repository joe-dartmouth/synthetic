"""E1's configuration and the requests it gives every arm."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Final

import pytest

from bko.baselines.request import Method, Strata, plan
from bko.datasets.split import split
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import (
    ADDS_ROWS,
    BKO_ARMS,
    E1,
    METHOD_ARMS,
    RUNS,
    Experiment,
    NamedRun,
    flagged,
    gap_request,
    replica_request,
)
from bko.gaps.definitions import Definition
from bko.generate.messages import LEVELS, SETTINGS, Level, Settings
from bko.generate.spend import price_of
from tests._baselines_fixtures import full_records
from tests._evaluate_fixtures import FLAGGED, GRADED


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_e1_judges_bko_against_every_method():
    assert E1.arms == (*METHOD_ARMS, *BKO_ARMS)
    assert E1.comparators == METHOD_ARMS
    assert E1.candidate == "bko"
    assert E1.budget == 3
    assert E1.validity == Fraction(19, 20)
    assert dict(BKO_ARMS) == {
        "bko": Definition.SUBGROUP,
        "bko-population": Definition.POPULATION,
        "bko-frequency": Definition.FREQUENCY,
    }
    assert E1.replica_arms == (*(m.value for m in Method if m.adds_rows), *BKO_ARMS)
    assert len(set(E1.holdout_seeds)) == len(E1.holdout_seeds) >= 2


def test_e1_is_its_stated_configuration():
    methods = ("none", "class-weight", "smote", "adasyn", "ctgan", "tvae")
    stated = {
        "holdout_seeds": tuple(range(100)),
        "arms": (*methods, "bko", "bko-population", "bko-frequency"),
        "candidate": "bko",
        "comparators": methods,
        "replica_arms": (
            *("smote", "adasyn", "ctgan", "tvae"),
            *("bko", "bko-population", "bko-frequency"),
        ),
        "fill_to": 20,
        "margin": 0.0,
        "privacy_margin": 0.9,
        "confidence": 0.95,
        "attempts": 5,
        "budget": 3,
        "validity": Fraction(19, 20),
        "spend_ceiling": 25.0,
        "criteria": ("primary", "secondary", "privacy", "validity"),
        "projection_after": 5,
        "idle_limit": 900.0,
    }
    assert set(stated) == {field.name for field in dataclasses.fields(Experiment)}
    assert Experiment(**stated) == E1
    assert methods == METHOD_ARMS
    assert ADDS_ROWS == ("smote", "adasyn", "ctgan", "tvae")
    assert RUNS["e1"] == NamedRun(E1)


def test_one_guidance_run_per_level():
    assert set(RUNS) == {
        "e1",
        "g2",
        "g3-cell",
        "g3-coherent",
        "g3-findings",
        "g3-ontology",
        "g3-profile",
        "g3-summary",
        "g4-low",
        "g4-medium",
    }


@pytest.mark.parametrize("level", LEVELS)
def test_a_guidance_run_differs_from_e1_only_in_its_arms_ceiling_criteria_and_level(level):
    run = RUNS[f"g3-{level}"]
    assert run.experiment.arms == (*METHOD_ARMS, "bko")
    assert run.experiment.replica_arms == (*ADDS_ROWS, "bko")
    assert run.experiment.spend_ceiling == pytest.approx(10.0)
    assert run.experiment.criteria == ()
    assert run.settings.guidance is level
    same = dataclasses.replace(
        run.experiment,
        arms=E1.arms,
        replica_arms=E1.replica_arms,
        spend_ceiling=E1.spend_ceiling,
        criteria=E1.criteria,
    )
    assert NamedRun(same, dataclasses.replace(run.settings, guidance=SETTINGS.guidance)) == (
        NamedRun(E1)
    )


@pytest.mark.parametrize("effort", ["low", "medium"])
def test_a_sweep_run_is_g3_summarys_experiment_with_literal_settings(effort):
    run = RUNS[f"g4-{effort}"]
    assert run.experiment == RUNS["g3-summary"].experiment
    assert run.settings == Settings("claude-haiku-5-5", effort, 16000, 20, Level.SUMMARY)


def test_an_ablation_arm_may_be_judged_as_a_dry_run():
    assert dataclasses.replace(E1, candidate="bko-population")


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
        ({"replica_arms": ("absent",)}, "not arms"),
        ({"candidate": "smote"}, "its comparators"),
        ({"attempts": 0}, "at least one attempt"),
        ({"confidence": 1.0}, "strictly between"),
        ({"candidate": "absent"}, "is not an arm"),
        ({"budget": 0}, "budget"),
        ({"validity": Fraction(0)}, "validity"),
        ({"validity": Fraction(21, 20)}, "validity"),
        ({"spend_ceiling": 0.0}, "ceiling"),
        ({"replica_arms": ("none",)}, "add no rows"),
        ({"replica_arms": ("smote", "none", "class-weight")}, r"\['class-weight', 'none'\] add no"),
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
        "candidate not an arm",
        "no budget",
        "no validity",
        "validity above one",
        "no ceiling",
        "control replica",
        "both control replicas",
    ],
)
def test_a_configuration_that_cannot_run_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(E1, **change)


E1_1_TOKENS: Final = (1_925_406, 564_976)
"""`e1-1`'s input and output tokens over its 6000 records asked (its Runs row in EVIDENCE)."""

GAP_ROWS: Final = 20
"""Each BKO arm's gap rows per holdout."""

REPLICA_ROWS: Final = 238
"""E1's replica quota per holdout on the pinned archive (E1's Configuration); the gate reads no
data."""


def test_the_spend_ceiling_covers_the_recorded_estimate_with_its_margin():
    price = price_of(SETTINGS.model)
    tokens_in, tokens_out = E1_1_TOKENS
    per_record = (tokens_in * price.input + tokens_out * price.output) / 6000 / 1_000_000
    estimate = len(E1.holdout_seeds) * len(BKO_ARMS) * (GAP_ROWS + REPLICA_ROWS) * per_record
    assert estimate == pytest.approx(6.128, abs=0.001)
    assert E1.spend_ceiling == pytest.approx(25.0)
    assert E1.spend_ceiling >= 1.25 * estimate
