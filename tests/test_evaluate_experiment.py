"""E1's configuration and the requests it gives every arm."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from typing import Final

import pytest

from bko.baselines.request import Method, Strata, plan
from bko.datasets.cells import UNDERREPRESENTED_BELOW
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.cllm import SHOW
from bko.evaluate.data import Dataset
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import (
    ADDS_ROWS,
    BKO_ARMS,
    CLLM_ARM,
    CLLM_BUDGET,
    CLLM_RAW_ARM,
    E1,
    MESSAGES_ARMS,
    METHOD_ARMS,
    REAL_ARM,
    SKELETON_ARM,
    Curation,
    Experiment,
    MessagesArm,
    Removal,
    flagged,
    gap_request,
    replica_request,
)
from bko.evaluate.named_runs import (
    E3,
    E3_SLICE_CEILING,
    E4_TARGETS,
    E5_FRACTIONS,
    G2,
    G5,
    G6_MODELS,
    RUNS,
    SETTINGS_SWEEP,
    NamedRun,
)
from bko.evaluate.run import ARMS, run
from bko.gaps.definitions import Definition
from bko.generate.messages import LEVELS, SETTINGS, Level, Settings
from bko.generate.spend import price_of
from bko.generate.weights import LOCAL_MODELS
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
    assert E1.replica_arms == ("smote", "adasyn", "ctgan", "tvae", *BKO_ARMS)
    assert len(set(E1.holdout_seeds)) == len(E1.holdout_seeds) >= 2


def test_e3_is_its_stated_configuration():
    methods = ("none", "class-weight", "smote", "adasyn", "ctgan", "tvae")
    assert E3.arms == (*methods, "bko", "bko-population", "bko-frequency", "real")
    assert E3.comparators == methods
    assert E3.replica_arms == ("smote", "adasyn", "ctgan", "tvae", *BKO_ARMS)
    assert E3.removal == Removal((">=65", "female", 1), 0.9)
    assert E3.decontaminate is False
    assert E3.spend_ceiling == pytest.approx(65.0)
    assert E3.criteria == ()


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
        "threshold": 20,
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
        "removal": None,
        "decontaminate": True,
        "training_fraction": Fraction(1),
    }
    assert set(stated) == {field.name for field in dataclasses.fields(Experiment)}
    assert Experiment(**stated) == E1
    assert methods == METHOD_ARMS
    assert ADDS_ROWS == ("smote", "adasyn", "ctgan", "tvae")
    assert RUNS["e1"] == NamedRun(E1)


def test_every_run_name_is_pinned():
    assert set(RUNS) == {
        "b1-synthea",
        "b1-uci",
        "b2-synthea",
        "b2-uci",
        "b3-synthea",
        "b3-uci",
        "e1",
        "e3",
        "e3-a",
        "e3-b",
        "e4-eighty",
        "e4-forty",
        "e4-thirty",
        "e5-half",
        "e5-quarter",
        "e5-threequarters",
        "g2",
        "g3-cell",
        "g3-coherent",
        "g3-findings",
        "g3-ontology",
        "g3-profile",
        "g3-summary",
        "g4-low",
        "g4-medium",
        "g5",
        "g6-gemma",
        "g6-nemotron",
        "g6-olmo",
        "g6-sonnet",
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
    assert run.experiment.spend_ceiling == pytest.approx(10.0)
    assert run.experiment.criteria == ()
    assert run.settings == Settings(
        model="claude-haiku-5-5",
        effort=effort,
        max_tokens=16000,
        per_call=20,
        guidance=Level.SUMMARY,
    )


def test_an_ablation_arm_may_be_judged_as_a_dry_run():
    assert dataclasses.replace(E1, candidate="bko-population")


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_gap_request_fills_each_flagged_cell_to_the_threshold(seed):
    train = split(GRADED, seed=seed, edges=UCI_FRAME.edges).train
    request = gap_request(train, E1, seed=seed, frame=UCI_FRAME)
    assert dict(request.quota) == {FLAGGED[0]: 18, FLAGGED[1]: 2}
    assert request.strata is Strata.CELL
    assert request.seed == seed
    assert flagged(train, E1, edges=UCI_FRAME.edges) == FLAGGED


def test_the_gap_request_is_what_the_baselines_command_asks_for():
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    assert dict(gap_request(train, E1, seed=0, frame=UCI_FRAME).quota) == dict(
        plan(train, 20).quota
    )


def test_a_cell_at_the_threshold_is_not_filled():
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    lower = dataclasses.replace(E1, threshold=18, fill_to=18)
    assert dict(gap_request(train, lower, seed=0, frame=UCI_FRAME).quota) == {FLAGGED[0]: 16}


def test_a_target_above_the_threshold_flags_by_the_threshold_and_fills_to_the_target():
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    dosed = dataclasses.replace(E1, fill_to=30)
    assert dict(gap_request(train, dosed, seed=0, frame=UCI_FRAME).quota) == {
        FLAGGED[0]: 28,
        FLAGGED[1]: 12,
    }
    edges = UCI_FRAME.edges
    assert flagged(train, dosed, edges=edges) == flagged(train, E1, edges=edges)


def test_e1_requests_are_unchanged():
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    assert (E1.threshold, E1.fill_to) == (UNDERREPRESENTED_BELOW, UNDERREPRESENTED_BELOW)
    assert dict(gap_request(train, E1, seed=0, frame=UCI_FRAME).quota) == {
        FLAGGED[0]: 18,
        FLAGGED[1]: 2,
    }


def test_a_target_below_the_threshold_is_refused():
    with pytest.raises(ValueError, match="fill target 19 is below the threshold 20"):
        dataclasses.replace(E1, fill_to=19)


def test_the_attempt_limit_scales_with_the_target():
    """An attempt may ask `budget` times its request's rows (`bko.evaluate.guided`)."""
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    rows = {
        target: gap_request(
            train, dataclasses.replace(E1, fill_to=target), seed=0, frame=UCI_FRAME
        ).rows
        for target in (20, 30)
    }
    assert rows == {20: 20, 30: 40}


def test_nothing_to_fill_is_refused():
    train = split(full_records(26), seed=0, edges=UCI_FRAME.edges).train
    with pytest.raises(EvaluationError, match="nothing to fill"):
        gap_request(train, E1, seed=0, frame=UCI_FRAME)


def test_the_replica_asks_for_every_cells_training_count():
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    request = replica_request(train, seed=3, frame=UCI_FRAME)
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


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"removal": None}, "needs a removal"),
        ({"replica_arms": (*E3.replica_arms, REAL_ARM)}, "has no replica"),
        ({"decontaminate": True}, "keeps copies"),
    ],
    ids=["no removal", "replica", "decontaminating"],
)
def test_a_real_arm_the_harness_cannot_score_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(E3, **change)


def test_a_removal_of_a_cell_outside_the_frame_is_refused_before_any_holdout():
    experiment = dataclasses.replace(E3, removal=Removal((">=65", "female", 7), 0.5))
    with pytest.raises(EvaluationError, match="not a cell of"):
        run(full_records(), experiment, {}, frame=UCI_FRAME)


def test_e3s_slices_partition_its_seeds_and_its_ceiling():
    slices = [RUNS["e3-a"].experiment, RUNS["e3-b"].experiment]
    assert (*slices[0].holdout_seeds, *slices[1].holdout_seeds) == E3.holdout_seeds
    assert sum(s.spend_ceiling for s in slices) == E3.spend_ceiling
    assert E3.spend_ceiling / 2 == E3_SLICE_CEILING
    for experiment in slices:
        assert (
            dataclasses.replace(experiment, holdout_seeds=E3.holdout_seeds, spend_ceiling=65.0)
            == E3
        )


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


@pytest.mark.parametrize("name", sorted(set(Method) - set(METHOD_ARMS)))
def test_a_method_outside_the_comparison_set_joins_the_harness_alone(name):
    assert name in ARMS
    for experiment in (E1, G2, G5, SETTINGS_SWEEP, E3):
        assert name not in experiment.arms


def test_b1_runs_are_their_stated_configurations():
    new = ("gaussian-copula", "cart")
    uci, synthea = RUNS["b1-uci"], RUNS["b1-synthea"]
    assert uci.experiment == dataclasses.replace(
        E1,
        arms=(*METHOD_ARMS, *new),
        candidate="gaussian-copula",
        comparators=METHOD_ARMS,
        replica_arms=(*ADDS_ROWS, *new),
        criteria=(),
    )
    assert (uci.settings, uci.dataset) == (SETTINGS, Dataset.UCI)
    assert synthea.experiment == dataclasses.replace(
        E3,
        arms=(*METHOD_ARMS, *new, "real"),
        candidate="gaussian-copula",
        comparators=METHOD_ARMS,
        replica_arms=(*ADDS_ROWS, *new),
    )
    assert (synthea.settings, synthea.dataset) == (SETTINGS, Dataset.SYNTHEA)


@pytest.mark.parametrize(
    ("name", "model", "ceiling"),
    [
        ("g6-sonnet", "claude-sonnet-5-5", 90.0),
        ("g6-gemma", "gemma-4-26b-a4b-q8_0", 1.0),
        ("g6-nemotron", "nemotron-3-nano-30b-a3b-q8_0", 1.0),
        ("g6-olmo", "olmo-3-7b-instruct-q8_0", 1.0),
    ],
)
def test_each_g6_run_differs_from_e1_settings_in_model_alone(name, model, ceiling):
    run = RUNS[name]
    assert run.settings == dataclasses.replace(SETTINGS, model=model)
    assert run.dataset is Dataset.UCI
    assert run.experiment == dataclasses.replace(
        E1,
        arms=(*METHOD_ARMS, "bko"),
        replica_arms=(*ADDS_ROWS, "bko"),
        criteria=(),
        spend_ceiling=ceiling,
    )


def test_the_sonnet_ceiling_covers_its_estimate():
    """e1-4's tokens per record asked (300.535 input, 211.162 output) over its 25829 records
    asked, at Sonnet 5.5's rates: about $70, and the ceiling is at least 1.25 times that."""
    price = price_of("claude-sonnet-5-5")
    estimate = 25829 * (300.535 * price.input + 211.162 * price.output) / 1e6
    ceiling = RUNS["g6-sonnet"].experiment.spend_ceiling
    assert 1.25 * estimate <= ceiling <= 1.5 * estimate


def test_each_g6_model_is_priced_and_routed_by_where_it_runs():
    assert {name for name in RUNS if name.startswith("g6-")} == set(G6_MODELS)
    for name, (model, _) in G6_MODELS.items():
        assert RUNS[name].settings.model == model
        price_of(model)
        assert (model in LOCAL_MODELS) == (name != "g6-sonnet"), name


@pytest.mark.parametrize(
    "spec",
    [
        {"over_ask": 0},
        {"over_ask": 2},
        {"over_ask": 2, "curation": Curation.CLASSIFIER},
        {"curation": Curation.LIKELIHOOD},
        {"budget": 0},
    ],
    ids=[
        "no-ask",
        "over-ask-uncurated",
        "over-ask-classifier",
        "ranked-without-over-ask",
        "budget",
    ],
)
def test_an_inconsistent_messages_arm_is_refused(spec):
    with pytest.raises(ValueError, match=r"over_ask|budget|not both"):
        MessagesArm(Definition.SUBGROUP, **spec)


def test_a_ranked_curation_with_its_over_ask_is_a_messages_arm():
    arm = MessagesArm(None, shows=True, curation=Curation.LIKELIHOOD, over_ask=3, budget=6)
    assert (arm.definition, arm.over_ask, arm.budget) == (None, 3, 6)


def test_every_bko_arm_reads_the_experiments_budget_and_only_the_skeleton_arm_prescribes():
    assert {name: spec.definition for name, spec in MESSAGES_ARMS.items() if name in BKO_ARMS} == (
        dict(BKO_ARMS)
    )
    assert all(
        spec.budget is None and not spec.shows
        for name, spec in MESSAGES_ARMS.items()
        if name not in {CLLM_ARM, CLLM_RAW_ARM}
    )
    assert {name for name, spec in MESSAGES_ARMS.items() if spec.prescribes} == {SKELETON_ARM}


def test_the_cllm_arms_show_records_read_no_gap_and_differ_in_curation_alone():
    cllm, raw = MESSAGES_ARMS[CLLM_ARM], MESSAGES_ARMS[CLLM_RAW_ARM]
    assert (cllm.definition, cllm.shows, cllm.curation) == (None, True, Curation.CLASSIFIER)
    assert dataclasses.replace(cllm, curation=None) == raw
    assert not cllm.prescribes


def test_e4_runs_differ_from_e1_in_arms_target_ceiling_and_criteria_alone():
    assert dict(E4_TARGETS) == {"e4-thirty": 30, "e4-forty": 40, "e4-eighty": 80}
    for name, target in E4_TARGETS.items():
        named = RUNS[name]
        assert named.settings == SETTINGS
        assert named.dataset is Dataset.UCI
        assert named.experiment == dataclasses.replace(
            E1,
            arms=(*METHOD_ARMS, "bko"),
            replica_arms=(*ADDS_ROWS, "bko"),
            fill_to=target,
            spend_ceiling=10.0,
            criteria=(),
        )
        assert named.experiment.threshold == E1.threshold


def test_e5_runs_differ_from_e1_in_arms_fraction_ceiling_and_criteria_alone():
    assert dict(E5_FRACTIONS) == {
        "e5-quarter": Fraction(1, 4),
        "e5-half": Fraction(1, 2),
        "e5-threequarters": Fraction(3, 4),
    }
    for name, fraction in E5_FRACTIONS.items():
        named = RUNS[name]
        assert named.settings == SETTINGS
        assert named.dataset is Dataset.UCI
        assert named.experiment == dataclasses.replace(
            E1,
            arms=(*METHOD_ARMS, "bko"),
            replica_arms=(*ADDS_ROWS, "bko"),
            training_fraction=fraction,
            spend_ceiling=10.0,
            criteria=(),
        )


def test_b2_runs_hold_the_network_beside_each_frames_methods():
    uci, synthea = RUNS["b2-uci"], RUNS["b2-synthea"]
    assert uci.experiment == dataclasses.replace(
        E1,
        arms=(*METHOD_ARMS, "bayes-net"),
        candidate="bayes-net",
        comparators=METHOD_ARMS,
        replica_arms=(*ADDS_ROWS, "bayes-net"),
        criteria=(),
    )
    assert uci.dataset is Dataset.UCI
    assert synthea.dataset is Dataset.SYNTHEA
    assert synthea.experiment.arms == (*METHOD_ARMS, "bayes-net", REAL_ARM)
    assert synthea.experiment.removal == E3.removal
    assert synthea.experiment.criteria == E3.criteria
    assert not any(arm in MESSAGES_ARMS for arm in (*uci.experiment.arms, *synthea.experiment.arms))


def test_b3_runs_hold_bko_and_both_cllm_arms():
    for name, base in (("b3-uci", E1), ("b3-synthea", E3)):
        run = RUNS[name]
        experiment = run.experiment
        assert experiment.arms[: len(METHOD_ARMS)] == METHOD_ARMS
        assert {"bko", CLLM_ARM, CLLM_RAW_ARM} <= set(experiment.arms)
        assert {"bko", CLLM_ARM, CLLM_RAW_ARM} <= set(experiment.replica_arms)
        assert (experiment.candidate, experiment.comparators, experiment.criteria) == (
            CLLM_ARM,
            METHOD_ARMS,
            (),
        )
        assert (experiment.removal, experiment.decontaminate) == (base.removal, base.decontaminate)
        assert run.settings == SETTINGS
    assert REAL_ARM in RUNS["b3-synthea"].experiment.arms


@pytest.mark.parametrize(
    ("name", "per_arm", "calls", "tokens", "asks"),
    [
        ("b3-uci", 10.51 / 3, 5490 / 3, 60, 1.1401),
        ("b3-synthea", 20.46 / 3, 5490 / 3 * (20.46 / 10.51), 100, 1.4432),
    ],
    ids=["uci", "synthea"],
)
def test_each_b3_ceiling_covers_its_estimate(name, per_arm, calls, tokens, asks):
    """`docs/EVIDENCE.md` (B3) states the estimate: per arm, `e1-4`'s and `e3-4`'s Messages spend,
    `asks` the pilot's records asked per requested row for the curated arm, and 20 shown records of
    `tokens` input tokens each per call at the model's input rate."""
    arms = per_arm * (1 + asks + 1)
    shown = calls * (asks + 1) * SHOW * tokens * price_of(SETTINGS.model).input / 1e6
    assert 1.25 * (arms + shown) <= RUNS[name].experiment.spend_ceiling


def test_the_cllm_budget_covers_the_least_pilot_keep_share():
    assert max(3, math.ceil(1.5 / 0.1734)) == CLLM_BUDGET
