"""E1's configuration and the requests it gives every arm."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from typing import Final

import pytest

from bko.baselines.request import Method, Strata, plan
from bko.datasets.cells import UNDERREPRESENTED_BELOW
from bko.datasets.split import kept_count, split
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
    E6_ASKED,
    E6_CAP,
    E6_EMPTY,
    E6_SETTINGS,
    E6_SLICES,
    E6_THIN,
    G2,
    G5,
    G6_BASE,
    G6_MODELS,
    G6_THINKING,
    RUNS,
    SETTINGS_SWEEP,
    NamedRun,
    _e6_runs,
)
from bko.evaluate.run import ARMS, run, score_holdout
from bko.evaluate.scores import Measure, Scope, Source
from bko.gaps.definitions import Definition
from bko.generate.local import TIMEOUT_SECONDS as LOCAL_TIMEOUT_SECONDS
from bko.generate.messages import LEVELS, SETTINGS, Level, Settings
from bko.generate.spend import price_of
from bko.generate.weights import LOCAL_MODELS
from tests._baselines_fixtures import full_records
from tests._evaluate_fixtures import FAST_ARMS, FLAGGED, GRADED, SMALL


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
        "cutoff": 0.5,
        "replica_rows": None,
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
        "e6-empty",
        *(f"e6-empty{letter}" for letter in "abcdefghij"),
        "e6-thin",
        *(f"e6-thin{letter}" for letter in "abcdefghij"),
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
        "g6-gemmathink",
        "g6-nemotron",
        "g6-nemotronthink",
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
    request = replica_request(train, E1, seed=3, frame=UCI_FRAME)
    assert request.rows == len(train)
    assert request.seed == 3
    assert request.quota[FLAGGED[0]] == 2


def test_a_capped_replica_asks_kept_count_per_cell():
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    cap = len(train) // 2
    capped = dataclasses.replace(E1, replica_rows=cap)
    request = replica_request(train, capped, seed=3, frame=UCI_FRAME)
    whole = replica_request(train, E1, seed=3, frame=UCI_FRAME)
    fraction = Fraction(cap, len(train))
    assert request.quota == {
        cell: kept_count(count, fraction) for cell, count in whole.quota.items()
    }
    assert request.rows < whole.rows
    assert request.seed == 3


def test_no_cap_keeps_the_whole_split_replica():
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    request = replica_request(train, E1, seed=3, frame=UCI_FRAME)
    assert request.rows == len(train)


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


E6_EVERY_ARM: Final = (
    "none",
    "class-weight",
    "smote",
    "adasyn",
    "ctgan",
    "tvae",
    "gaussian-copula",
    "cart",
    "bayes-net",
    "bko",
    "bko-population",
    "bko-frequency",
    "bko-direct",
    "bko-skeleton",
    "cllm",
    "cllm-uncurated",
    "real",
)
"""E6's arms as ADR `research/diabetes-subgroup-removal-design` (Decision, Arms) lists them."""


def test_e6s_runs_pin_the_diabetes_design():
    assert E6_THIN.arms == E6_EVERY_ARM
    assert E6_THIN.candidate == "bko"
    assert E6_THIN.comparators == tuple(a for a in E6_EVERY_ARM if a not in {"bko", REAL_ARM})
    assert E6_THIN.replica_arms == ("bko",)
    assert E6_THIN.replica_rows == 2000
    assert E6_THIN.removal == Removal(("6-7", "male", 1), 0.985)
    assert E6_THIN.fill_to == 1269
    assert E6_THIN.threshold == 20
    assert E6_THIN.cutoff == pytest.approx(0.0902)
    assert E6_THIN.decontaminate is False
    assert E6_THIN.criteria == ()
    assert E6_THIN.holdout_seeds == tuple(range(100))
    emptied = dataclasses.replace(
        E6_THIN,
        removal=Removal(("6-7", "male", 1), 1.0),
        replica_arms=(),
        replica_rows=None,
        spend_ceiling=E6_EMPTY.spend_ceiling,
    )
    assert emptied == E6_EMPTY
    assert dataclasses.replace(SETTINGS, max_tokens=32000) == E6_SETTINGS
    assert RUNS["e6-thin"] == NamedRun(E6_THIN, E6_SETTINGS, Dataset.DIABETES)
    assert RUNS["e6-empty"] == NamedRun(E6_EMPTY, E6_SETTINGS, Dataset.DIABETES)


@pytest.mark.parametrize(("name", "whole"), [("thin", E6_THIN), ("empty", E6_EMPTY)])
def test_e6s_slices_take_ten_consecutive_holdouts_at_a_tenth_of_the_ceiling(name, whole):
    assert E6_SLICES == 10
    slices = [RUNS[f"e6-{name}{letter}"] for letter in "abcdefghij"]
    for index, named in enumerate(slices):
        assert named.experiment.holdout_seeds == tuple(range(10 * index, 10 * index + 10))
        assert named.experiment.spend_ceiling == whole.spend_ceiling / 10
        assert named.experiment.projection_after < 10
        assert named.settings == E6_SETTINGS
        assert named.dataset is Dataset.DIABETES
        assert (
            dataclasses.replace(
                named.experiment,
                holdout_seeds=whole.holdout_seeds,
                spend_ceiling=whole.spend_ceiling,
            )
            == whole
        )
    seeds = tuple(seed for named in slices for seed in named.experiment.holdout_seeds)
    assert seeds == whole.holdout_seeds
    total = sum(named.experiment.spend_ceiling for named in slices)
    assert total == pytest.approx(whole.spend_ceiling)


@pytest.mark.parametrize("rows", [0, -5])
def test_a_replica_cap_below_one_row_is_refused(rows):
    with pytest.raises(ValueError, match=f"replica rows {rows} must be 1 or more"):
        dataclasses.replace(E6_THIN, replica_rows=rows)


def test_e6_slices_refuse_holdouts_they_do_not_divide():
    uneven = dataclasses.replace(E6_THIN, holdout_seeds=tuple(range(99)))
    with pytest.raises(ValueError, match="99 holdouts do not divide into 10 slices"):
        _e6_runs("thin", uneven)


def test_each_e6_ceiling_covers_its_estimate():
    """E6's estimate per record asked (`docs/EVIDENCE.md`, E6 Configuration), written out.

    Rate: the larger of `e1-4`'s $1.35635e-4 per record x 34 Diabetes columns / 14 UCI columns,
    $3.2940e-4, and pilot `e6-1`'s $3.1536e-4. Records asked per holdout: the five BKO-type arms
    and `cllm-uncurated` ask (fill_to - R) x (1 + 0.0075), the drop share pooled over the two
    10-call pilots (3 of 400 records); `cllm` asks (fill_to - R) x 7.837, `b3-3`'s CLLM records
    asked per requested row; `bko`'s replica asks 2000 x (1 + 0.0075), in the f run only.
    """
    rate = max(1.35635e-4 * 34 / 14, 3.1536e-4)
    assert rate == pytest.approx(3.294e-4, rel=1e-4)
    thin = 6 * (1269 - 19) * 1.0075 + (1269 - 19) * 7.837 + 2000 * 1.0075
    empty = 6 * (1269 - 0) * 1.0075 + (1269 - 0) * 7.837
    assert (math.floor(thin), math.floor(empty)) == (19_367, 17_616)
    assert E6_ASKED == {"thin": 19_367, "empty": 17_616}
    assert math.isclose(E6_CAP, 1500.0)
    # Ceilings: $1,500 x each fraction's share of 36,983 records asked per holdout.
    # thin: 1500 x 19,367 / 36,983 = $785.51; empty: 1500 x 17,616 / 36,983 = $714.49.
    assert E6_THIN.spend_ceiling == pytest.approx(785.510, abs=1e-3)
    assert E6_EMPTY.spend_ceiling == pytest.approx(714.490, abs=1e-3)
    assert E6_THIN.spend_ceiling + E6_EMPTY.spend_ceiling == pytest.approx(E6_CAP)
    # Estimates over 100 holdouts at $3.294e-4: thin $637.95, empty $580.27, total $1,218.22;
    # each ceiling is 1500 / 1218.22 = 1.2313 times its estimate, and each slice's tenth of its
    # ceiling covers its 10 holdouts' tenth of the estimate.
    for name, whole in (("thin", E6_THIN), ("empty", E6_EMPTY)):
        estimate = 100 * E6_ASKED[name] * 3.294e-4
        assert whole.spend_ceiling / estimate == pytest.approx(1.2313, abs=1e-4)
        for letter in "abcdefghij":
            sliced = RUNS[f"e6-{name}{letter}"].experiment
            assert sliced.spend_ceiling >= len(sliced.holdout_seeds) * E6_ASKED[name] * 3.294e-4


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


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("g6-gemmathink", "gemma-4-26b-a4b-q8_0-thinking"),
        ("g6-nemotronthink", "nemotron-3-nano-30b-a3b-q8_0-thinking"),
    ],
)
def test_each_thinking_run_is_g6_base_on_holdouts_0_to_4(name, model):
    run = RUNS[name]
    assert G6_THINKING[name] == model
    assert run.settings == dataclasses.replace(SETTINGS, model=model)
    assert run.dataset is Dataset.UCI
    assert run.experiment.holdout_seeds == tuple(range(5))
    assert run.experiment == dataclasses.replace(
        G6_BASE, holdout_seeds=tuple(range(5)), spend_ceiling=1.0, idle_limit=1700.0
    )
    assert run.experiment.idle_limit < LOCAL_TIMEOUT_SECONDS
    assert model in LOCAL_MODELS
    price_of(model)


def test_each_g6_model_is_priced_and_routed_by_where_it_runs():
    assert {name for name in RUNS if name.startswith("g6-")} == set(G6_MODELS) | set(G6_THINKING)
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
    `tokens` input tokens each per call at the model's input rate. B3 ran at a CLLM budget of 9,
    before `CLLM_BUDGET` moved to 22, so the estimate reads its pilot's asks, not the budget."""
    arms = per_arm * (1 + asks + 1)
    shown = calls * (asks + 1) * SHOW * tokens * price_of(SETTINGS.model).input / 1e6
    assert 1.25 * (arms + shown) <= RUNS[name].experiment.spend_ceiling


def test_the_cllm_budget_covers_the_least_observed_keep_share():
    """B3's `cllm` on Synthea kept 0.0707 of the replica `<65/female/1` rows reaching its curator
    (ADR `research/cllm-arm-design`, Consequences)."""
    assert max(3, math.ceil(1.5 / 0.0707)) == CLLM_BUDGET


@pytest.mark.parametrize("name", ["e1", "e3-a", "e3-b"])
def test_e1_and_e3_keep_0_5(name):
    assert RUNS[name].experiment.cutoff == pytest.approx(0.5)


def test_score_holdout_calls_positives_at_the_experiments_cutoff():
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    experiment = dataclasses.replace(
        SMALL, arms=("none",), candidate="none", comparators=(), replica_arms=()
    )

    def recall(experiment):
        scores = score_holdout(parts, experiment, FAST_ARMS, frame=UCI_FRAME)
        return next(
            s.value
            for s in scores
            if s.arm == "none" and s.measure is Measure.RECALL and s.scope is Scope.ALL
        )

    assert recall(dataclasses.replace(experiment, cutoff=0.01)) == pytest.approx(1.0)
    assert recall(dataclasses.replace(experiment, cutoff=0.99)) == pytest.approx(0.0)


def test_score_holdout_calls_a_replicas_positives_at_the_experiments_cutoff():
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    experiment = dataclasses.replace(
        SMALL, arms=("none", "smote"), candidate="none", comparators=(), replica_arms=("smote",)
    )

    def recall(experiment):
        scores = score_holdout(parts, experiment, FAST_ARMS, frame=UCI_FRAME)
        return next(
            s.value
            for s in scores
            if s.source is Source.REPLICA and s.measure is Measure.RECALL and s.scope is Scope.ALL
        )

    assert recall(dataclasses.replace(experiment, cutoff=0.01)) == pytest.approx(1.0)
    assert recall(dataclasses.replace(experiment, cutoff=0.99)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "cutoff", [0.0, 1.0, -0.5, 1.5, math.nan, math.inf, True, 1, "0.5"], ids=repr
)
def test_a_cutoff_outside_the_open_unit_interval_is_refused(cutoff):
    with pytest.raises(ValueError, match=f"cutoff {cutoff!r} must be a float strictly between"):
        dataclasses.replace(E1, cutoff=cutoff)
