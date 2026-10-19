"""The harness on the graded frame: every arm on every holdout, and no test record for an arm."""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter

import numpy as np
import pytest

from bko.baselines.errors import BaselineError, OutputContractError
from bko.baselines.records import derived_seed
from bko.datasets.split import split
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1, gap_request
from bko.evaluate.run import held_out_ratio, run
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Source, render
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL, Spy


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_a_control_arm_ignores_the_request():
    train = split(GRADED, seed=0).train
    baseline = FAST_ARMS["none"](train, gap_request(train, E1, seed=0))
    assert baseline.added.empty


@pytest.fixture(scope="module")
def scores():
    return run(GRADED, SMALL, FAST_ARMS)


def test_every_arm_is_scored_on_every_holdout(scores):
    counts = Counter((score.holdout, score.arm, score.source) for score in scores)
    for seed in SMALL.holdout_seeds:
        assert counts[seed, HELD_OUT_ARM, Source.HELD_OUT] == 3
        for arm in SMALL.arms:
            assert counts[seed, arm, Source.AUGMENTED] == 6
        for arm in SMALL.replica_arms:
            assert counts[seed, arm, Source.REPLICA] == 9
            assert counts[seed, arm, Source.ADDED] == 3
        for arm in ("none", "class-weight"):
            assert counts[seed, arm, Source.ADDED] == 1


def test_every_row_adding_arm_adds_the_same_rows(scores):
    rows = {(s.holdout, s.arm): s.value for s in scores if s.measure is Measure.ROWS}
    assert {value for (_, arm), value in rows.items() if arm in SMALL.replica_arms} == {20.0}
    assert (rows[0, "none"], rows[0, "class-weight"]) == pytest.approx((0.0, 0.0))


def test_the_held_out_ratio_is_the_largest_cells():
    assert held_out_ratio(split(GRADED, seed=0)) == pytest.approx(5 / 18)


def test_class_weights_reach_the_fit_through_the_harness():
    spy = Spy()
    experiment = dataclasses.replace(
        SMALL,
        arms=("class-weight",),
        candidate="class-weight",
        comparators=(),
        replica_arms=(),
    )
    run(GRADED, experiment, FAST_ARMS, factory=spy)
    labels, weights = spy.models[0].seen
    total, positives = len(labels), int(labels.sum())
    expected = np.where(labels == 1, total / (2 * positives), total / (2 * (total - positives)))
    assert weights == pytest.approx(expected)


def test_no_arm_sees_a_test_record():
    seen = []

    def spying(train, request):
        seen.append(set(train.index))
        return FAST_ARMS["smote"](train, request)

    experiment = dataclasses.replace(
        SMALL, arms=("smote",), candidate="smote", comparators=(), replica_arms=("smote",)
    )
    run(GRADED, experiment, {"smote": spying})
    assert len(seen) == 2 * len(SMALL.holdout_seeds)
    for position, indexes in enumerate(seen):
        parts = split(GRADED, seed=SMALL.holdout_seeds[position // 2])
        assert indexes == set(parts.train.index)
        assert not indexes & set(parts.test.index)


def test_an_arm_without_a_runner_is_refused():
    with pytest.raises(EvaluationError, match="no runner"):
        run(GRADED, SMALL, {"none": FAST_ARMS["none"]})


def test_a_refusing_arm_stops_the_run_after_every_attempt():
    calls = []

    def refuse(train, request):
        calls.append(request.seed)
        msg = "refused"
        raise BaselineError(msg)

    with pytest.raises(BaselineError, match="refused"):
        run(GRADED, SMALL, {**FAST_ARMS, "smote": refuse})
    assert len(calls) == SMALL.attempts


def test_a_refusal_is_retried_with_a_derived_seed():
    seeds = []

    def flaky(train, request):
        seeds.append(request.seed)
        if request.seed in SMALL.holdout_seeds:
            msg = "refused"
            raise BaselineError(msg)
        return FAST_ARMS["smote"](train, request)

    experiment = dataclasses.replace(
        SMALL, arms=("smote",), candidate="smote", comparators=(), replica_arms=()
    )
    scores = run(GRADED, experiment, {"smote": flaky})
    expected = [s for seed in SMALL.holdout_seeds for s in (seed, derived_seed(seed, "attempt", 1))]
    assert seeds == expected
    assert {score.holdout for score in scores} == set(SMALL.holdout_seeds)


def test_a_contract_break_is_not_retried():
    calls = []

    def broken(train, request):
        calls.append(request.seed)
        msg = "broken"
        raise OutputContractError(msg)

    with pytest.raises(OutputContractError, match="broken"):
        run(GRADED, SMALL, {**FAST_ARMS, "smote": broken})
    assert len(calls) == 1


def test_a_rerun_writes_the_same_bytes():
    experiment = dataclasses.replace(
        SMALL,
        arms=("none", "smote", "adasyn", "ctgan", "tvae"),
        candidate="smote",
        comparators=(),
        replica_arms=("smote",),
    )
    first = render(run(GRADED, experiment, FAST_ARMS))
    np.random.seed(1)
    assert render(run(GRADED, experiment, FAST_ARMS)) == first


def test_a_runner_that_changes_its_input_changes_nothing_else():
    def vandal(train, request):
        baseline = FAST_ARMS["smote"](train, request)
        train.loc[:, "target"] = 0
        train.drop(train.index[:10], inplace=True)
        return baseline

    experiment = dataclasses.replace(
        SMALL, arms=("smote", "none"), candidate="smote", comparators=(), replica_arms=("smote",)
    )
    honest = render(run(GRADED, experiment, FAST_ARMS))
    assert render(run(GRADED, experiment, {**FAST_ARMS, "smote": vandal})) == honest


def test_every_retry_is_logged_with_its_own_derived_seed(caplog):
    seeds = []

    def twice_refused_replica(train, request):
        if request.rows == len(train):
            seeds.append(request.seed)
            if len(seeds) < 3:
                msg = "refused"
                raise BaselineError(msg)
        return FAST_ARMS["smote"](train, request)

    experiment = dataclasses.replace(
        SMALL, arms=("smote",), candidate="smote", comparators=(), replica_arms=("smote",)
    )
    with caplog.at_level(logging.INFO, logger="bko.evaluate.run"):
        run(GRADED, experiment, {"smote": twice_refused_replica})
    first = SMALL.holdout_seeds[0]
    derived = [derived_seed(first, "attempt", attempt) for attempt in (1, 2)]
    assert seeds[:3] == [first, *derived]
    assert f"retry holdout={first} arm=smote-replica attempt=2 reason=refused" in caplog.text
    assert f"arm holdout={first} arm=smote-replica attempt=3 seed={derived[1]}" in caplog.text
