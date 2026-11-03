"""The harness names the holdout and the fill each runner call serves, for log attribution."""

from __future__ import annotations

import dataclasses

import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.fills import CURRENT_FILL, Fill
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.run import score_holdout
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

PAIR = dataclasses.replace(
    SMALL, arms=("none", "smote"), candidate="smote", comparators=("none",), replica_arms=()
)


def test_the_harness_sets_the_fill_around_each_runner_call():
    seen: list[Fill | None] = []

    def spy(train, request):
        seen.append(CURRENT_FILL.get())
        return FAST_ARMS["smote"](train, request)

    experiment = dataclasses.replace(
        SMALL,
        arms=("none", "smote"),
        candidate="smote",
        comparators=("none",),
        replica_arms=("smote",),
    )
    score_holdout(
        split(GRADED, seed=1, edges=UCI_FRAME.edges),
        experiment,
        {"none": FAST_ARMS["none"], "smote": spy},
        frame=UCI_FRAME,
    )
    assert seen == [Fill(1, "gap"), Fill(1, "replica")]
    assert CURRENT_FILL.get() is None


def test_each_attempt_sees_the_fill_and_a_refusal_resets_it():
    seen: list[Fill | None] = []

    def refuse_once(train, request):
        seen.append(CURRENT_FILL.get())
        if len(seen) == 1:
            raise BaselineError("refused")
        return FAST_ARMS["smote"](train, request)

    def always_refuse(train, request):
        seen.append(CURRENT_FILL.get())
        raise BaselineError("refused")

    score_holdout(
        split(GRADED, seed=0, edges=UCI_FRAME.edges),
        PAIR,
        {"none": FAST_ARMS["none"], "smote": refuse_once},
        frame=UCI_FRAME,
    )
    assert seen == [Fill(0, "gap"), Fill(0, "gap")]
    assert CURRENT_FILL.get() is None
    with pytest.raises(BaselineError):
        score_holdout(
            split(GRADED, seed=0, edges=UCI_FRAME.edges),
            PAIR,
            {"none": FAST_ARMS["none"], "smote": always_refuse},
            frame=UCI_FRAME,
        )
    assert CURRENT_FILL.get() is None
