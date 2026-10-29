"""The harness names the holdout and the fill each runner call serves, for log attribution."""

from __future__ import annotations

import dataclasses

import pytest

from bko.datasets.split import split
from bko.evaluate.fills import CURRENT_FILL, Fill
from bko.evaluate.run import score_holdout
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


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
    score_holdout(split(GRADED, seed=0), experiment, {"none": FAST_ARMS["none"], "smote": spy})
    assert seen == [Fill(0, "gap"), Fill(0, "replica")]
    assert CURRENT_FILL.get() is None
