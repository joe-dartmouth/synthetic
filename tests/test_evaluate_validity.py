"""The validity reading: candidates passed over generated, summed over E1's holdouts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from fractions import Fraction
from typing import Final

import pytest

from bko.evaluate.criteria import Verdict
from bko.evaluate.errors import EvaluationError
from bko.evaluate.scores import Measure, Scope, Score, Source
from bko.evaluate.validity import reported, validity, verdict
from tests._evaluate_fixtures import SMALL


FULL: Final = {
    Measure.GENERATED: 20.0,
    Measure.PASSED: 19.0,
    Measure.DROPPED_RANGE: 1.0,
    Measure.DROPPED_CELL: 0.0,
    Measure.DROPPED_QUOTA: 0.0,
}
"""One holdout's counts: 19 of 20 passed, one of them dropped for range."""


NO_REPLICA: Final = dataclasses.replace(SMALL, replica_arms=())
"""`SMALL` with no replica arm: a BKO arm fills only the gap request."""

WITH_REPLICA: Final = dataclasses.replace(SMALL, replica_arms=("bko",))
"""`SMALL` with `bko` also filling a replica."""

REPLICA: Final = {
    Measure.GENERATED: 100.0,
    Measure.PASSED: 100.0,
    Measure.DROPPED_RANGE: 3.0,
    Measure.DROPPED_CELL: 0.0,
    Measure.DROPPED_QUOTA: 2.0,
}
"""One holdout's replica counts: every candidate passed, five then dropped."""


def counts(
    arm: str, per_holdout: Mapping[Measure, float], source: Source = Source.CANDIDATES
) -> list[Score]:
    """`arm`'s rows under `source`, `per_holdout` on every holdout of `SMALL`."""
    return [
        Score(holdout, arm, source, measure, Scope.ALL, value)
        for holdout in SMALL.holdout_seeds
        for measure, value in per_holdout.items()
    ]


def test_the_rate_sums_over_holdouts_and_a_tie_meets():
    found = validity(counts("bko", FULL), NO_REPLICA, "bko")
    assert found.rate == Fraction(38, 40)
    assert verdict(found, NO_REPLICA) is Verdict.MET


def test_a_rate_below_the_threshold_is_not_met():
    found = validity(counts("bko", {**FULL, Measure.PASSED: 18.0}), NO_REPLICA, "bko")
    assert verdict(found, NO_REPLICA) is Verdict.NOT_MET


def test_nothing_generated_is_refused():
    empty = {**FULL, Measure.GENERATED: 0.0, Measure.PASSED: 0.0}
    with pytest.raises(EvaluationError, match="nothing"):
        validity(counts("bko", empty), NO_REPLICA, "bko")


def test_a_missing_holdout_is_refused():
    with pytest.raises(EvaluationError, match="holdouts"):
        validity(counts("bko", FULL)[:5], NO_REPLICA, "bko")


def test_reported_lists_every_arm_with_candidates_rows_in_order():
    scores = [*counts("bko-frequency", FULL), *counts("bko", FULL)]
    assert reported(scores) == ("bko-frequency", "bko")


def test_a_count_that_is_not_whole_is_refused():
    with pytest.raises(EvaluationError, match="not a whole number"):
        validity(counts("bko", {**FULL, Measure.PASSED: 18.5}), NO_REPLICA, "bko")


def both(replica: Mapping[Measure, float] = REPLICA) -> list[Score]:
    """`bko`'s gap counts `FULL` and its replica counts `replica`, on every holdout."""
    return [*counts("bko", FULL), *counts("bko", replica, Source.REPLICA_CANDIDATES)]


def test_the_rate_reads_both_fills_and_each_fill_is_reported_apart():
    found = validity(both(), WITH_REPLICA, "bko")
    assert found.rate == Fraction(238, 240)
    assert found.line() == (
        "arm=bko generated=240 passed=238 dropped_range=8 dropped_cell=0 dropped_quota=4 "
        "rate=0.9917"
    )
    assert found.fill_lines() == (
        (
            "arm=bko fill=gap generated=40 passed=38 dropped_range=2 dropped_cell=0 "
            "dropped_quota=0 rate=0.9500"
        ),
        (
            "arm=bko fill=replica generated=200 passed=200 dropped_range=6 dropped_cell=0 "
            "dropped_quota=4 rate=1.0000"
        ),
    )


def test_the_replica_fill_moves_the_verdict():
    found = validity(both({**REPLICA, Measure.PASSED: 80.0}), WITH_REPLICA, "bko")
    assert verdict(found, WITH_REPLICA) is Verdict.NOT_MET


def test_a_replica_arm_without_replica_counts_is_refused():
    with pytest.raises(EvaluationError, match="in its replica fill cover 0 holdouts"):
        validity(counts("bko", FULL), WITH_REPLICA, "bko")


def test_replica_counts_of_an_arm_that_fills_no_replica_are_refused():
    with pytest.raises(EvaluationError, match="fills no replica"):
        validity(both(), NO_REPLICA, "bko")


def test_a_fill_that_generated_nothing_is_refused():
    empty = {**REPLICA, Measure.GENERATED: 0.0, Measure.PASSED: 0.0}
    with pytest.raises(EvaluationError, match="nothing in its replica fill"):
        validity(both(empty), WITH_REPLICA, "bko")


def test_a_replica_count_that_is_not_whole_is_refused():
    with pytest.raises(EvaluationError, match="replica fill is not a whole number"):
        validity(both({**REPLICA, Measure.PASSED: 99.5}), WITH_REPLICA, "bko")


def test_reported_lists_an_arm_with_only_replica_rows():
    assert reported(counts("bko", REPLICA, Source.REPLICA_CANDIDATES)) == ("bko",)


def test_a_replica_arm_without_gap_counts_is_refused():
    with pytest.raises(EvaluationError, match="in its gap fill cover 0 holdouts"):
        validity(counts("bko", REPLICA, Source.REPLICA_CANDIDATES), WITH_REPLICA, "bko")
