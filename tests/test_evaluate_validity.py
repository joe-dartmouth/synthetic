"""The validity reading: candidates passed over generated, summed over E1's holdouts."""

from __future__ import annotations

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


def counts(arm: str, per_holdout: Mapping[Measure, float]) -> list[Score]:
    """`arm`'s `candidates` rows, `per_holdout` on every holdout of `SMALL`."""
    return [
        Score(holdout, arm, Source.CANDIDATES, measure, Scope.ALL, value)
        for holdout in SMALL.holdout_seeds
        for measure, value in per_holdout.items()
    ]


def test_the_rate_sums_over_holdouts_and_a_tie_meets():
    found = validity(counts("bko", FULL), SMALL, "bko")
    assert found.rate == Fraction(38, 40)
    assert verdict(found, SMALL) is Verdict.MET


def test_a_rate_below_the_threshold_is_not_met():
    found = validity(counts("bko", {**FULL, Measure.PASSED: 18.0}), SMALL, "bko")
    assert verdict(found, SMALL) is Verdict.NOT_MET


def test_nothing_generated_is_refused():
    empty = {**FULL, Measure.GENERATED: 0.0, Measure.PASSED: 0.0}
    with pytest.raises(EvaluationError, match="nothing"):
        validity(counts("bko", empty), SMALL, "bko")


def test_a_missing_holdout_is_refused():
    with pytest.raises(EvaluationError, match="holdouts"):
        validity(counts("bko", FULL)[:5], SMALL, "bko")


def test_reported_lists_every_arm_with_candidates_rows_in_order():
    scores = [*counts("bko-frequency", FULL), *counts("bko", FULL)]
    assert reported(scores) == ("bko-frequency", "bko")


def test_a_count_that_is_not_whole_is_refused():
    with pytest.raises(EvaluationError, match="not a whole number"):
        validity(counts("bko", {**FULL, Measure.PASSED: 18.5}), SMALL, "bko")
