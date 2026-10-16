"""E1's criteria applied to scores: paired contrasts, corrected intervals and three-valued verdicts.

Every contrast pairs the candidate's score with a reference score of the same holdout. Its interval
is Nadeau and Bengio's corrected resampled t: the variance of the mean is (1/J + n2/n1) S^2, since
the J holdouts overlap. A criterion is met when every interval it reads lies above zero, not met
when one lies below zero at the family-wise level (each of its k intervals read at confidence
1 - (1 - c)/k), and inconclusive otherwise (ADR `research/e1-evaluation-protocol`).
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import t as student

from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source


JUDGED: Final = (Measure.RECALL, Measure.AUC_PR)
"""The measures the separation criteria judge."""

CRITERIA: Final = (("primary", Scope.ALL), ("secondary", Scope.FLAGGED))
"""Each separation criterion and the test records it judges."""

CONTRAST_COLUMNS: Final = (
    "criterion",
    "scope",
    "verdict",
    "arm",
    "against",
    "measure",
    "mean",
    "low",
    "high",
    "refuting_high",
    "holdouts",
    "differing",
)
"""The contrasts CSV's columns, in order."""


class Verdict(enum.StrEnum):
    """A criterion's outcome."""

    MET = "met"
    NOT_MET = "not met"
    INCONCLUSIVE = "inconclusive"


class Key(NamedTuple):
    """One series of scores across holdouts."""

    arm: str
    source: Source
    measure: Measure
    scope: Scope


@dataclass(frozen=True, slots=True)
class Interval:
    """A paired contrast's mean over holdouts and its corrected interval.

    Attributes:
        mean: The mean contrast.
        low: The interval's lower bound, which the met reading checks.
        high: The interval's upper bound.
        refuting_high: The upper bound at the family-wise level, which the not-met reading checks.
        holdouts: The holdouts the contrast pairs.
    """

    mean: float
    low: float
    high: float
    refuting_high: float
    holdouts: int


@dataclass(frozen=True, slots=True)
class Contrast:
    """One interval a criterion reads.

    Attributes:
        criterion: The criterion's name.
        scope: The test records the contrast covers.
        verdict: The criterion's verdict, the same on every contrast it reads.
        arm: The candidate.
        against: The reference arm.
        measure: The measure contrasted.
        interval: The contrast's interval.
        differing: The holdouts where the two scores differ; few means the interval rests on few
            holdouts.
    """

    criterion: str
    scope: Scope
    verdict: Verdict
    arm: str
    against: str
    measure: Measure
    interval: Interval
    differing: int


def refuting_confidence(confidence: float, family: int) -> float:
    """Bonferroni's level for each of `family` intervals read together: 1 - (1 - c)/k."""
    return 1 - (1 - confidence) / family


def corrected_interval(
    contrast: Sequence[float], *, ratio: float, confidence: float, family: int
) -> Interval:
    """Nadeau and Bengio's corrected resampled t interval for the mean of `contrast`.

    Args:
        contrast: One paired contrast per holdout.
        ratio: n2/n1, held-out over training records.
        confidence: The two-sided level of `low` and `high`.
        family: The intervals the criterion reads; `refuting_high` is read at
            `refuting_confidence(confidence, family)`.

    Raises:
        EvaluationError: If fewer than 2 contrasts are given or one is not finite.
    """
    values = np.asarray(contrast, dtype="float64")
    finite = int(np.isfinite(values).sum())
    if len(values) < 2 or finite < len(values):
        msg = f"an interval needs 2 or more finite contrasts; got {finite} of {len(values)}"
        raise EvaluationError(msg)
    holdouts = len(values)
    mean = float(values.mean())
    scale = math.sqrt((1 / holdouts + ratio) * float(values.var(ddof=1)))
    half = _quantile(confidence, holdouts) * scale
    refuting = _quantile(refuting_confidence(confidence, family), holdouts) * scale
    return Interval(mean, mean - half, mean + half, mean + refuting, holdouts)


def _quantile(confidence: float, holdouts: int) -> float:
    return float(student.ppf((1 + confidence) / 2, holdouts - 1))


def verdict(intervals: Sequence[Interval]) -> Verdict:
    """Met if every interval lies above zero, not met if one's refuting bound lies below zero.

    Raises:
        EvaluationError: If there is no interval to read.
    """
    if not intervals:
        msg = "a criterion needs at least one interval"
        raise EvaluationError(msg)
    if all(interval.low > 0 for interval in intervals):
        return Verdict.MET
    if any(interval.refuting_high < 0 for interval in intervals):
        return Verdict.NOT_MET
    return Verdict.INCONCLUSIVE


def judge(
    scores: Sequence[Score], experiment: Experiment, *, candidate: str | None = None
) -> list[Contrast]:
    """Every contrast of the criteria for `candidate`, the experiment's own unless named.

    Raises:
        EvaluationError: If the scores cover other holdouts than the experiment's, hold no
            augmented scores of the candidate, leave it no comparator, hold no held-out ratio, or
            miss a series the criteria read.
    """
    arm = experiment.candidate if candidate is None else candidate
    holdouts = sorted({score.holdout for score in scores})
    if holdouts != sorted(experiment.holdout_seeds):
        msg = f"the scores cover {len(holdouts)} holdouts, not the experiment's"
        raise EvaluationError(msg)
    if not any(score.arm == arm and score.source is Source.AUGMENTED for score in scores):
        msg = f"the scores hold no arm {arm!r}"
        raise EvaluationError(msg)
    others = [name for name in experiment.comparators if name != arm]
    if not others:
        msg = f"the candidate {arm!r} has no comparator"
        raise EvaluationError(msg)
    series = _series(scores)
    ratio = _ratio(series)
    contrasts: list[Contrast] = []
    for name, scope in CRITERIA:
        pairs = [
            (
                Key(arm, Source.AUGMENTED, measure, scope),
                Key(other, Source.AUGMENTED, measure, scope),
            )
            for measure in JUDGED
            for other in others
        ]
        contrasts += _criterion(
            name,
            scope,
            pairs,
            series,
            holdouts=set(holdouts),
            ratio=ratio,
            factor=1.0,
            margin=experiment.margin,
            confidence=experiment.confidence,
        )
    privacy = [
        (
            Key(arm, Source.ADDED, Measure.DCR_P5, Scope.ALL),
            Key(HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_P5, Scope.ALL),
        )
    ]
    contrasts += _criterion(
        "privacy",
        Scope.ALL,
        privacy,
        series,
        holdouts=set(holdouts),
        ratio=ratio,
        factor=experiment.privacy_margin,
        margin=0.0,
        confidence=experiment.confidence,
    )
    return contrasts


def table(contrasts: Sequence[Contrast]) -> str:
    """Write `contrasts` as the contrasts CSV, in the order given."""
    rows = [
        (
            c.criterion,
            c.scope.value,
            c.verdict.value,
            c.arm,
            c.against,
            c.measure.value,
            c.interval.mean,
            c.interval.low,
            c.interval.high,
            c.interval.refuting_high,
            c.interval.holdouts,
            c.differing,
        )
        for c in contrasts
    ]
    return pd.DataFrame(rows, columns=list(CONTRAST_COLUMNS)).to_csv(
        index=False, lineterminator="\n"
    )


def _criterion(
    name: str,
    scope: Scope,
    pairs: Sequence[tuple[Key, Key]],
    series: dict[Key, dict[int, float]],
    *,
    holdouts: set[int],
    ratio: float,
    factor: float,
    margin: float,
    confidence: float,
) -> list[Contrast]:
    paired = [_paired(series, left, right, holdouts) for left, right in pairs]
    intervals = [
        corrected_interval(
            [a - factor * b - margin for a, b in zip(ours, theirs, strict=True)],
            ratio=ratio,
            confidence=confidence,
            family=len(pairs),
        )
        for ours, theirs in paired
    ]
    outcome = verdict(intervals)
    return [
        Contrast(
            name,
            scope,
            outcome,
            left.arm,
            right.arm,
            left.measure,
            interval,
            sum(a != b for a, b in zip(ours, theirs, strict=True)),
        )
        for (left, right), interval, (ours, theirs) in zip(pairs, intervals, paired, strict=True)
    ]


def _paired(
    series: dict[Key, dict[int, float]], left: Key, right: Key, holdouts: set[int]
) -> tuple[list[float], list[float]]:
    """The left and right scores, holdout by holdout in seed order, each on every holdout."""
    for key in (left, right):
        if key not in series:
            msg = f"the scores hold no {key.measure} of {key.source} rows of {key.arm}"
            raise EvaluationError(msg)
    ours, theirs = series[left], series[right]
    if ours.keys() != holdouts or theirs.keys() != holdouts:
        msg = f"{left.arm} and {right.arm} cover different holdouts from the experiment's"
        raise EvaluationError(msg)
    order = sorted(ours)
    return [ours[holdout] for holdout in order], [theirs[holdout] for holdout in order]


def _series(scores: Sequence[Score]) -> dict[Key, dict[int, float]]:
    found: dict[Key, dict[int, float]] = {}
    for score in scores:
        key = Key(score.arm, score.source, score.measure, score.scope)
        found.setdefault(key, {})[score.holdout] = score.value
    return found


def _ratio(series: dict[Key, dict[int, float]]) -> float:
    ratios = series.get(Key(HELD_OUT_ARM, Source.HELD_OUT, Measure.HELD_OUT_RATIO, Scope.ALL))
    if not ratios:
        msg = "the scores hold no held-out ratio"
        raise EvaluationError(msg)
    return max(ratios.values())
