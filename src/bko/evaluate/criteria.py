"""E1's criteria applied to scores: paired contrasts, corrected intervals and three-valued verdicts.

Every contrast pairs the candidate's score with a reference score of the same holdout. Its interval
is Nadeau and Bengio's corrected resampled t: the variance of the mean is (1/J + n2/n1) S^2, since
the J holdouts overlap. A criterion is met when every interval it reads lies above zero, not met
when one lies below zero at the family-wise level (each of its k intervals read at confidence
1 - (1 - c)/k), and inconclusive otherwise (ADR `research/e1-evaluation-protocol`). The boosting
model's contrasts are read the same way and reported with no verdict (`reported_contrasts`).
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

from bko.evaluate.classifier import Model
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, suffixed


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


@dataclass(frozen=True, slots=True)
class Reported:
    """One interval read as a criterion reads it, for a measure no criterion judges.

    Attributes:
        model: The model whose measure it contrasts.
        criterion: The criterion whose reading it follows.
        scope: The test records the contrast covers.
        arm: The candidate.
        against: The reference arm.
        measure: The measure contrasted.
        interval: The contrast's interval, its family the criterion's.
        differing: The holdouts where the two scores differ.
    """

    model: Model
    criterion: str
    scope: Scope
    arm: str
    against: str
    measure: Measure
    interval: Interval
    differing: int


@dataclass(frozen=True, slots=True)
class Absent:
    """A pair of arms whose contrasts of a model are skipped: one of them holds none of its rows.

    Attributes:
        model: The model whose rows are missing.
        arm: The candidate.
        against: The reference arm.
    """

    model: Model
    arm: str
    against: str


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
        EvaluationError: If fewer than 2 contrasts are given, one is not finite, or `family` is
            under 1.
    """
    if family < 1:
        msg = f"an interval's family needs 1 or more intervals; got {family}"
        raise EvaluationError(msg)
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
    facts = checked(scores, experiment, candidate)
    contrasts: list[Contrast] = []
    for name, scope, pairs, factor, margin in families(facts, experiment):
        contrasts += _criterion(
            name,
            scope,
            pairs,
            facts.series,
            holdouts=facts.holdouts,
            ratio=facts.ratio,
            factor=factor,
            margin=margin,
            confidence=experiment.confidence,
        )
    return contrasts


@dataclass(frozen=True, slots=True)
class Checked:
    """The scores' series and the pairs' common facts, once `judge`'s checks pass.

    Attributes:
        arm: The candidate.
        others: Its comparators.
        series: Each key's score by holdout.
        ratio: The held-out ratio, n2/n1.
        holdouts: The experiment's holdouts.
    """

    arm: str
    others: list[str]
    series: dict[Key, dict[int, float]]
    ratio: float
    holdouts: set[int]


def checked(scores: Sequence[Score], experiment: Experiment, candidate: str | None) -> Checked:
    """Check the scores as `judge` does and return what its contrasts read.

    Raises:
        EvaluationError: As `judge`, but for a missing series.
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
    series = series_of(scores)
    return Checked(arm, others, series, held_out_ratio_of(series), set(holdouts))


def families(
    facts: Checked, experiment: Experiment, *, measures: Sequence[Measure] = JUDGED
) -> list[tuple[str, Scope, list[tuple[Key, Key]], float, float]]:
    """Each criterion held: its name, scope, pairs (of `measures`), reference factor and margin."""
    found = [
        (
            name,
            scope,
            _separation(facts.arm, facts.others, measures, scope),
            1.0,
            experiment.margin,
        )
        for name, scope in CRITERIA
        if name in experiment.criteria
    ]
    if "privacy" in experiment.criteria:
        privacy = [
            (
                Key(facts.arm, Source.ADDED, Measure.DCR_P5, Scope.ALL),
                Key(HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_P5, Scope.ALL),
            )
        ]
        found.append(("privacy", Scope.ALL, privacy, experiment.privacy_margin, 0.0))
    return found


def reported_contrasts(
    scores: Sequence[Score],
    experiment: Experiment,
    *,
    candidate: str | None = None,
    model: Model,
) -> list[Reported | Absent]:
    """Each separation criterion's contrasts of `model`'s measures, read as it reads `JUDGED`'s.

    Each criterion the experiment holds gets its intervals of `model`'s measures at its own family
    size, margin and confidence, with no verdict. A pair of arms either of which holds no
    augmented row of `model` comes first as one `Absent`, and the family still counts it. Scores
    that hold no augmented row of `model` give nothing.

    Raises:
        EvaluationError: As `judge`, for a series of `model`'s measures it reads.
    """
    measures = tuple(suffixed(measure, model) for measure in JUDGED)
    held = {s.arm for s in scores if s.source is Source.AUGMENTED and s.measure in measures}
    if not held:
        return []
    arm = experiment.candidate if candidate is None else candidate
    others = [name for name in experiment.comparators if name != arm]
    found: list[Reported | Absent] = [
        Absent(model, arm, other) for other in others if not {arm, other} <= held
    ]
    present = [other for other in others if {arm, other} <= held]
    series = series_of(scores)
    ratio = held_out_ratio_of(series)
    for name, scope in CRITERIA:
        if name not in experiment.criteria:
            continue
        pairs = _separation(arm, present, measures, scope)
        paired, intervals = _intervals(
            pairs,
            series,
            holdouts=set(experiment.holdout_seeds),
            ratio=ratio,
            factor=1.0,
            margin=experiment.margin,
            confidence=experiment.confidence,
            family=len(measures) * len(others),
        )
        found += [
            Reported(model, name, scope, left.arm, right.arm, left.measure, interval, differing(*b))
            for (left, right), interval, b in zip(pairs, intervals, paired, strict=True)
        ]
    return found


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


def _separation(
    arm: str, others: Sequence[str], measures: Sequence[Measure], scope: Scope
) -> list[tuple[Key, Key]]:
    """The candidate's augmented series of each measure in `scope` against each comparator's."""
    return [
        (Key(arm, Source.AUGMENTED, measure, scope), Key(other, Source.AUGMENTED, measure, scope))
        for measure in measures
        for other in others
    ]


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
    paired, intervals = _intervals(
        pairs,
        series,
        holdouts=holdouts,
        ratio=ratio,
        factor=factor,
        margin=margin,
        confidence=confidence,
    )
    outcome = verdict(intervals)
    return [
        Contrast(
            name, scope, outcome, left.arm, right.arm, left.measure, interval, differing(*both)
        )
        for (left, right), interval, both in zip(pairs, intervals, paired, strict=True)
    ]


def _intervals(
    pairs: Sequence[tuple[Key, Key]],
    series: dict[Key, dict[int, float]],
    *,
    holdouts: set[int],
    ratio: float,
    factor: float,
    margin: float,
    confidence: float,
    family: int | None = None,
) -> tuple[list[tuple[list[float], list[float]]], list[Interval]]:
    """Each pair's scores by holdout and its interval, the family every pair unless given."""
    paired = [paired_scores(series, left, right, holdouts) for left, right in pairs]
    intervals = [
        corrected_interval(
            [a - factor * b - margin for a, b in zip(ours, theirs, strict=True)],
            ratio=ratio,
            confidence=confidence,
            family=len(pairs) if family is None else family,
        )
        for ours, theirs in paired
    ]
    return paired, intervals


def differing(ours: Sequence[float], theirs: Sequence[float]) -> int:
    """The holdouts where the two scores differ."""
    return sum(a != b for a, b in zip(ours, theirs, strict=True))


def paired_scores(
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


def series_of(scores: Sequence[Score]) -> dict[Key, dict[int, float]]:
    """Each key's score by holdout."""
    found: dict[Key, dict[int, float]] = {}
    for score in scores:
        key = Key(score.arm, score.source, score.measure, score.scope)
        found.setdefault(key, {})[score.holdout] = score.value
    return found


def held_out_ratio_of(series: dict[Key, dict[int, float]]) -> float:
    """The largest held-out ratio of `series`.

    Raises:
        EvaluationError: If `series` holds no held-out ratio.
    """
    ratios = series.get(Key(HELD_OUT_ARM, Source.HELD_OUT, Measure.HELD_OUT_RATIO, Scope.ALL))
    if not ratios:
        msg = "the scores hold no held-out ratio"
        raise EvaluationError(msg)
    return max(ratios.values())
