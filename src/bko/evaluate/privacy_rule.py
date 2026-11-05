"""The privacy reading rule: each arm's paired contrast with a reference arm, logged per measure.

`readings` reads each arm's contrast with the reference arm per holdout through
`criteria.corrected_interval`, at Bonferroni's level over its family: the arms with a defined
contrast in the measure's group times the group's measures. A NaN holdout is counted and left out,
never read as 0. `log_readings` logs one `privacy arm=` line per reading; no criterion is judged
(`bko.evaluate.membership`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from bko.evaluate.criteria import (
    corrected_interval,
    held_out_ratio_of,
    refuting_confidence,
    series_of,
)
from bko.evaluate.scores import Measure, Score, Source


logger = logging.getLogger("bko.evaluate")

CONFIDENCE: Final = 0.95
"""The family-wise level of the rule's intervals."""


@dataclass(frozen=True, slots=True)
class Group:
    """Measures read as one family.

    Attributes:
        scope: The scope its log lines name.
        measures: Its measures.
    """

    scope: str
    measures: tuple[Measure, ...]


@dataclass(frozen=True, slots=True)
class Reading:
    """One arm's paired contrast with the reference arm on one measure, as the rule reads it.

    Attributes:
        arm: The arm.
        scope: The attack's scope, or `all` for the DCR and NNDR measures.
        measure: The measure.
        mean: The arm's mean over its defined holdouts.
        against: The reference arm.
        contrast: The mean of the defined contrasts.
        low: The interval's lower bound at Bonferroni's level over `family`.
        high: Its upper bound.
        family: The intervals read together.
        holdouts: The run's holdouts.
        undefined: The holdouts whose contrast is NaN, which the interval leaves out.
        exceeds_reference: Whether the interval lies above 0: the arm's value exceeds the
            reference's. A higher attack statistic leaks more; a higher distance leaks less.
    """

    arm: str
    scope: str
    measure: Measure
    mean: float
    against: str
    contrast: float
    low: float
    high: float
    family: int
    holdouts: int
    undefined: int
    exceeds_reference: bool


def readings(
    scores: Sequence[Score], groups: Sequence[Group], *, arms: Sequence[str], reference: str
) -> list[Reading]:
    """The rule's reading of every arm but `reference` on every measure of `groups` `scores` holds.

    An arm's contrast on a holdout is NaN when either value is; the interval reads the defined
    ones, and needs two. A family's size is its arms with a defined interval on any of its measures
    times its measures; an arm with none reads NaN, never 0.

    Raises:
        EvaluationError: If `scores` hold no held-out ratio.
    """
    series = series_of([s for s in scores if s.source is not Source.ADDED])
    ratio = held_out_ratio_of(series)
    values: dict[tuple[str, Measure], dict[int, float]] = {
        (s.arm, s.measure): {} for s in scores if s.source is Source.ADDED
    }
    for s in scores:
        if s.source is Source.ADDED:
            values[s.arm, s.measure][s.holdout] = s.value
    holdouts = sorted({s.holdout for s in scores})
    others = [arm for arm in arms if arm != reference]
    found: list[Reading] = []
    for group in groups:
        present = [m for m in group.measures if any(key[1] is m for key in values)]
        contrasts = {
            (arm, m): _contrasts(values, arm, reference, m, holdouts)
            for arm in others
            for m in present
        }
        defined = {arm for (arm, _), c in contrasts.items() if np.isfinite(c).sum() >= 2}
        family = len(defined) * len(group.measures)
        for arm in others:
            found.extend(
                _reading(
                    arm,
                    group.scope,
                    measure,
                    values.get((arm, measure), {}),
                    contrasts[arm, measure],
                    reference=reference,
                    family=family,
                    ratio=ratio,
                )
                for measure in present
            )
    return found


def _contrasts(
    values: Mapping[tuple[str, Measure], Mapping[int, float]],
    arm: str,
    reference: str,
    measure: Measure,
    holdouts: Sequence[int],
) -> np.ndarray:
    ours, theirs = values.get((arm, measure), {}), values.get((reference, measure), {})
    return np.array([ours.get(h, math.nan) - theirs.get(h, math.nan) for h in holdouts])


def _reading(
    arm: str,
    scope: str,
    measure: Measure,
    own: Mapping[int, float],
    contrasts: np.ndarray,
    *,
    reference: str,
    family: int,
    ratio: float,
) -> Reading:
    finite = contrasts[np.isfinite(contrasts)]
    present = [v for v in own.values() if math.isfinite(v)]
    mean = float(np.mean(present)) if present else math.nan
    contrast = low = high = math.nan
    exceeds = False
    if len(finite) >= 2 and family:
        level = refuting_confidence(CONFIDENCE, family)
        interval = corrected_interval(finite.tolist(), ratio=ratio, confidence=level, family=1)
        contrast, low, high = interval.mean, interval.low, interval.high
        exceeds = low > 0
    undefined = len(contrasts) - len(finite)
    return Reading(
        arm,
        scope,
        measure,
        mean,
        reference,
        contrast,
        low,
        high,
        family,
        len(contrasts),
        undefined,
        exceeds,
    )


def log_readings(found: Iterable[Reading]) -> None:
    """Log each reading's `privacy arm=` line."""
    for r in found:
        logger.info(
            "privacy arm=%s scope=%s measure=%s mean=%.6f against=%s contrast=%.6f low=%.6f"
            " high=%.6f family=%d holdouts=%d undefined=%d exceeds_reference=%s",
            r.arm,
            r.scope,
            r.measure,
            r.mean,
            r.against,
            r.contrast,
            r.low,
            r.high,
            r.family,
            r.holdouts,
            r.undefined,
            "true" if r.exceeds_reference else "false",
        )
