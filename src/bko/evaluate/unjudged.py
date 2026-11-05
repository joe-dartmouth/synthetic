"""Each contrast E1's criteria read, read with no verdict for a run that judges none.

`bko.evaluate.compare` logs them as `unjudged` lines, for the regression and for each other model
whose rows the scores hold. A contrast some holdout leaves undefined comes with no interval, so the
other contrasts are still read.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from bko.evaluate.classifier import Model
from bko.evaluate.criteria import (
    JUDGED,
    Interval,
    checked,
    corrected_interval,
    differing,
    families,
    paired_scores,
)
from bko.evaluate.experiment import Experiment
from bko.evaluate.scores import Measure, Scope, Score, Source


@dataclass(frozen=True, slots=True)
class Unjudged:
    """One contrast a criterion reads, read with no verdict for a run that judges none.

    Attributes:
        criterion: The criterion's name.
        scope: The test records the contrast covers.
        arm: The candidate.
        against: The reference arm.
        measure: The measure contrasted.
        interval: The contrast's interval, its family the criterion's; None when some holdout
            leaves it undefined.
        differing: The holdouts where the two scores differ.
        defined: The holdouts whose contrast is finite.
        holdouts: The holdouts the contrast pairs.
    """

    criterion: str
    scope: Scope
    arm: str
    against: str
    measure: Measure
    interval: Interval | None
    differing: int
    defined: int
    holdouts: int


def judged_of(model: Model) -> tuple[Measure, ...]:
    """`JUDGED` under `model`'s names: the regression's as they are, others suffixed."""
    if model is Model.LR:
        return JUDGED
    return tuple(Measure(f"{measure}-{model}") for measure in JUDGED)


def models_in(scores: Sequence[Score]) -> list[Model]:
    """Each model, in `Model`'s order, whose `JUDGED` measures some augmented score holds."""
    held = {s.measure for s in scores if s.source is Source.AUGMENTED}
    return [model for model in Model if held & set(judged_of(model))]


def unjudged(
    scores: Sequence[Score],
    experiment: Experiment,
    *,
    candidate: str | None = None,
    model: Model = Model.LR,
) -> list[Unjudged]:
    """Each contrast `judge` reads for the experiment's criteria, of `model`'s measures, no verdict.

    A contrast some holdout leaves undefined (a score that is not finite, as when decontamination
    leaves a holdout no positive test record in the flagged cells) comes with no interval, and the
    family still counts it. The privacy criterion reads no model, so only the regression's reading
    holds it.

    Raises:
        EvaluationError: As `judge`, for anything but an undefined contrast.
    """
    facts = checked(scores, experiment, candidate)
    found: list[Unjudged] = []
    read = families(facts, experiment, measures=judged_of(model))
    for name, scope, pairs, factor, margin in read:
        if model is not Model.LR and name == "privacy":
            continue
        for left, right in pairs:
            ours, theirs = paired_scores(facts.series, left, right, facts.holdouts)
            contrast = [a - factor * b - margin for a, b in zip(ours, theirs, strict=True)]
            defined = sum(math.isfinite(value) for value in contrast)
            interval = (
                corrected_interval(
                    contrast,
                    ratio=facts.ratio,
                    confidence=experiment.confidence,
                    family=len(pairs),
                )
                if defined == len(contrast)
                else None
            )
            found.append(
                Unjudged(
                    name,
                    scope,
                    left.arm,
                    right.arm,
                    left.measure,
                    interval,
                    differing(ours, theirs),
                    defined,
                    len(contrast),
                )
            )
    return found
