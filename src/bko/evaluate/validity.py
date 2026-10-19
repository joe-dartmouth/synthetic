"""The validity criterion: a BKO arm's candidates that pass every check, summed over the run.

Each BKO arm writes per holdout the candidates `bko.validate` saw and passed, and those it then
dropped, as `Source.CANDIDATES` scores. The rate is `passed` summed over `generated` summed. The
criterion is met when the rate reaches `Experiment.validity`, read as a point with no interval (ADR
`research/e1-evaluation-protocol`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from bko.evaluate.criteria import Verdict
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment
from bko.evaluate.scores import Measure, Score, Source


COUNTS: Final = (
    Measure.GENERATED,
    Measure.PASSED,
    Measure.DROPPED_RANGE,
    Measure.DROPPED_CELL,
    Measure.DROPPED_QUOTA,
)
"""The `candidates` measures, each summed over the holdouts."""


@dataclass(frozen=True, slots=True)
class Validity:
    """One arm's candidate counts over the run.

    Attributes:
        arm: The arm.
        counts: Each of `COUNTS`, summed over the holdouts.
    """

    arm: str
    counts: Mapping[Measure, int]

    @property
    def rate(self) -> Fraction:
        """`passed` over `generated`."""
        return Fraction(self.counts[Measure.PASSED], self.counts[Measure.GENERATED])

    def line(self) -> str:
        """The counts and the rate as `key=value` pairs for a log line."""
        pairs = " ".join(f"{m.value.replace('-', '_')}={self.counts[m]}" for m in COUNTS)
        return f"arm={self.arm} {pairs} rate={float(self.rate):.4f}"


def reported(scores: Sequence[Score]) -> tuple[str, ...]:
    """Every arm holding `candidates` scores, in the order they first appear.

    Args:
        scores: A run's scores.

    Returns:
        The arms' names.
    """
    return tuple(dict.fromkeys(s.arm for s in scores if s.source is Source.CANDIDATES))


def validity(scores: Sequence[Score], experiment: Experiment, arm: str) -> Validity:
    """Sum `arm`'s candidate counts over the experiment's holdouts.

    Args:
        scores: A run's scores.
        experiment: The experiment whose holdouts the counts must cover.
        arm: The arm read.

    Returns:
        The arm's summed counts.

    Raises:
        EvaluationError: If a count misses one of the experiment's holdouts, a count is not a
            whole number, or nothing was generated.
    """
    holdouts = sorted(experiment.holdout_seeds)
    counts: dict[Measure, int] = {}
    for measure in COUNTS:
        values = {
            s.holdout: s.value
            for s in scores
            if s.arm == arm and s.source is Source.CANDIDATES and s.measure is measure
        }
        if sorted(values) != holdouts:
            msg = f"the {measure} counts of {arm!r} cover {len(values)} holdouts, not E1's"
            raise EvaluationError(msg)
        if not all(float(value).is_integer() for value in values.values()):
            msg = f"a {measure} count of {arm!r} is not a whole number"
            raise EvaluationError(msg)
        counts[measure] = int(sum(values.values()))
    if counts[Measure.GENERATED] == 0:
        msg = f"{arm!r} generated nothing, so its validity is undefined"
        raise EvaluationError(msg)
    return Validity(arm, counts)


def verdict(found: Validity, experiment: Experiment) -> Verdict:
    """Met when the rate reaches the experiment's threshold, not met below it.

    Args:
        found: An arm's summed counts.
        experiment: The experiment holding the threshold.

    Returns:
        The validity criterion's verdict.
    """
    return Verdict.MET if found.rate >= experiment.validity else Verdict.NOT_MET
