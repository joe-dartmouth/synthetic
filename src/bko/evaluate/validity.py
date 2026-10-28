"""The validity criterion: a BKO arm's candidates that pass every check, summed over the run.

Each BKO arm writes per holdout and fill the candidates `bko.validate` saw and passed, and those
it then dropped, as `Source.CANDIDATES` (the gap fill) and `Source.REPLICA_CANDIDATES` (the replica
fill) scores; a prescribing arm also writes `OPTIONAL`'s counts, summed only for an arm that holds
them. The rate is `passed` summed over `generated` summed over both fills; each fill is
reported apart with no verdict. The criterion is met when the rate reaches `Experiment.validity`,
read as a point with no interval (ADR `research/e1-evaluation-protocol`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
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

OPTIONAL: Final = (Measure.DROPPED_CONCEPTS,)
"""The `candidates` measures only a prescribing arm writes, each summed when the arm holds it."""


FILLS: Final = MappingProxyType({Source.CANDIDATES: "gap", Source.REPLICA_CANDIDATES: "replica"})
"""Each fill's candidates source, by the name a log line gives the fill."""


@dataclass(frozen=True, slots=True)
class Validity:
    """One arm's candidate counts over the run, by fill.

    Attributes:
        arm: The arm.
        fills: Each fill's `COUNTS`, and each of `OPTIONAL` the arm holds, summed over the
            holdouts, by its source in `FILLS`; the replica fill's only for an arm that fills one.
    """

    arm: str
    fills: Mapping[Source, Mapping[Measure, int]]

    @property
    def counts(self) -> dict[Measure, int]:
        """Each of `COUNTS`, then each of `OPTIONAL` the arm holds, summed.

        Each is summed over the holdouts and the fills.
        """
        held = [m for m in OPTIONAL if any(m in found for found in self.fills.values())]
        return {m: sum(found.get(m, 0) for found in self.fills.values()) for m in (*COUNTS, *held)}

    @property
    def rate(self) -> Fraction:
        """`passed` over `generated` over every fill: the rate the criterion reads."""
        return _rate(self.counts)

    def line(self) -> str:
        """The counts and the rate over every fill as `key=value` pairs for a log line."""
        return f"arm={self.arm} {_pairs(self.counts)}"

    def fill_lines(self) -> tuple[str, ...]:
        """Each fill's counts and rate as `key=value` pairs for a log line; no verdict."""
        return tuple(
            f"arm={self.arm} fill={FILLS[source]} {_pairs(found)}"
            for source, found in self.fills.items()
        )


def _rate(counts: Mapping[Measure, int]) -> Fraction:
    return Fraction(counts[Measure.PASSED], counts[Measure.GENERATED])


def _pairs(counts: Mapping[Measure, int]) -> str:
    pairs = " ".join(f"{m.value.replace('-', '_')}={count}" for m, count in counts.items())
    return f"{pairs} rate={float(_rate(counts)):.4f}"


def reported(scores: Sequence[Score]) -> tuple[str, ...]:
    """Every arm holding candidate scores of either fill, in the order they first appear.

    Args:
        scores: A run's scores.

    Returns:
        The arms' names.
    """
    return tuple(dict.fromkeys(s.arm for s in scores if s.source in FILLS))


def validity(scores: Sequence[Score], experiment: Experiment, arm: str) -> Validity:
    """Sum `arm`'s candidate counts over the experiment's holdouts, fill by fill.

    The replica fill's counts are read exactly when `arm` is one of the experiment's replica arms.

    Args:
        scores: A run's scores.
        experiment: The experiment whose holdouts and replica arms the counts must match.
        arm: The arm read.

    Returns:
        The arm's summed counts, by fill.

    Raises:
        EvaluationError: If a count misses one of the experiment's holdouts, a count is not a
            whole number, a fill generated nothing, or an arm that fills no replica holds replica
            counts.
    """
    replica = arm in experiment.replica_arms
    if not replica and any(s.arm == arm and s.source is Source.REPLICA_CANDIDATES for s in scores):
        msg = f"{arm!r} holds replica fill counts but fills no replica in this experiment"
        raise EvaluationError(msg)
    sources = tuple(FILLS) if replica else (Source.CANDIDATES,)
    return Validity(arm, {source: _summed(scores, experiment, arm, source) for source in sources})


def _summed(
    scores: Sequence[Score], experiment: Experiment, arm: str, source: Source
) -> dict[Measure, int]:
    """`arm`'s counts under `source`, each summed over the experiment's holdouts.

    Each of `OPTIONAL` is read when `arm` holds it under `source`, and then must cover every
    holdout as `COUNTS` do.
    """
    fill = FILLS[source]
    holdouts = sorted(experiment.holdout_seeds)
    held = {s.measure for s in scores if s.arm == arm and s.source is source}
    counts: dict[Measure, int] = {}
    for measure in (*COUNTS, *(m for m in OPTIONAL if m in held)):
        values = {
            s.holdout: s.value
            for s in scores
            if s.arm == arm and s.source is source and s.measure is measure
        }
        if sorted(values) != holdouts:
            msg = (
                f"the {measure} counts of {arm!r} in its {fill} fill cover {len(values)} "
                "holdouts, not E1's"
            )
            raise EvaluationError(msg)
        if not all(float(value).is_integer() for value in values.values()):
            msg = f"a {measure} count of {arm!r} in its {fill} fill is not a whole number"
            raise EvaluationError(msg)
        counts[measure] = int(sum(values.values()))
    if counts[Measure.GENERATED] == 0:
        msg = f"{arm!r} generated nothing in its {fill} fill, so its rate is undefined"
        raise EvaluationError(msg)
    return counts


def verdict(found: Validity, experiment: Experiment) -> Verdict:
    """Met when the rate reaches the experiment's threshold, not met below it.

    Args:
        found: An arm's summed counts.
        experiment: The experiment holding the threshold.

    Returns:
        The validity criterion's verdict.
    """
    return Verdict.MET if found.rate >= experiment.validity else Verdict.NOT_MET
