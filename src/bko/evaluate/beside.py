"""`compare --beside`: one run's arm read beside an arm of another scores file, with no verdict.

Each reading pairs the two arms' scores holdout by holdout, ours minus theirs, under Nadeau and
Bengio's corrected interval at the run's confidence with family 1. A holdout pairs only when both
files record the same held-out rows' scores on it (`pairable` holds the rule), and no arm in either
file copied a test record there: `score` drops every copied record from the test split, so a copy
in either run would put the two arms' classifiers on different test records. A series either arm
lacks, or that fewer than 2 paired holdouts define, is skipped rather than refused.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from bko.evaluate.criteria import Interval, Key, corrected_interval, series_of
from bko.evaluate.errors import EvaluationError
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source


_UTILITY: Final = tuple(
    (measure, scope)
    for measure in (Measure.RECALL, Measure.AUC_PR)
    for scope in (Scope.ALL, Scope.FLAGGED)
)

PAIRED: Final = (
    *((Source.AUGMENTED, measure, scope) for measure, scope in _UTILITY),
    (Source.ADDED, Measure.DCR_P5, Scope.ALL),
    *((Source.REPLICA, measure, scope) for measure, scope in _UTILITY),
    *(
        (Source.REPLICA, measure, Scope.ALL)
        for measure in (Measure.QUALITY, Measure.COLUMN_SHAPES, Measure.COLUMN_PAIR_TRENDS)
    ),
)
"""The series a reading pairs, in the order it reads them."""


@dataclass(frozen=True, slots=True)
class Beside:
    """One series of an arm read beside another file's arm.

    Attributes:
        source: The rows the scores describe.
        measure: The measure paired.
        scope: The test records it covers.
        interval: Ours minus theirs over the holdouts where both are defined.
        differing: The holdouts of the interval where the two scores differ.
        undefined: The paired holdouts left out because either score is NaN or absent.
    """

    source: Source
    measure: Measure
    scope: Scope
    interval: Interval
    differing: int
    undefined: int


@dataclass(frozen=True, slots=True)
class Skipped:
    """A series of `PAIRED` the reading leaves out.

    Attributes:
        source: The rows the scores describe.
        measure: The measure.
        scope: The test records it covers.
        reason: `absent` when either arm lacks the series, `undefined` when fewer than 2 paired
            holdouts define both scores.
        defined: The paired holdouts where both scores are defined.
    """

    source: Source
    measure: Measure
    scope: Scope
    reason: str
    defined: int


@dataclass(frozen=True, slots=True)
class Pairing:
    """The holdouts a reading pairs, and why the other shared holdouts do not.

    Attributes:
        holdouts: The holdouts that pair.
        shared: The holdouts both files hold; a holdout one file holds is not counted.
        copied: The shared holdouts excluded because an arm in either file copied a test record.
        mismatched: The other excluded shared holdouts, whose held-out rows do not match.
    """

    holdouts: frozenset[int]
    shared: int
    copied: int
    mismatched: int

    @property
    def excluded(self) -> int:
        """The shared holdouts that do not pair."""
        return self.shared - len(self.holdouts)


def pairable(ours: Sequence[Score], theirs: Sequence[Score]) -> Pairing:
    """The shared holdouts whose held-out rows match and where no arm copied a test record.

    The held-out rows match on a holdout when the `none` arm's `held-out` scores (the held-out
    ratio, `dcr-p5` and `dcr-median`) are present in both files with equal keys and bitwise-equal
    values, NaN equal to NaN. Bitwise, because a replay reproduces them bit for bit. The scores
    file records no row identity, so equal summaries are the evidence that the two splits match.
    An arm copied a test record when any `test-copies` score of either file, of any arm, added or
    replica, is not 0; NaN counts as a copy.

    Returns:
        The holdouts that pair, and the shared holdouts excluded for a copy and for a mismatch.
    """
    shared = {s.holdout for s in ours} & {s.holdout for s in theirs}
    held_ours, held_theirs = _held_out(ours), _held_out(theirs)
    copied = shared & {
        s.holdout for s in (*ours, *theirs) if s.measure is Measure.TEST_COPIES and s.value != 0
    }
    paired = frozenset(
        holdout
        for holdout in shared - copied
        if holdout in held_ours and _same(held_ours[holdout], held_theirs.get(holdout, {}))
    )
    return Pairing(paired, len(shared), len(copied), len(shared) - len(copied) - len(paired))


def check(
    ours: Sequence[Score],
    theirs: Sequence[Score],
    *,
    arm: str,
    against: str,
    holdouts: frozenset[int],
) -> None:
    """Refuse a reading `pair` cannot make.

    Raises:
        EvaluationError: If fewer than 2 holdouts pair, or either file holds no augmented scores
            of its arm.
    """
    if len(holdouts) < 2:
        msg = f"a reading beside another run needs 2 or more paired holdouts; got {len(holdouts)}"
        raise EvaluationError(msg)
    for scores, name in ((ours, arm), (theirs, against)):
        if not any(s.arm == name and s.source is Source.AUGMENTED for s in scores):
            msg = f"the scores hold no augmented scores of {name!r}"
            raise EvaluationError(msg)


def pair(
    ours: Sequence[Score],
    theirs: Sequence[Score],
    *,
    arm: str,
    against: str,
    holdouts: frozenset[int],
    ratio: float,
    confidence: float,
) -> tuple[Beside | Skipped, ...]:
    """Ours minus theirs, family 1, over each series of `PAIRED`, in its order.

    A series either arm lacks, or that fewer than 2 paired holdouts define, is `Skipped`.

    Args:
        ours: The run's scores.
        theirs: The other file's scores.
        arm: Our arm.
        against: Their arm.
        holdouts: The holdouts `pairable` found.
        ratio: n2/n1 of the corrected interval, held-out over training records.
        confidence: The two-sided level of each interval.

    Raises:
        EvaluationError: As `check`.
    """
    check(ours, theirs, arm=arm, against=against, holdouts=holdouts)
    left, right = series_of(ours), series_of(theirs)
    found: list[Beside | Skipped] = []
    for source, measure, scope in PAIRED:
        mine, others = (
            left.get(Key(arm, source, measure, scope)),
            right.get(Key(against, source, measure, scope)),
        )
        if mine is None or others is None:
            found.append(Skipped(source, measure, scope, "absent", 0))
            continue
        pairs = [
            (mine.get(holdout, math.nan), others.get(holdout, math.nan))
            for holdout in sorted(holdouts)
        ]
        defined = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
        if len(defined) < 2:
            found.append(Skipped(source, measure, scope, "undefined", len(defined)))
            continue
        interval = corrected_interval(
            [a - b for a, b in defined], ratio=ratio, confidence=confidence, family=1
        )
        differing = sum(a != b for a, b in defined)
        found.append(Beside(source, measure, scope, interval, differing, len(pairs) - len(defined)))
    return tuple(found)


def _held_out(scores: Sequence[Score]) -> dict[int, dict[tuple[Measure, Scope], float]]:
    found: dict[int, dict[tuple[Measure, Scope], float]] = {}
    for s in scores:
        if s.arm == HELD_OUT_ARM and s.source is Source.HELD_OUT:
            found.setdefault(s.holdout, {})[s.measure, s.scope] = s.value
    return found


def _same(a: dict[tuple[Measure, Scope], float], b: dict[tuple[Measure, Scope], float]) -> bool:
    """Equal keys and bitwise-equal values, NaN equal to NaN."""
    return a.keys() == b.keys() and all(float(a[key]).hex() == float(b[key]).hex() for key in a)
