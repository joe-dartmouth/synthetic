"""`compare --beside`: one run's arm read beside an arm of another scores file, with no verdict.

Each reading pairs the two arms' scores holdout by holdout, ours minus theirs, under Nadeau and
Bengio's corrected interval at the run's confidence with family 1. A holdout pairs only when both
files record the same held-out rows' scores on it, and no arm in either file copied a test record
there: `score` drops every copied record from the test split, so a copy in either run would put
the two arms' classifiers on different test records.
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


def pairable(ours: Sequence[Score], theirs: Sequence[Score]) -> tuple[frozenset[int], int]:
    """The shared holdouts whose held-out rows match and where every arm has no test copy.

    The held-out rows match when both files record the same held-out scores (the held-out ratio
    and the test rows' distances to the training rows) under `HELD_OUT_ARM`, and record some. No
    arm has a test copy when every `test-copies` score of either file, added and replica, is 0.

    Returns:
        The holdouts that pair, and how many shared holdouts were excluded.
    """
    shared = {s.holdout for s in ours} & {s.holdout for s in theirs}
    held_ours, held_theirs = _held_out(ours), _held_out(theirs)
    copied = {
        s.holdout for s in (*ours, *theirs) if s.measure is Measure.TEST_COPIES and s.value != 0
    }
    paired = frozenset(
        holdout
        for holdout in shared
        if holdout not in copied
        and holdout in held_ours
        and _same(held_ours[holdout], held_theirs.get(holdout, {}))
    )
    return paired, len(shared) - len(paired)


def pair(
    ours: Sequence[Score],
    theirs: Sequence[Score],
    *,
    arm: str,
    against: str,
    holdouts: frozenset[int],
    ratio: float,
    confidence: float,
) -> tuple[Beside, ...]:
    """Ours minus theirs, family 1, over each series of `PAIRED` that both arms hold.

    Args:
        ours: The run's scores.
        theirs: The other file's scores.
        arm: Our arm.
        against: Their arm.
        holdouts: The holdouts `pairable` found.
        ratio: n2/n1 of the corrected interval, held-out over training records.
        confidence: The two-sided level of each interval.

    Raises:
        EvaluationError: If fewer than 2 holdouts pair, either file holds no augmented scores of
            its arm, or a series both arms hold is defined on fewer than 2 paired holdouts.
    """
    if len(holdouts) < 2:
        msg = f"a reading beside another run needs 2 or more paired holdouts; got {len(holdouts)}"
        raise EvaluationError(msg)
    for scores, name in ((ours, arm), (theirs, against)):
        if not any(s.arm == name and s.source is Source.AUGMENTED for s in scores):
            msg = f"the scores hold no augmented scores of {name!r}"
            raise EvaluationError(msg)
    left, right = series_of(ours), series_of(theirs)
    found: list[Beside] = []
    for source, measure, scope in PAIRED:
        mine, others = (
            left.get(Key(arm, source, measure, scope)),
            right.get(Key(against, source, measure, scope)),
        )
        if mine is None or others is None:
            continue
        pairs = [
            (mine.get(holdout, math.nan), others.get(holdout, math.nan))
            for holdout in sorted(holdouts)
        ]
        defined = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
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
    """Equal keys and values, NaN equal to NaN."""
    return a.keys() == b.keys() and all(
        a[key] == b[key] or (math.isnan(a[key]) and math.isnan(b[key])) for key in a
    )
