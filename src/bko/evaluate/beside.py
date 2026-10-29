"""`compare --beside`: one run's arm read beside an arm of another scores file, with no verdict.

Each reading pairs the two arms' scores holdout by holdout, ours minus theirs, under Nadeau and
Bengio's corrected interval at the run's confidence with family 1. A holdout pairs only when both
files record the same held-out rows' scores on it and flag the same cells (`pairable` holds the
rule), and no arm in either file copied a test record there: under E1's rule `score` drops every
copied record from the test split, so a copy in either run would put the two arms' classifiers on
different test records. A run that keeps copies (`Experiment.decontaminate` false) scores every arm
on the whole split and writes a `copies-kept` held-out row per holdout, so a copy excludes nothing
on a holdout where both files hold that row. A series either arm lacks, or that fewer than 2
paired holdouts define, is skipped rather than refused.
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
        mismatched: The other excluded shared holdouts, whose held-out rows or flagged cells do
            not match.
    """

    holdouts: frozenset[int]
    shared: int
    copied: int
    mismatched: int

    @property
    def excluded(self) -> int:
        """The shared holdouts that do not pair."""
        return self.shared - len(self.holdouts)


def pairable(
    ours: Sequence[Score], theirs: Sequence[Score], *, allow_copies: bool = False
) -> Pairing:
    """The shared holdouts whose held-out rows and flagged cells match, and where no arm copied.

    The held-out rows match on a holdout when the `none` arm's `held-out` scores other than the
    flagged fingerprint and `copies-kept` (the held-out ratio, `dcr-p5` and `dcr-median`) are
    present in both files with equal keys and bitwise-equal values, NaN equal to NaN. Bitwise,
    because a replay reproduces them bit for bit. The scores file records no row identity, so
    equal summaries are the evidence that the two splits match. The flagged cells match when the
    two `flagged-fingerprint` scores are bitwise equal, or either file lacks one. An arm copied a
    test record when any `test-copies` score of either file, of any arm, added or replica, is not
    0; NaN counts as a copy. With `allow_copies`, a copy excludes nothing on a holdout where both
    files hold a `copies-kept` row, since both scored every arm on the whole test split there; on
    any other holdout a copy excludes it as without `allow_copies`, since the file that does not
    keep copies scored its arms on a split without the copied records.

    Returns:
        The holdouts that pair, and the shared holdouts excluded for a copy and for a mismatch.
    """
    shared = {s.holdout for s in ours} & {s.holdout for s in theirs}
    held_ours, held_theirs = _held_out(ours), _held_out(theirs)
    copied = shared & {
        s.holdout for s in (*ours, *theirs) if s.measure is Measure.TEST_COPIES and s.value != 0
    }
    if allow_copies:
        copied -= _keeps(held_ours) & _keeps(held_theirs)
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


_FINGERPRINT: Final = (Measure.FLAGGED_FINGERPRINT, Scope.ALL)

_KEPT: Final = (Measure.COPIES_KEPT, Scope.ALL)

_UNMATCHED: Final = frozenset({_FINGERPRINT, _KEPT})
"""The held-out keys left out of the summaries' equality: the fingerprint is compared on its own,
and `copies-kept` says how the arms were scored, not which split they were scored on."""


def _keeps(held: dict[int, dict[tuple[Measure, Scope], float]]) -> set[int]:
    """The holdouts whose held-out rows hold a `copies-kept` row."""
    return {holdout for holdout, values in held.items() if _KEPT in values}


def _same(a: dict[tuple[Measure, Scope], float], b: dict[tuple[Measure, Scope], float]) -> bool:
    """Whether two holdouts' held-out scores match.

    The scores but the fingerprint and `copies-kept` have equal keys and bitwise-equal values, NaN
    equal to NaN, and the fingerprints are equal when both hold one.
    """
    if _FINGERPRINT in a and _FINGERPRINT in b and not _bitwise(a[_FINGERPRINT], b[_FINGERPRINT]):
        return False
    rest_a = {key: value for key, value in a.items() if key not in _UNMATCHED}
    rest_b = {key: value for key, value in b.items() if key not in _UNMATCHED}
    return rest_a.keys() == rest_b.keys() and all(
        _bitwise(rest_a[key], rest_b[key]) for key in rest_a
    )


def _bitwise(a: float, b: float) -> bool:
    return float(a).hex() == float(b).hex()
