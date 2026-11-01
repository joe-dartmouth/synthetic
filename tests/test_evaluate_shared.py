"""A holdout scored on a shared split: excluded records, its marker row, and `beside`'s pairing."""

from __future__ import annotations

import dataclasses
import logging
from hashlib import sha256

import pytest

import bko.evaluate.run as run_module
from bko.datasets.split import split
from bko.datasets.uci import FRAME_COLUMNS, UCI_FRAME
from bko.evaluate.beside import Pairing, pairable
from bko.evaluate.run import score_holdout
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, render
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

PAIR = dataclasses.replace(
    SMALL, arms=("none", "smote"), candidate="smote", comparators=("none",), replica_arms=()
)


def fits_on(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Record the test index every classifier in the harness is scored on."""
    seen: list[list[int]] = []
    real = run_module.probabilities

    def spy(training, test, **options):
        seen.append(list(test.index))
        return real(training, test, **options)

    monkeypatch.setattr(run_module, "probabilities", spy)
    return seen


def test_no_exclusion_keeps_todays_bytes(caplog):
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    today = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        unshared = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME, exclude=None)
    assert render(unshared) == render(today)
    assert all(s.measure is not Measure.SHARED_SPLIT for s in unshared)
    assert "shared holdout=" not in caplog.text


def test_an_excluded_record_leaves_the_split_and_the_marker_fingerprints_it(monkeypatch, caplog):
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    foreign = parts.test.iloc[[3]].loc[:, list(FRAME_COLUMNS)].reset_index(drop=True)
    seen = fits_on(monkeypatch)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        scores = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME, exclude=[foreign])
    gone = parts.test.index[3]
    kept = sorted(int(i) for i in parts.test.index if i != gone)
    assert seen
    assert all(index == kept for index in seen)
    digits = sha256(repr(kept).encode()).hexdigest()[:12]
    held = [s for s in scores if s.source is Source.HELD_OUT]
    assert held[-1] == Score(
        0, HELD_OUT_ARM, Source.HELD_OUT, Measure.SHARED_SPLIT, Scope.ALL, float(int(digits, 16))
    )
    assert (
        f"shared holdout=0 records={len(parts.test)} copied=0 foreign=1 fingerprint={digits}"
        in caplog.text
    )
    unshared = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME)
    nothing = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME, exclude=[foreign.iloc[:0]])
    assert render([s for s in nothing if s.measure is not Measure.SHARED_SPLIT]) == render(unshared)


def holdout_rows(holdout, *, copies=0.0, marker=None):
    """One holdout of a decontaminating file: held-out rows, an arm's copies and its recall."""
    held = {Measure.HELD_OUT_RATIO: 0.25, Measure.DCR_P5: 0.02, Measure.DCR_MEDIAN: 0.09}
    if marker is not None:
        held[Measure.SHARED_SPLIT] = marker
    return [
        *(Score(holdout, HELD_OUT_ARM, Source.HELD_OUT, m, Scope.ALL, v) for m, v in held.items()),
        Score(holdout, "smote", Source.ADDED, Measure.TEST_COPIES, Scope.ALL, copies),
        Score(holdout, "smote", Source.AUGMENTED, Measure.RECALL, Scope.ALL, 0.5 + holdout / 100),
    ]


def scored(copied, markers):
    """Four holdouts; `copied` holds a copy, and `markers` maps a holdout to its marker value."""
    return [
        row
        for holdout in range(4)
        for row in holdout_rows(
            holdout, copies=float(holdout in copied), marker=markers.get(holdout)
        )
    ]


def test_beside_pairs_copied_holdouts_when_both_files_share_the_split():
    markers = dict.fromkeys(range(4), 123456.0)
    ours, theirs = scored({1}, markers), scored({2}, markers)
    assert pairable(ours, theirs) == Pairing(
        frozenset(range(4)), shared=4, copied=0, mismatched=0, copies_excluded=False
    )
    other = scored({2}, {**markers, 2: 654321.0, 3: 654321.0})
    assert pairable(ours, other) == Pairing(frozenset({0, 1}), shared=4, copied=1, mismatched=1)


def test_a_shared_file_never_pairs_with_an_unshared_one():
    shared = scored(set(), dict.fromkeys(range(4), 123456.0))
    for unshared in (scored(set(), {}), scored({1}, {})):
        assert pairable(shared, unshared).holdouts == frozenset()
        assert pairable(unshared, shared).holdouts == frozenset()
        assert pairable(shared, unshared, allow_copies=True).holdouts == frozenset()
