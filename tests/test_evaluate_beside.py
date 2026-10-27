"""`compare --beside`: a run's arm read beside another run's, holdout by holdout, no verdict."""

from __future__ import annotations

import dataclasses
import io
import logging
import math

import pytest

from bko.evaluate.__main__ import Context, main
from bko.evaluate.beside import pair, pairable
from bko.evaluate.criteria import corrected_interval
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, render


RATIO = 0.25
UTILITY = (
    (Measure.RECALL, Scope.ALL),
    (Measure.RECALL, Scope.FLAGGED),
    (Measure.AUC_PR, Scope.ALL),
    (Measure.AUC_PR, Scope.FLAGGED),
)
FIDELITY = (Measure.QUALITY, Measure.COLUMN_SHAPES, Measure.COLUMN_PAIR_TRENDS)
PAIRED = (
    *((Source.AUGMENTED, measure, scope) for measure, scope in UTILITY),
    (Source.ADDED, Measure.DCR_P5, Scope.ALL),
    *((Source.REPLICA, measure, scope) for measure, scope in UTILITY),
    *((Source.REPLICA, measure, Scope.ALL) for measure in FIDELITY),
)
"""Every series `pair` reads, in the order it reads them."""


def scores(holdouts, *, arms=("bko",), shift=0.0):
    """Every paired series of each arm, its copies at 0, and the held-out rows' scores."""
    rows = []
    for holdout in holdouts:
        held = {
            Measure.HELD_OUT_RATIO: RATIO,
            Measure.DCR_P5: 0.02 + holdout / 1000,
            Measure.DCR_MEDIAN: 0.09 + holdout / 1000,
        }
        rows += [
            Score(holdout, HELD_OUT_ARM, Source.HELD_OUT, m, Scope.ALL, v) for m, v in held.items()
        ]
        for arm in arms:
            rows += [
                Score(holdout, arm, Source.ADDED, Measure.TEST_COPIES, Scope.ALL, 0.0),
                Score(holdout, arm, Source.REPLICA, Measure.TEST_COPIES, Scope.ALL, 0.0),
            ]
            rows += [
                Score(holdout, arm, source, measure, scope, 0.5 + shift + holdout / 100)
                for source, measure, scope in PAIRED
            ]
    return rows


def replaced(rows, holdout, arm, source, measure, value, scope=Scope.ALL):
    found = [
        dataclasses.replace(s, value=value)
        if (s.holdout, s.arm, s.source, s.measure, s.scope)
        == (holdout, arm, source, measure, scope)
        else s
        for s in rows
    ]
    assert found != rows
    return found


def read(ours, theirs, *, arm="bko", against="bko", holdouts=None):
    if holdouts is None:
        holdouts, _ = pairable(ours, theirs)
    return pair(
        ours, theirs, arm=arm, against=against, holdouts=holdouts, ratio=RATIO, confidence=0.95
    )


def test_pairs_over_the_shared_holdouts():
    ours = scores(range(4), shift=0.1)
    theirs = scores(range(4))
    differences = [0.03, -0.02, 0.05, 0.01]
    for holdout, difference in enumerate(differences):
        value = 0.5 + holdout / 100 + difference
        ours = replaced(ours, holdout, "bko", Source.AUGMENTED, Measure.RECALL, value)
    found = read(ours, theirs)
    assert [(b.source, b.measure, b.scope) for b in found] == list(PAIRED)
    recall = found[0]
    expected = corrected_interval(
        [0.5 + h / 100 + d - (0.5 + h / 100) for h, d in enumerate(differences)],
        ratio=RATIO,
        confidence=0.95,
        family=1,
    )
    assert (recall.interval.mean, recall.interval.low, recall.interval.high) == pytest.approx(
        (expected.mean, expected.low, expected.high)
    )
    assert recall.interval.holdouts == 4
    assert (recall.differing, recall.undefined) == (4, 0)
    assert found[1].interval.mean == pytest.approx(0.1)


def test_reads_only_shared_holdouts():
    ours, theirs = scores(range(2)), scores(range(4))
    holdouts, excluded = pairable(ours, theirs)
    assert (holdouts, excluded) == (frozenset({0, 1}), 0)
    assert all(b.interval.holdouts == 2 for b in read(ours, theirs, holdouts=holdouts))


def test_refuses_fewer_than_two_paired():
    ours, theirs = scores(range(3)), scores(range(3))
    with pytest.raises(EvaluationError, match="2 or more paired holdouts"):
        read(ours, theirs, holdouts=frozenset({0}))


def test_beside_arm_names_another_arm():
    ours = scores(range(3), arms=("bko",), shift=0.2)
    theirs = scores(range(3), arms=("bko", "bko-direct"))
    found = read(ours, theirs, against="bko-direct")
    assert found[0].interval.mean == pytest.approx(0.2)


def test_reading_an_arm_a_file_does_not_hold_is_refused():
    ours, theirs = scores(range(3)), scores(range(3))
    with pytest.raises(EvaluationError, match="bko-direct"):
        read(ours, theirs, against="bko-direct")


@pytest.mark.parametrize("source", [Source.ADDED, Source.REPLICA])
@pytest.mark.parametrize("copied", ["ours", "theirs"])
def test_a_holdout_with_a_test_copy_in_either_file_is_excluded(source, copied):
    ours, theirs = scores(range(4), arms=("bko", "smote")), scores(range(4))
    if copied == "ours":
        ours = replaced(ours, 2, "smote", source, Measure.TEST_COPIES, 1.0)
    else:
        theirs = replaced(theirs, 2, "bko", source, Measure.TEST_COPIES, 1.0)
    assert pairable(ours, theirs) == (frozenset({0, 1, 3}), 1)


def test_refuses_a_holdout_whose_held_out_rows_differ():
    ours, theirs = scores(range(4)), scores(range(4))
    theirs = replaced(theirs, 1, HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_MEDIAN, 0.5)
    assert pairable(ours, theirs) == (frozenset({0, 2, 3}), 1)


def test_a_holdout_neither_file_records_held_out_rows_for_is_excluded():
    ours = [s for s in scores(range(3)) if not (s.holdout == 0 and s.source is Source.HELD_OUT)]
    theirs = [s for s in scores(range(3)) if not (s.holdout == 0 and s.source is Source.HELD_OUT)]
    assert pairable(ours, theirs) == (frozenset({1, 2}), 1)


def test_pairs_only_keys_both_files_hold():
    ours = scores(range(3))
    theirs = [s for s in scores(range(3)) if s.source is not Source.ADDED]
    found = read(ours, theirs)
    assert (Source.ADDED, Measure.DCR_P5, Scope.ALL) not in {
        (b.source, b.measure, b.scope) for b in found
    }
    assert Measure.DCR_MEDIAN not in {b.measure for b in read(ours, scores(range(3)))}


def test_nan_holdouts_are_counted_undefined():
    ours, theirs = scores(range(4)), scores(range(4), shift=0.1)
    ours = replaced(ours, 0, "bko", Source.REPLICA, Measure.QUALITY, math.nan)
    theirs = replaced(theirs, 3, "bko", Source.REPLICA, Measure.QUALITY, math.nan)
    quality = next(b for b in read(ours, theirs) if b.measure is Measure.QUALITY)
    assert (quality.interval.holdouts, quality.undefined) == (2, 2)
    assert quality.interval.mean == pytest.approx(-0.1)


LIGHT = dataclasses.replace(
    E1,
    holdout_seeds=(0, 1, 2),
    arms=("none", "smote"),
    candidate="smote",
    comparators=("none",),
    replica_arms=("smote",),
    criteria=(),
)
"""A run whose `compare` reads no criterion, so hand-built scores serve it."""


def written(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(render(rows), encoding="utf-8")
    return path


def run(argv):
    out = io.StringIO()
    code = main(["compare", *map(str, argv)], context=Context(experiment=LIGHT), out=out)
    return code, out.getvalue()


def test_beside_arm_without_beside_exits_two(tmp_path):
    ours = written(tmp_path, "ours.csv", scores(range(3), arms=("none", "smote")))
    assert run([ours, "--beside-arm", "smote"]) == (2, "")


def test_beside_leaves_stdout_and_logs_after_compares_lines(tmp_path, caplog):
    ours = written(tmp_path, "ours.csv", scores(range(3), arms=("none", "smote"), shift=0.1))
    theirs = written(tmp_path, "theirs.csv", scores(range(3), arms=("none", "smote")))
    alone = run([ours])
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        beside = run([ours, "--beside", theirs])
    assert beside == alone
    assert alone[0] == 0
    messages = [record.getMessage() for record in caplog.records]
    first = next(i for i, m in enumerate(messages) if m.startswith("beside "))
    assert not any(m.startswith("compare ") for m in messages[first:])
    assert messages[first].startswith("beside scores_sha256=")
    assert messages[first].endswith(" shared=3 paired=3 excluded=0 arm=smote against=smote")
    lines = messages[first + 1 :]
    assert len(lines) == len(PAIRED)
    assert lines[0] == (
        "beside arm=smote against=smote source=augmented measure=recall scope=all "
        "mean=0.100000 low=0.100000 high=0.100000 holdouts=3 differing=3 undefined=0"
    )
