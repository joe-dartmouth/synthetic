"""`compare --beside`: a run's arm read beside another run's, holdout by holdout, no verdict."""

from __future__ import annotations

import dataclasses
import io
import logging
import math

import pytest

from bko.evaluate.__main__ import Context, main
from bko.evaluate.beside import Beside, Pairing, Skipped, pair, pairable
from bko.evaluate.criteria import corrected_interval
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, render
from bko.generate.messages import SETTINGS, Level


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


def value(holdout, series, shift=0.0):
    """A score that differs by series, by holdout and by their product, plus the arm's shift."""
    return 0.3 + 0.03 * series + 0.01 * holdout + 0.001 * holdout * series + shift


def scores(holdouts, *, shifts=None):
    """Every paired series of each arm in `shifts`, its copies at 0, and the held-out rows'."""
    shifts = {"bko": 0.0} if shifts is None else shifts
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
        for arm, shift in shifts.items():
            rows += [
                Score(holdout, arm, Source.ADDED, Measure.TEST_COPIES, Scope.ALL, 0.0),
                Score(holdout, arm, Source.REPLICA, Measure.TEST_COPIES, Scope.ALL, 0.0),
            ]
            rows += [
                Score(holdout, arm, source, measure, scope, value(holdout, i, shift))
                for i, (source, measure, scope) in enumerate(PAIRED)
            ]
    return rows


def replaced(rows, holdout, arm, source, measure, new, scope=Scope.ALL):
    found = [
        dataclasses.replace(s, value=new)
        if (s.holdout, s.arm, s.source, s.measure, s.scope)
        == (holdout, arm, source, measure, scope)
        else s
        for s in rows
    ]
    assert found != rows
    return found


def dropped(rows, holdout, arm, source, measure, scope=Scope.ALL):
    key = (holdout, arm, source, measure, scope)
    found = [s for s in rows if (s.holdout, s.arm, s.source, s.measure, s.scope) != key]
    assert len(found) == len(rows) - 1
    return found


def read(ours, theirs, *, arm="bko", against="bko", holdouts=None):
    if holdouts is None:
        holdouts = pairable(ours, theirs).holdouts
    return pair(
        ours, theirs, arm=arm, against=against, holdouts=holdouts, ratio=RATIO, confidence=0.95
    )


def readings(found):
    return [b for b in found if isinstance(b, Beside)]


def test_pairs_over_the_shared_holdouts():
    ours = scores(range(4), shifts={"bko": 0.1, "smote": 0.7})
    theirs = scores(range(4), shifts={"bko": 0.0, "smote": 0.4})
    differences = [0.03, -0.02, 0.05, 0.01]
    for holdout, difference in enumerate(differences):
        new = value(holdout, 0) + difference
        ours = replaced(ours, holdout, "bko", Source.AUGMENTED, Measure.RECALL, new)
    found = read(ours, theirs)
    assert [(b.source, b.measure, b.scope) for b in found] == list(PAIRED)
    assert all(isinstance(b, Beside) for b in found)
    recall = found[0]
    expected = corrected_interval(
        [value(h, 0) + d - value(h, 0) for h, d in enumerate(differences)],
        ratio=RATIO,
        confidence=0.95,
        family=1,
    )
    assert dataclasses.astuple(recall.interval) == pytest.approx(dataclasses.astuple(expected))
    assert (recall.differing, recall.undefined) == (4, 0)
    assert all(b.interval.mean == pytest.approx(0.1) for b in readings(found)[1:])


def test_reads_only_shared_holdouts():
    ours, theirs = scores(range(2)), scores(range(4))
    assert pairable(ours, theirs) == Pairing(frozenset({0, 1}), shared=2, copied=0, mismatched=0)
    assert all(b.interval.holdouts == 2 for b in readings(read(ours, theirs)))


def test_refuses_fewer_than_two_paired():
    ours, theirs = scores(range(3)), scores(range(3))
    with pytest.raises(EvaluationError, match="2 or more paired holdouts"):
        read(ours, theirs, holdouts=frozenset({0}))


def test_beside_arm_names_another_arm():
    ours = scores(range(3), shifts={"bko": 0.2})
    theirs = scores(range(3), shifts={"bko": 0.1, "bko-direct": 0.0})
    assert readings(read(ours, theirs, against="bko-direct"))[0].interval.mean == pytest.approx(0.2)
    assert readings(read(ours, theirs))[0].interval.mean == pytest.approx(0.1)


def test_reading_an_arm_a_file_does_not_hold_is_refused():
    ours, theirs = scores(range(3)), scores(range(3))
    with pytest.raises(EvaluationError, match="bko-direct"):
        read(ours, theirs, against="bko-direct")


@pytest.mark.parametrize("source", [Source.ADDED, Source.REPLICA])
@pytest.mark.parametrize("copied", ["ours", "theirs"])
def test_a_holdout_with_a_test_copy_of_any_arm_in_either_file_is_excluded(source, copied):
    ours = scores(range(4), shifts={"bko": 0.0, "smote": 0.3})
    theirs = scores(range(4), shifts={"bko": 0.0, "smote": 0.3})
    if copied == "ours":
        ours = replaced(ours, 2, "smote", source, Measure.TEST_COPIES, 1.0)
    else:
        theirs = replaced(theirs, 2, "smote", source, Measure.TEST_COPIES, 1.0)
    found = pairable(ours, theirs)
    assert found == Pairing(frozenset({0, 1, 3}), shared=4, copied=1, mismatched=0)
    assert found.excluded == 1


def test_a_nan_test_copies_score_counts_as_a_copy():
    ours = replaced(scores(range(3)), 1, "bko", Source.ADDED, Measure.TEST_COPIES, math.nan)
    assert pairable(ours, scores(range(3))) == Pairing(
        frozenset({0, 2}), shared=3, copied=1, mismatched=0
    )


def test_a_holdout_whose_held_out_rows_differ_is_excluded():
    ours, theirs = scores(range(4)), scores(range(4))
    theirs = replaced(theirs, 1, HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_MEDIAN, 0.5)
    assert pairable(ours, theirs) == Pairing(frozenset({0, 2, 3}), shared=4, copied=0, mismatched=1)


def test_a_held_out_score_one_file_lacks_is_a_mismatch():
    ours = dropped(scores(range(3)), 2, HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_MEDIAN)
    assert pairable(ours, scores(range(3))) == Pairing(
        frozenset({0, 1}), shared=3, copied=0, mismatched=1
    )


def test_nan_held_out_scores_in_both_files_match():
    ours, theirs = scores(range(3)), scores(range(3))
    ours = replaced(ours, 0, HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_P5, math.nan)
    theirs = replaced(theirs, 0, HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_P5, math.nan)
    assert pairable(ours, theirs).holdouts == frozenset({0, 1, 2})


def test_a_holdout_neither_file_records_held_out_rows_for_is_excluded():
    ours = [s for s in scores(range(3)) if not (s.holdout == 0 and s.source is Source.HELD_OUT)]
    theirs = [s for s in scores(range(3)) if not (s.holdout == 0 and s.source is Source.HELD_OUT)]
    assert pairable(ours, theirs) == Pairing(frozenset({1, 2}), shared=3, copied=0, mismatched=1)


def test_a_series_one_file_lacks_is_skipped_as_absent():
    ours = scores(range(3))
    theirs = [s for s in scores(range(3)) if s.source is not Source.ADDED]
    found = read(ours, theirs)
    assert found[4] == Skipped(Source.ADDED, Measure.DCR_P5, Scope.ALL, "absent", 0)
    assert len(readings(found)) == len(PAIRED) - 1


def test_a_series_defined_on_fewer_than_two_holdouts_is_skipped_as_undefined():
    ours = scores(range(4))
    for holdout in (0, 1, 2):
        ours = replaced(ours, holdout, "bko", Source.REPLICA, Measure.QUALITY, math.nan)
    found = read(ours, scores(range(4)))
    assert Skipped(Source.REPLICA, Measure.QUALITY, Scope.ALL, "undefined", 1) in found
    assert len(readings(found)) == len(PAIRED) - 1


def test_nan_holdouts_are_counted_undefined():
    ours, theirs = scores(range(4)), scores(range(4), shifts={"bko": 0.1})
    ours = replaced(ours, 0, "bko", Source.REPLICA, Measure.QUALITY, math.nan)
    theirs = replaced(theirs, 3, "bko", Source.REPLICA, Measure.QUALITY, math.nan)
    quality = next(b for b in readings(read(ours, theirs)) if b.measure is Measure.QUALITY)
    assert (quality.interval.holdouts, quality.undefined) == (2, 2)
    assert quality.interval.mean == pytest.approx(-0.1)


def test_a_holdout_absent_from_a_series_is_counted_undefined():
    ours = dropped(scores(range(4)), 1, "bko", Source.REPLICA, Measure.QUALITY)
    quality = next(
        b for b in readings(read(ours, scores(range(4)))) if b.measure is Measure.QUALITY
    )
    assert (quality.interval.holdouts, quality.undefined) == (3, 1)


def test_a_holdout_where_both_scores_agree_is_not_differing():
    ours = scores(range(4), shifts={"bko": 0.1})
    ours = replaced(ours, 2, "bko", Source.AUGMENTED, Measure.RECALL, value(2, 0))
    recall = read(ours, scores(range(4)))[0]
    assert (recall.differing, recall.interval.holdouts) == (3, 4)


CLI = dataclasses.replace(
    E1,
    holdout_seeds=(0, 1, 2, 3),
    arms=("none", "smote", "adasyn"),
    candidate="smote",
    comparators=("none",),
    replica_arms=("smote", "adasyn"),
    criteria=("primary",),
)
"""A run whose criteria hand-built scores can serve: primary, smote against none."""

OURS = {"none": 0.0, "smote": 0.3, "adasyn": 0.5}
THEIRS = {"none": 0.0, "smote": 0.1, "adasyn": 0.2}


def written(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(render(rows), encoding="utf-8")
    return path


def run(argv):
    out = io.StringIO()
    code = main(["compare", *map(str, argv)], context=Context(experiment=CLI), out=out)
    return code, out.getvalue()


def beside_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("beside ")]


def test_beside_arm_without_beside_exits_two(tmp_path):
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    assert run([ours, "--beside-arm", "smote"]) == (2, "")


def test_beside_leaves_stdout_and_logs_after_compares_lines(tmp_path, caplog):
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = written(tmp_path, "theirs.csv", scores(range(4), shifts=THEIRS))
    alone = run([ours])
    assert alone[0] == 0
    assert "primary,all," in alone[1]
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        assert run([ours, "--beside", theirs]) == alone
    messages = [record.getMessage() for record in caplog.records]
    first = next(i for i, m in enumerate(messages) if m.startswith("beside "))
    assert not any(m.startswith(("compare ", "criterion=")) for m in messages[first:])
    assert messages[first].startswith("beside scores_sha256=")
    assert " copies_excluded=true shared=" in messages[first]
    assert messages[first].endswith(
        " shared=4 paired=4 excluded=0 copied=0 mismatched=0 arm=smote against=smote "
        "run=e1 level=summary"
    )
    lines = messages[first + 1 :]
    assert len(lines) == len(PAIRED)
    assert lines[0] == (
        "beside arm=smote against=smote run=e1 level=summary source=augmented "
        "measure=recall scope=all mean=0.200000 low=0.200000 high=0.200000 holdouts=4 "
        "differing=4 undefined=0"
    )


def test_beside_lines_name_the_run_and_level(tmp_path, caplog):
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = written(tmp_path, "theirs.csv", scores(range(4), shifts=THEIRS))
    context = Context(
        experiment=CLI,
        runs={"cellrun": NamedRun(CLI, dataclasses.replace(SETTINGS, guidance=Level.CELL))},
    )
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        code = main(
            ["compare", "cellrun", str(ours), "--beside", str(theirs)],
            context=context,
            out=io.StringIO(),
        )
    assert code == 0
    lines = beside_lines(caplog)
    assert lines[0].endswith(" arm=smote against=smote run=cellrun level=cell")
    assert all(" run=cellrun level=cell " in line for line in lines[1:])


def test_beside_arm_reads_that_arm_of_the_other_file(tmp_path, caplog):
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = written(tmp_path, "theirs.csv", scores(range(4), shifts=THEIRS))
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        assert run([ours, "--beside", theirs, "--beside-arm", "adasyn"])[0] == 0
    line = beside_lines(caplog)[1]
    assert line.startswith(
        "beside arm=smote against=adasyn run=e1 level=summary source=augmented measure=recall "
    )
    assert " mean=0.100000 " in line


def test_candidate_is_the_arm_read_beside(tmp_path, caplog):
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = written(tmp_path, "theirs.csv", scores(range(4), shifts=THEIRS))
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        assert run([ours, "--candidate", "adasyn", "--beside", theirs])[0] == 0
    lines = beside_lines(caplog)
    assert lines[0].endswith(" arm=adasyn against=adasyn run=e1 level=summary")
    assert lines[1].startswith(
        "beside arm=adasyn against=adasyn run=e1 level=summary source=augmented measure=recall "
    )
    assert " mean=0.300000 " in lines[1]


def test_the_interval_reads_the_held_out_ratio_of_our_scores(tmp_path, caplog):
    rows = scores(range(4), shifts=OURS)
    differences = [0.03, -0.02, 0.05, 0.01]
    for holdout, difference in enumerate(differences):
        new = value(holdout, 0, THEIRS["smote"]) + difference
        rows = replaced(rows, holdout, "smote", Source.AUGMENTED, Measure.RECALL, new)
    ours = written(tmp_path, "ours.csv", rows)
    theirs = written(tmp_path, "theirs.csv", scores(range(4), shifts=THEIRS))
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        assert run([ours, "--beside", theirs])[0] == 0
    expected = corrected_interval(differences, ratio=RATIO, confidence=0.95, family=1)
    assert (
        f" mean={expected.mean:.6f} low={expected.low:.6f} high={expected.high:.6f} "
        in beside_lines(caplog)[1]
    )


def test_the_counts_line_splits_copied_from_mismatched_holdouts(tmp_path, caplog):
    rows = scores([*range(4), 7], shifts=THEIRS)
    rows = replaced(rows, 1, "none", Source.ADDED, Measure.TEST_COPIES, 2.0)
    rows = replaced(rows, 2, HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_P5, 0.5)
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = written(tmp_path, "theirs.csv", rows)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        assert run([ours, "--beside", theirs])[0] == 0
    lines = beside_lines(caplog)
    assert lines[0].endswith(
        " shared=4 paired=2 excluded=2 copied=1 mismatched=1 arm=smote against=smote "
        "run=e1 level=summary"
    )
    assert " holdouts=2 " in lines[1]


def test_a_skipped_series_is_logged(tmp_path, caplog):
    rows = [s for s in scores(range(4), shifts=THEIRS) if s.measure is not Measure.QUALITY]
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = written(tmp_path, "theirs.csv", rows)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        assert run([ours, "--beside", theirs])[0] == 0
    assert (
        "beside skip arm=smote against=smote run=e1 level=summary source=replica "
        "measure=quality scope=all reason=absent defined=0"
    ) in beside_lines(caplog)


def test_too_few_paired_holdouts_exit_two_before_the_table(tmp_path, caplog):
    rows = scores(range(3), shifts=THEIRS)
    for holdout in (0, 1):
        rows = replaced(rows, holdout, "adasyn", Source.REPLICA, Measure.TEST_COPIES, 1.0)
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = written(tmp_path, "theirs.csv", rows)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        assert run([ours, "--beside", theirs]) == (2, "")
    assert "2 or more paired holdouts" in caplog.text
    assert " shared=3 paired=1 excluded=2 copied=2 mismatched=0 " in caplog.text


def test_an_arm_the_other_file_lacks_exits_two_before_the_table(tmp_path):
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = written(tmp_path, "theirs.csv", scores(range(4), shifts=THEIRS))
    assert run([ours, "--beside", theirs, "--beside-arm", "tvae"]) == (2, "")


def test_a_malformed_beside_file_exits_three_before_the_table(tmp_path):
    ours = written(tmp_path, "ours.csv", scores(range(4), shifts=OURS))
    theirs = tmp_path / "theirs.csv"
    theirs.write_text("a,b\n1,2\n", encoding="utf-8")
    assert run([ours, "--beside", theirs]) == (3, "")


def fingerprinted(rows, value):
    """`rows` with a flagged-fingerprint held-out row of `value` on each of their holdouts."""
    holdouts = sorted({s.holdout for s in rows})
    fingerprint = Measure.FLAGGED_FINGERPRINT
    return [
        *rows,
        *(Score(h, HELD_OUT_ARM, Source.HELD_OUT, fingerprint, Scope.ALL, value) for h in holdouts),
    ]


def test_beside_refuses_scores_of_another_flagged_scope(tmp_path, caplog):
    ours = fingerprinted(scores(range(4), shifts=OURS), 1.0)
    theirs = fingerprinted(scores(range(4), shifts=THEIRS), 2.0)
    assert pairable(ours, theirs) == Pairing(
        frozenset(), shared=4, copied=0, mismatched=4, flagged=tuple((h, 1, 2) for h in range(4))
    )
    same = fingerprinted(scores(range(4), shifts=THEIRS), 1.0)
    assert pairable(ours, same).holdouts == frozenset(range(4))
    paths = written(tmp_path, "ours.csv", ours), written(tmp_path, "theirs.csv", theirs)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        assert run([paths[0], "--beside", paths[1]]) == (2, "")
    assert " shared=4 paired=0 excluded=4 copied=0 mismatched=4 " in caplog.text
    lines = beside_lines(caplog)
    assert lines[:4] == [f"beside flagged holdout={h} ours=1 theirs=2" for h in range(4)]
    assert lines[4].startswith("beside scores_sha256=")


def test_beside_pairs_old_files_without_a_fingerprint():
    ours, theirs = fingerprinted(scores(range(4)), 1.0), scores(range(4))
    assert pairable(ours, theirs) == Pairing(frozenset(range(4)), shared=4, copied=0, mismatched=0)
    assert pairable(theirs, ours).holdouts == frozenset(range(4))


def kept(rows):
    """`rows` with a copies-kept held-out row on each of their holdouts, as a keeping run writes."""
    holdouts = sorted({s.holdout for s in rows})
    copies = Measure.COPIES_KEPT
    return [
        *rows,
        *(Score(h, HELD_OUT_ARM, Source.HELD_OUT, copies, Scope.ALL, 1.0) for h in holdouts),
    ]


def test_pairable_allows_copies_for_a_run_that_keeps_them(tmp_path, caplog):
    ours = replaced(scores(range(4), shifts=OURS), 2, "smote", Source.ADDED, Measure.TEST_COPIES, 3)
    ours, theirs = kept(ours), kept(scores(range(4), shifts=THEIRS))
    assert pairable(ours, theirs).copied == 1
    assert pairable(ours, theirs, allow_copies=True) == Pairing(
        frozenset(range(4)), shared=4, copied=0, mismatched=0, copies_excluded=False
    )
    paths = written(tmp_path, "ours.csv", ours), written(tmp_path, "theirs.csv", theirs)
    keeps = Context(experiment=dataclasses.replace(CLI, criteria=(), decontaminate=False))
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        code = main(["compare", str(paths[0]), "--beside", str(paths[1])], context=keeps)
    assert code == 0
    assert " copies_excluded=false shared=4 paired=4 excluded=0 copied=0 mismatched=0 " in (
        caplog.text
    )


@pytest.mark.parametrize("copied", ["ours", "theirs"])
def test_a_copy_beside_a_decontaminating_file_still_excludes_its_holdout(copied):
    ours, theirs = kept(scores(range(4), shifts=OURS)), scores(range(4), shifts=THEIRS)
    if copied == "ours":
        ours = replaced(ours, 2, "smote", Source.ADDED, Measure.TEST_COPIES, 1.0)
    else:
        theirs = replaced(theirs, 2, "smote", Source.ADDED, Measure.TEST_COPIES, 1.0)
    found = pairable(ours, theirs, allow_copies=True)
    assert found == Pairing(frozenset({0, 1, 3}), shared=4, copied=1, mismatched=0)
    assert pairable(theirs, ours, allow_copies=True) == found


def test_the_copies_kept_row_is_not_a_held_out_score_to_match():
    ours, theirs = kept(scores(range(4), shifts=OURS)), scores(range(4), shifts=THEIRS)
    whole = Pairing(frozenset(range(4)), shared=4, copied=0, mismatched=0)
    assert pairable(ours, theirs) == whole
    assert pairable(ours, theirs, allow_copies=True) == whole
