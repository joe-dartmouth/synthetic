"""E3's rules on the toy frame with a removal: the ratio, the copy rule, `real` and the fingerprint.

The toy split holds 20 training records per cell but 5 in each `45-69` cell with the condition;
removing 0.9 of `>=70 female 1` leaves 2, and at `fill_to` 5 that cell alone is flagged.
"""

from __future__ import annotations

import dataclasses
import logging
from hashlib import sha256

import pandas as pd
import pytest

import bko.evaluate.run as run_module
from bko.baselines.records import Baseline, number, unit_weights
from bko.datasets.cells import cell_counts
from bko.datasets.split import split
from bko.evaluate.classifier import Model, build
from bko.evaluate.data import SYNTHEA_FRAME
from bko.evaluate.experiment import E1, E3, REAL_ARM, Removal, flagged
from bko.evaluate.run import held_out_ratio, method_arms, prepared, run, score_holdout
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Source
from tests._evaluate_fixtures import FAST_METHODS
from tests._frame_fixtures import TOY, TOY_PARTS, toy_records


REMOVED = (">=70", "female", 1)

TOY_E3 = dataclasses.replace(
    E1,
    holdout_seeds=(0, 1),
    arms=("none", REAL_ARM),
    candidate=REAL_ARM,
    comparators=("none",),
    replica_arms=(),
    fill_to=5,
    criteria=(),
    removal=Removal(REMOVED, 0.9),
    decontaminate=False,
)
"""`none` and `real` on the toy frame, with E3's removal rule and copy rule."""

ARMS = {"none": method_arms(FAST_METHODS)["none"]}
"""The runners: `real` needs none."""

LR = {Model.LR: build}

C8_COUNTS = {
    ("<65", 0, 0): 319,
    ("<65", 0, 1): 29,
    ("<65", 1, 0): 302,
    ("<65", 1, 1): 56,
    (">=65", 0, 0): 69,
    (">=65", 0, 1): 65,
    (">=65", 1, 0): 34,
    (">=65", 1, 1): 111,
}
"""The pinned population's cell sizes at 40 and over under `(65,)`, train plus test ([C8])."""


def c8_records() -> pd.DataFrame:
    """Records of the Synthea frame in [C8]'s cell sizes, every indicator 0."""
    indicators = dict.fromkeys(SYNTHEA_FRAME.features[2:], 0)
    rows = [
        {"age": (50 if band == "<65" else 70) + i % 9, "sex": sex, **indicators, "target": target}
        for (band, sex, target), size in C8_COUNTS.items()
        for i in range(size)
    ]
    columns, dtypes = list(SYNTHEA_FRAME.columns), dict(SYNTHEA_FRAME.dtypes)
    frame = pd.DataFrame(rows).loc[:, columns].astype(dtypes)
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="record_id")
    return frame


def test_every_case_stratum_keeps_two_training_rows_on_every_holdout():
    records = c8_records()
    for seed in range(100):
        parts = split(records, seed=seed, edges=SYNTHEA_FRAME.edges)
        train = prepared(parts, E3.removal, edges=SYNTHEA_FRAME.edges)[0].train
        counts = cell_counts(train, edges=SYNTHEA_FRAME.edges)
        cases = counts.loc[counts["target"] == 1, "count"]
        assert (cases >= 2).all(), (seed, counts)


def held_out(scores, measure):
    found = [
        s.value
        for s in scores
        if (s.arm, s.source, s.measure, s.scope)
        == (HELD_OUT_ARM, Source.HELD_OUT, measure, Scope.ALL)
    ]
    assert len(found) == 1
    return found[0]


def test_the_held_out_ratio_ignores_the_removal():
    scores = score_holdout(TOY_PARTS, TOY_E3, ARMS, models=LR, frame=TOY)
    thinned = prepared(TOY_PARTS, TOY_E3.removal, edges=TOY.edges)[0]
    expected = held_out_ratio(TOY_PARTS, edges=TOY.edges)
    assert held_out(scores, Measure.HELD_OUT_RATIO) == expected
    assert expected != held_out_ratio(thinned, edges=TOY.edges)


def copier(test: pd.DataFrame, rows: int):
    """An arm whose added rows are the first `rows` records of `test`."""

    def fill(train, request):
        added = number(test.iloc[:rows].loc[:, list(TOY.columns)])
        return Baseline(added=added, weights=unit_weights(train), params={})

    return fill


def fits_on(monkeypatch):
    """Record the training rows and test index of every classifier the harness fits."""
    seen = []
    real = run_module.probabilities

    def spy(training, test, **options):
        seen.append((training, list(test.index)))
        return real(training, test, **options)

    monkeypatch.setattr(run_module, "probabilities", spy)
    return seen


@pytest.mark.parametrize("decontaminate", [False, True])
def test_copies_are_counted_not_removed_when_decontaminate_is_false(monkeypatch, decontaminate):
    experiment = dataclasses.replace(
        TOY_E3, arms=("none", "copier"), candidate="copier", decontaminate=decontaminate
    )
    seen = fits_on(monkeypatch)
    arms = {**ARMS, "copier": copier(TOY_PARTS.test, 3)}
    scores = score_holdout(TOY_PARTS, experiment, arms, models=LR, frame=TOY)
    found = {(s.arm, s.measure): s.value for s in scores if s.source is Source.ADDED}
    assert found["copier", Measure.TEST_COPIES] == pytest.approx(3.0)
    assert seen
    kept = len(TOY_PARTS.test) - (3 if decontaminate else 0)
    assert {len(index) for _, index in seen} == {kept}


def test_the_real_arm_adds_back_the_removed_rows(monkeypatch):
    seen = fits_on(monkeypatch)
    scores = score_holdout(TOY_PARTS, TOY_E3, ARMS, models=LR, frame=TOY)
    removed = prepared(TOY_PARTS, TOY_E3.removal, edges=TOY.edges)[1]
    asked = TOY_E3.fill_to - 2
    assert len(removed) > asked
    rows = {(s.arm, s.measure): s.value for s in scores if s.source is Source.ADDED}
    assert rows[REAL_ARM, Measure.ROWS] == pytest.approx(len(removed))
    training = seen[1][0]
    columns = list(TOY.columns)
    augmented = training.loc[:, columns].sort_values(columns).reset_index(drop=True)
    whole = TOY_PARTS.train.loc[:, columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(augmented, whole)
    assert training["weight"].tolist() == pytest.approx([1.0] * len(training))


def test_the_real_arm_needs_no_runner():
    scores = run(toy_records(), TOY_E3, ARMS, models=LR, frame=TOY)
    real = {s.holdout for s in scores if s.arm == REAL_ARM and s.source is Source.AUGMENTED}
    assert real == set(TOY_E3.holdout_seeds)


def fingerprint(cells):
    return float(int(sha256(repr(sorted(cells)).encode()).hexdigest()[:12], 16))


def test_the_flagged_subgroup_is_the_removed_cells_subgroup():
    scores = run(toy_records(), TOY_E3, ARMS, models=LR, frame=TOY)
    for seed in TOY_E3.holdout_seeds:
        parts = split(toy_records(), seed=seed, edges=TOY.edges)
        train = prepared(parts, TOY_E3.removal, edges=TOY.edges)[0].train
        assert flagged(train, TOY_E3, edges=TOY.edges) == (REMOVED,)
        mine = [s for s in scores if s.holdout == seed]
        value = held_out(mine, Measure.FLAGGED_FINGERPRINT)
        assert value == fingerprint([REMOVED])
        assert value < 2**48
    unthinned = flagged(TOY_PARTS.train, TOY_E3, edges=TOY.edges)
    assert fingerprint(unthinned) != fingerprint([REMOVED])


@pytest.mark.parametrize("decontaminate", [False, True])
def test_a_run_that_keeps_copies_says_so_on_every_holdout(decontaminate):
    experiment = dataclasses.replace(TOY_E3, decontaminate=decontaminate)
    scores = run(toy_records(), experiment, ARMS, models=LR, frame=TOY)
    found = [s for s in scores if s.measure is Measure.COPIES_KEPT]
    expected = [] if decontaminate else [(h, 1.0) for h in experiment.holdout_seeds]
    assert [(s.holdout, s.value) for s in found] == expected
    assert all(
        (s.arm, s.source, s.scope) == (HELD_OUT_ARM, Source.HELD_OUT, Scope.ALL) for s in found
    )


def _lines(caplog, prefix):
    return [m for m in caplog.messages if m.startswith(prefix)]


def test_each_holdout_logs_its_removal_and_flagged_cells(caplog):
    with caplog.at_level(logging.INFO):
        score_holdout(TOY_PARTS, TOY_E3, ARMS, models=LR, frame=TOY)
    kept, removed = prepared(TOY_PARTS, TOY_E3.removal, edges=TOY.edges)
    whole = len(TOY_PARTS.train)
    line = (
        f"removal holdout={TOY_PARTS.seed} cell=>=70/female/1 fraction=0.9 train={whole} "
        f"removed={len(removed)} kept={len(kept.train)}"
    )
    assert _lines(caplog, "removal ") == [line]
    assert _lines(caplog, "flagged ") == [f"flagged holdout={TOY_PARTS.seed} cells=>=70/female/1"]


def test_no_removal_logs_no_removal_line(caplog):
    with caplog.at_level(logging.INFO):
        prepared(TOY_PARTS, None, edges=TOY.edges)
    assert not _lines(caplog, "removal ")


def test_the_real_arms_line_reads_attempt_1_the_gap_seed_and_no_params(caplog):
    with caplog.at_level(logging.INFO):
        score_holdout(TOY_PARTS, TOY_E3, ARMS, models=LR, frame=TOY)
    removed = prepared(TOY_PARTS, TOY_E3.removal, edges=TOY.edges)[1]
    (line,) = _lines(caplog, f"arm holdout={TOY_PARTS.seed} arm={REAL_ARM} ")
    assert line.startswith(
        f"arm holdout={TOY_PARTS.seed} arm={REAL_ARM} attempt=1 seed={TOY_PARTS.seed} "
        f"added={len(removed)} sha256="
    )
    assert line.endswith(" params={}")


@pytest.mark.parametrize("decontaminate", [False, True])
def test_the_test_line_says_whether_copies_are_kept(caplog, decontaminate):
    experiment = dataclasses.replace(
        TOY_E3, arms=("none", "copier"), candidate="copier", decontaminate=decontaminate
    )
    arms = {**ARMS, "copier": copier(TOY_PARTS.test, 3)}
    with caplog.at_level(logging.INFO):
        score_holdout(TOY_PARTS, experiment, arms, models=LR, frame=TOY)
    kept = "false" if decontaminate else "true"
    assert _lines(caplog, "test ") == [
        f"test holdout={TOY_PARTS.seed} records={len(TOY_PARTS.test)} copied=3 kept={kept}"
    ]
