"""The membership attack, its scopes, the reading rule and `python -m bko.evaluate privacy`."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import math
import operator
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from bko.datasets.cells import cell_keys
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import membership
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, REAL_ARM, Experiment
from bko.evaluate.membership import (
    GROUPS,
    SCOPES,
    Attack,
    arm_measures,
    attack,
    exposure,
    mi,
    reference_of,
    scopes,
)
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.prepare import Prepared
from bko.evaluate.privacy_rule import Reading, readings
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, parse
from tests._datasets_fixtures import archive, cleveland_line, pin
from tests._gaps_fixtures import ladder_releases
from tests._generate_fixtures import SIZES
from tests._ontology_fixtures import RoutingOpener
from tests.test_evaluate_privacy import random_records
from tests.test_evaluate_rows import BASELINE_ARMS, REAL
from tests.test_evaluate_shared import FORMAT, edit_log, rescorer


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CELL = ("<55", "female", 1)
"""The removed cell of `PRIVATE` and of the planted removal."""

PRIVATE = dataclasses.replace(REAL, arms=("none", REAL_ARM, "smote"), comparators=("none", "smote"))
"""`REAL` with a row-adding baseline beside `real`, and `none`, which adds no rows."""


def _attack(
    members: pd.DataFrame, nonmembers: pd.DataFrame, added: pd.DataFrame, train: pd.DataFrame
) -> Attack:
    """The attack on `members` and `nonmembers` through their distances to `added`."""
    return attack(
        exposure(members, added, train=train, frame=UCI_FRAME),
        exposure(nonmembers, added, train=train, frame=UCI_FRAME),
    )


def test_copying_every_member_is_caught_at_every_member():
    members = random_records(UCI_FRAME, 300, seed=11)
    nonmembers = random_records(UCI_FRAME, 300, seed=12)
    found = _attack(members, nonmembers, members.copy(), members)
    assert (found.auc, found.tpr[0.01]) == (1.0, 1.0)


def test_copying_half_the_members_reads_three_quarters():
    members = random_records(UCI_FRAME, 300, seed=11)
    nonmembers = random_records(UCI_FRAME, 300, seed=12)
    found = _attack(members, nonmembers, members.iloc[:150], members)
    assert found.auc == pytest.approx(0.75, abs=0.05)


def test_independent_draws_read_chance():
    members = random_records(UCI_FRAME, 500, seed=21)
    nonmembers = random_records(UCI_FRAME, 500, seed=22)
    found = _attack(members, nonmembers, random_records(UCI_FRAME, 500, seed=23), members)
    assert abs(found.auc - 0.5) < 0.08
    assert found.tpr[0.01] <= 0.05


def _placed(records: pd.DataFrame, *, inside: bool) -> pd.DataFrame:
    """`records` moved into `CELL`, or out of it into men."""
    if inside:
        return records.assign(age=45, sex=0, target=1)
    return records.assign(sex=1)


def test_a_planted_removal_biases_all_records_but_not_outside_the_cell():
    members = _placed(random_records(UCI_FRAME, 500, seed=31), inside=False)
    outside = _placed(random_records(UCI_FRAME, 500, seed=32), inside=False)
    inside = _placed(random_records(UCI_FRAME, 100, seed=33), inside=True)
    nonmembers = pd.concat([outside, inside], ignore_index=True)
    fill = _placed(random_records(UCI_FRAME, 500, seed=34), inside=False)
    added = pd.concat([fill, inside], ignore_index=True)
    member_d = exposure(members, added, train=members, frame=UCI_FRAME)
    nonmember_d = exposure(nonmembers, added, train=members, frame=UCI_FRAME)
    masks = scopes(members, CELL, edges=UCI_FRAME.edges)
    others = scopes(nonmembers, CELL, edges=UCI_FRAME.edges)
    scoped = attack(member_d[masks["outside"]], nonmember_d[others["outside"]])
    assert abs(scoped.auc - 0.5) < 0.08
    assert attack(member_d, nonmember_d).auc < 0.5 - 0.05


def test_scopes_split_records_by_the_cell_and_the_label():
    records = random_records(UCI_FRAME, 400, seed=41)
    keys = cell_keys(records, edges=UCI_FRAME.edges)
    inside = (keys["age_band"] == "<55") & (keys["sex"] == "female") & (keys["target"] == 1)
    masks = scopes(records, CELL, edges=UCI_FRAME.edges)
    assert set(masks) == set(SCOPES)
    assert np.array_equal(masks["cell"], inside.to_numpy())
    assert np.array_equal(masks["outside"], ~inside.to_numpy())
    assert np.array_equal(masks["label"], (~inside & (keys["target"] == 1)).to_numpy())


def test_no_rows_read_undefined():
    members = random_records(UCI_FRAME, 20, seed=51)
    distances = exposure(members, members.iloc[:0], train=members, frame=UCI_FRAME)
    assert distances.shape == (20,)
    assert np.isnan(distances).all()
    for found in (attack(distances, distances), attack(np.zeros(3), np.zeros(0))):
        assert math.isnan(found.auc)
        assert all(math.isnan(value) for value in found.tpr.values())


def test_planted_ties_read_their_midrank_auc():
    # Members' scores 0, -1, -1 against non-members' -1, -2: 2 + 1.5 + 1.5 of 6 pairs.
    found = attack(np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0]))
    assert found.auc == pytest.approx(5 / 6)


def test_reading_the_distance_as_the_score_fails_the_copier():
    members = random_records(UCI_FRAME, 300, seed=11)
    nonmembers = random_records(UCI_FRAME, 300, seed=12)
    member_d = exposure(members, members, train=members, frame=UCI_FRAME)
    nonmember_d = exposure(nonmembers, members, train=members, frame=UCI_FRAME)
    flipped = attack(-member_d, -nonmember_d)
    assert (flipped.auc, flipped.tpr[0.01]) == (0.0, 0.0)


def test_tpr_reads_the_roc_curve_by_linear_interpolation():
    # One member above every non-member, one tied with the top ten of 1,000 non-members: the
    # curve runs straight from (0, 0.5) to (0.01, 1.0) across the tie.
    nonmember_d = np.array([0.2] * 10 + [*np.linspace(0.3, 0.9, 990)])
    found = attack(np.array([0.1, 0.2]), nonmember_d)
    assert found.tpr[0.01] == pytest.approx(1.0)
    assert found.tpr[0.001] == pytest.approx(0.55)


def test_a_rate_below_one_over_the_non_members_is_nan():
    found = attack(np.zeros(5), np.zeros(100))
    assert found.auc == pytest.approx(0.5)
    assert found.tpr[0.01] == pytest.approx(0.01)
    assert math.isnan(found.tpr[0.001])


def test_a_partly_missing_distance_or_no_member_reads_nan_with_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for found in (
            attack(np.array([0.1, math.nan]), np.array([0.2, 0.3])),
            attack(np.zeros(0), np.zeros(3)),
        ):
            assert math.isnan(found.auc)
            assert all(math.isnan(value) for value in found.tpr.values())


def _split(seed: int, *, test_outside: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test records: men outside `CELL`, a few women in it on each side."""
    train = pd.concat(
        [
            _placed(random_records(UCI_FRAME, 200, seed=seed), inside=False),
            _placed(random_records(UCI_FRAME, 5, seed=seed + 1), inside=True),
        ],
        ignore_index=True,
    )
    test = pd.concat(
        [
            _placed(random_records(UCI_FRAME, test_outside, seed=seed + 2), inside=False),
            _placed(random_records(UCI_FRAME, 5, seed=seed + 3), inside=True),
        ],
        ignore_index=True,
    )
    return train, test


def test_a_copied_test_record_stays_a_non_member():
    train, test = _split(61)
    added = pd.concat([random_records(UCI_FRAME, 50, seed=65), test.iloc[[0]]], ignore_index=True)
    found = arm_measures(
        added, train=train, test=test, cell=CELL, frame=UCI_FRAME, holdout=0, arm="arm"
    )
    inner, outer = (scopes(x, CELL, edges=UCI_FRAME.edges)["outside"] for x in (train, test))
    member_d = exposure(train, added, train=train, frame=UCI_FRAME)[inner]
    nonmember_d = exposure(test, added, train=train, frame=UCI_FRAME)[outer]
    assert found[mi("auc", "outside")] == attack(member_d, nonmember_d).auc
    assert found[mi("auc", "outside")] != attack(member_d, nonmember_d[1:]).auc


def _measured(added: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return arm_measures(
        added, train=train, test=test, cell=CELL, frame=UCI_FRAME, holdout=7, arm="arm"
    )


def test_an_arm_measures_every_privacy_measure_and_none_is_read_as_zero():
    train, test = _split(71)
    found = _measured(test.iloc[:30], train, test)
    assert list(found) == list(membership.ADDED_MEASURES)
    rates = {m for m in found if m.value.startswith("mi-tpr")}
    assert all(math.isfinite(v) for m, v in found.items() if m not in rates)
    empty = _measured(test.iloc[:0], train, test)
    assert list(empty) == list(membership.ADDED_MEASURES)
    assert all(math.isnan(value) for value in empty.values())


def test_each_rate_is_the_attacks_at_its_own_false_positive_rate():
    train, test = _split(75, test_outside=1_000)
    added = pd.concat([train.iloc[:60], random_records(UCI_FRAME, 60, seed=79)], ignore_index=True)
    found = _measured(added, train, test)
    inner, outer = (scopes(x, CELL, edges=UCI_FRAME.edges)["outside"] for x in (train, test))
    member_d = exposure(train, added, train=train, frame=UCI_FRAME)[inner]
    nonmember_d = exposure(test, added, train=train, frame=UCI_FRAME)[outer]
    read = attack(member_d, nonmember_d)
    assert read.tpr[0.01] != read.tpr[0.001]
    assert found[mi("tpr01", "outside")] == read.tpr[0.01]
    assert found[mi("tpr001", "outside")] == read.tpr[0.001]
    assert found[mi("auc", "outside")] > 0.5


def test_the_cell_scope_is_undefined_with_no_member_in_the_cell(caplog):
    train, test = _split(81)
    outside = train[scopes(train, CELL, edges=UCI_FRAME.edges)["outside"]]
    caplog.set_level(logging.INFO)
    found = _measured(test.iloc[:30], outside, test)
    assert math.isnan(found[mi("auc", "cell")])
    assert math.isnan(found[mi("tpr01", "cell")])
    assert math.isfinite(found[mi("auc", "outside")])
    assert math.isfinite(found[mi("auc", "label")])
    (record,) = [r for r in caplog.records if "undefined" in r.getMessage()]
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        "privacy holdout=7 arm=arm scope=cell undefined members=0 nonmembers=5 nan_distances=0"
    )


def test_an_attack_undefined_with_members_logs_a_warning(caplog):
    train, test = _split(83)
    inside = test[scopes(test, CELL, edges=UCI_FRAME.edges)["cell"]]
    _measured(test.iloc[:30], train, test.drop(inside.index))
    (record,) = [r for r in caplog.records if "undefined" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "scope=cell undefined members=5 nonmembers=0" in record.getMessage()


# The rule.

HOLDOUTS = 10
RATIO = 0.1
WOBBLE = [0.03 * (1 if h % 2 else -1) for h in range(HOLDOUTS)]
"""A contrast's spread about its mean: at k = 2 a 0.05 mean is flagged, at k = 40 it is not."""


def _series(arm: str, measure: Measure, values: list[float]) -> list[Score]:
    return [
        Score(h, arm, Source.ADDED, measure, Scope.ALL, value) for h, value in enumerate(values)
    ]


def _ratio() -> list[Score]:
    return [
        Score(h, HELD_OUT_ARM, Source.HELD_OUT, Measure.HELD_OUT_RATIO, Scope.ALL, RATIO)
        for h in range(HOLDOUTS)
    ]


def _scored(arms: dict[str, list[float]], measures: list[Measure]) -> list[Score]:
    """`real` at 0.5 on every measure, and each arm at `real` plus its contrasts."""
    scores = _ratio()
    for measure in measures:
        scores += _series(REAL_ARM, measure, [0.5] * HOLDOUTS)
        for arm, contrasts in arms.items():
            scores += _series(arm, measure, [0.5 + c for c in contrasts])
    return scores


MI_OUTSIDE = [mi("auc", "outside"), mi("tpr01", "outside")]


def _reading(found: list[Reading], arm: str, measure: Measure) -> Reading:
    (only,) = [r for r in found if r.arm == arm and r.measure is measure]
    return only


def test_a_leaking_arm_is_flagged_and_a_null_arm_is_not():
    arms = {"leak": [0.1 + w / 3 for w in WOBBLE], "null": WOBBLE}
    found = readings(
        _scored(arms, MI_OUTSIDE), GROUPS, arms=["leak", "null", REAL_ARM], reference=REAL_ARM
    )
    leak, null = (_reading(found, arm, mi("auc", "outside")) for arm in ("leak", "null"))
    assert leak.leaks_more is True
    assert null.leaks_more is False
    assert (leak.family, leak.against, leak.scope) == (4, REAL_ARM, "outside")
    assert leak.contrast == pytest.approx(0.1)
    assert leak.mean == pytest.approx(0.6)
    assert not any(r.arm == REAL_ARM for r in found)


def test_the_family_size_changes_the_outcome():
    alone = {"arm": [0.05 + w for w in WOBBLE]}
    crowded = {**alone, **{f"null-{n}": WOBBLE for n in range(19)}}
    for arms, family, flagged in ((alone, 2, True), (crowded, 40, False)):
        scores = _scored(arms, MI_OUTSIDE)
        found = readings(scores, GROUPS, arms=[*arms, REAL_ARM], reference=REAL_ARM)
        reading = _reading(found, "arm", mi("auc", "outside"))
        assert (reading.family, reading.leaks_more) == (family, flagged)


def test_an_absent_or_undefined_measure_leaves_the_family():
    arms = {"arm": [0.05 + w for w in WOBBLE]}
    alone = readings(
        _scored(arms, MI_OUTSIDE[:1]), GROUPS, arms=["arm", REAL_ARM], reference=REAL_ARM
    )
    assert [(r.measure, r.family) for r in alone] == [(MI_OUTSIDE[0], 1)]
    scores = [
        s for s in _scored(arms, MI_OUTSIDE) if not (s.arm == "arm" and s.measure is MI_OUTSIDE[1])
    ]
    scores += _series("arm", MI_OUTSIDE[1], [math.nan] * HOLDOUTS)
    found = readings(scores, GROUPS, arms=["arm", REAL_ARM], reference=REAL_ARM)
    auc, rate = (_reading(found, "arm", m) for m in MI_OUTSIDE)
    assert (auc.family, rate.family, rate.leaks_more) == (1, 1, None)


def test_nan_holdouts_are_counted_and_never_read_as_zero():
    contrasts = [0.1 + w / 3 for w in WOBBLE]
    gappy = [math.nan, math.nan, math.nan, *contrasts[3:]]
    arms = {"gappy": gappy, "empty": [math.nan] * HOLDOUTS}
    found = readings(_scored(arms, MI_OUTSIDE), GROUPS, arms=[*arms, REAL_ARM], reference=REAL_ARM)
    reading = _reading(found, "gappy", mi("auc", "outside"))
    assert (reading.holdouts, reading.undefined) == (HOLDOUTS, 3)
    assert reading.contrast == pytest.approx(float(np.mean(contrasts[3:])))
    assert reading.mean == pytest.approx(0.5 + float(np.mean(contrasts[3:])))
    assert reading.family == 2
    empty = _reading(found, "empty", mi("auc", "outside"))
    assert (empty.undefined, empty.leaks_more) == (HOLDOUTS, None)
    assert math.isnan(empty.contrast)
    assert math.isnan(empty.low)
    assert math.isnan(empty.mean)


def test_two_finite_contrasts_read_an_interval_and_one_does_not():
    two = [0.1, 0.2] + [math.nan] * (HOLDOUTS - 2)
    one = [0.1] + [math.nan] * (HOLDOUTS - 1)
    found = readings(
        _scored({"two": two, "one": one}, MI_OUTSIDE[:1]),
        GROUPS,
        arms=["two", "one", REAL_ARM],
        reference=REAL_ARM,
    )
    pair, single = (_reading(found, arm, MI_OUTSIDE[0]) for arm in ("two", "one"))
    assert math.isfinite(pair.low)
    assert math.isfinite(pair.high)
    assert (pair.family, pair.undefined) == (1, HOLDOUTS - 2)
    assert (single.leaks_more, single.undefined) == (None, HOLDOUTS - 1)


def test_the_interval_is_the_corrected_t_interval_at_family_wise_95_percent():
    contrasts = [0.05 + w for w in WOBBLE]
    found = readings(
        _scored({"arm": contrasts}, MI_OUTSIDE), GROUPS, arms=["arm", REAL_ARM], reference=REAL_ARM
    )
    reading = _reading(found, "arm", MI_OUTSIDE[0])
    values = np.array(contrasts)
    scale = math.sqrt((1 / HOLDOUTS + RATIO) * float(values.var(ddof=1)))
    half = float(stats.t.ppf(1 - 0.05 / 2 / 2, HOLDOUTS - 1)) * scale
    assert reading.family == 2
    assert reading.low == pytest.approx(values.mean() - half)
    assert reading.high == pytest.approx(values.mean() + half)


def test_the_largest_held_out_ratio_is_the_one_read():
    contrasts = [0.05 + w for w in WOBBLE]
    scores = [s for s in _scored({"arm": contrasts}, MI_OUTSIDE) if s.source is Source.ADDED]
    scores += [
        Score(h, HELD_OUT_ARM, Source.HELD_OUT, Measure.HELD_OUT_RATIO, Scope.ALL, ratio)
        for h, ratio in enumerate([RATIO / 2] * (HOLDOUTS - 1) + [RATIO])
    ]
    varied = readings(scores, GROUPS, arms=["arm", REAL_ARM], reference=REAL_ARM)
    fixed = readings(
        _scored({"arm": contrasts}, MI_OUTSIDE), GROUPS, arms=["arm", REAL_ARM], reference=REAL_ARM
    )
    assert _reading(varied, "arm", MI_OUTSIDE[0]).low == _reading(fixed, "arm", MI_OUTSIDE[0]).low


def test_the_holdouts_are_the_held_out_rows_holdouts():
    scores = _ratio()
    for arm in (REAL_ARM, "arm"):
        scores += _series(arm, MI_OUTSIDE[0], [0.5] * (HOLDOUTS - 2))
    (reading,) = readings(scores, GROUPS, arms=["arm", REAL_ARM], reference=REAL_ARM)
    assert (reading.holdouts, reading.undefined) == (HOLDOUTS, 2)


def test_a_distance_nearer_than_the_reference_leaks_more():
    measures = [Measure.DCR_P5, Measure.DCR_MEDIAN, Measure.NNDR_P5, Measure.NNDR_MEDIAN]
    arms = {"near": [-0.1 + w / 3 for w in WOBBLE], "far": [0.1 + w / 3 for w in WOBBLE]}
    found = readings(_scored(arms, measures), GROUPS, arms=[*arms, REAL_ARM], reference=REAL_ARM)
    near, far = (_reading(found, arm, Measure.DCR_P5) for arm in ("near", "far"))
    assert near.leaks_more is True
    assert far.leaks_more is False
    assert (near.family, near.scope) == (8, "all")


# The command.


def _distinct() -> bytes:
    """`UCI_E1`'s cells, each record's blood pressure, cholesterol, heart rate and ST depression
    drawn at random."""
    rng = np.random.default_rng(0)
    lines = [
        cleveland_line(
            age=age,
            sex=sex,
            num=num,
            cp="4.0" if (age, sex) == (40, 0) else "1.0",
            trestbps=f"{rng.integers(100, 180)}.0",
            chol=f"{rng.integers(150, 350)}.0",
            thalach=f"{rng.integers(100, 190)}.0",
            oldpeak=f"{rng.integers(0, 40) / 10}",
        )
        for (age, sex, num), size in {**SIZES, (40, 0, 1): 5}.items()
        for _ in range(size)
    ]
    return archive("\n".join(lines) + "\n")


DISTINCT = _distinct()
DISTINCT_PIN = pin(DISTINCT)


def _no_send() -> object:
    msg = "the privacy reading made an API call"
    raise AssertionError(msg)


def _context(
    tmp_path: Path,
    *,
    experiment: Experiment = PRIVATE,
    runs: dict | None = None,
    runs_dir: Path | None = None,
) -> Context:
    """A context over `DISTINCT` with `experiment`, whose any call fails, its runs root
    `runs_dir` or else `tmp_path`."""
    ctx = rescorer(experiment=experiment, arms=BASELINE_ARMS)
    _, served = ladder_releases()
    opener = RoutingOpener({**served, DISTINCT_PIN.url: DISTINCT})
    root = tmp_path if runs_dir is None else runs_dir
    ctx = dataclasses.replace(ctx, open_url=opener, uci=DISTINCT_PIN, send=_no_send, runs_dir=root)
    return ctx if runs is None else dataclasses.replace(ctx, runs=runs)


@pytest.fixture
def reference(monkeypatch):
    """`e1`, which the test context's experiment stands in for, read against `real`."""
    families = {"e1": REAL_ARM, "whole": REAL_ARM, "replicated": REAL_ARM}
    monkeypatch.setattr(membership, "REFERENCE", {**membership.REFERENCE, **families})


def _stored(
    tmp_path: Path,
    caplog,
    *,
    name: str = "run",
    experiment: Experiment = PRIVATE,
    root: Path | None = None,
) -> tuple[Path, Path]:
    """`experiment`'s folder `name` under `root`, default `tmp_path`, scored on `DISTINCT`, and
    its store, in `tmp_path`."""
    ctx = _context(tmp_path, experiment=experiment)
    folder = (tmp_path if root is None else root) / name
    folder.mkdir(parents=True)
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    caplog.clear()
    assert main(["score", "--data-dir", str(tmp_path / "data")], context=ctx, out=out) == 0
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    store = tmp_path / f"{name}.jsonl"
    argv = ["rows", "--folder", str(folder), "--rows", str(store)]
    argv += ["--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    caplog.clear()
    return folder, store


def _privacy(
    tmp_path: Path,
    *stores: Path,
    run: str = "e1",
    runs: dict | None = None,
    runs_dir: Path | None = None,
    flags: tuple[str, ...] = (),
) -> tuple[int, str]:
    argv = ["privacy", run, "--data-dir", str(tmp_path / "data"), *flags]
    for store in stores:
        argv += ["--rows", str(store)]
    out = io.StringIO()
    code = main(argv, context=_context(tmp_path, runs=runs, runs_dir=runs_dir), out=out)
    return code, out.getvalue()


@pytest.mark.usefixtures("reference")
def test_the_privacy_command_reads_a_store_with_no_api_call(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog)
    caplog.set_level(logging.INFO)
    code, text = _privacy(tmp_path, store)
    assert code == 0
    ours = {(s.holdout, s.arm, s.source, s.measure): s.value for s in parse(text)}
    theirs = parse((folder / "scores.csv").read_text(encoding="utf-8"))
    compared = 0
    for score in theirs:
        if score.measure in {Measure.DCR_P5, Measure.DCR_MEDIAN, Measure.HELD_OUT_RATIO}:
            key = (score.holdout, score.arm, score.source, score.measure)
            assert ours[key] == score.value
            compared += 1
    # Per holdout: the held-out ratio and DCR, then `real`'s and `smote`'s DCR; `none` adds none.
    assert compared == 2 * (3 + 2 + 2)
    for h in (0, 1):
        assert math.isnan(ours[h, "none", Source.ADDED, mi("auc", "outside")])
        assert math.isfinite(ours[h, "smote", Source.ADDED, mi("auc", "cell")])
        for arm, source in ((HELD_OUT_ARM, Source.HELD_OUT), ("smote", Source.ADDED)):
            assert ours[h, arm, source, Measure.DCR_MEDIAN] > 0
            assert 0 < ours[h, arm, source, Measure.NNDR_MEDIAN] < 1
    assert text.splitlines()[0] == "holdout,arm,source,measure,scope,value"
    sha = hashlib.sha256(text.encode()).hexdigest()
    rows = text.count(chr(10)) - 1
    assert re.search(
        rf"privacy rows={rows} sha256={sha} readings=\d+ undefined=\d+ seconds=\S+", caplog.text
    )
    # A test split under 100 records leaves TPR at 0.01 undefined, so the family is AUC's alone.
    line = re.search(
        r"privacy arm=smote scope=outside measure=mi-auc-outside mean=\S+ against=real"
        r" contrast=\S+ low=\S+ high=\S+ family=(\d+) holdouts=2 undefined=0"
        r" leaks_more=(true|false)",
        caplog.text,
    )
    assert line is not None
    assert line[1] == "1"
    assert "measure=mi-tpr01-outside" in caplog.text
    assert "undefined=2 leaks_more=undefined" in caplog.text
    assert "privacy arm=none scope=outside measure=mi-auc-outside" in caplog.text
    header = f"privacy run=e1 stores=1 holdouts=2 reference=real cell=<55/female/1 root={tmp_path}"
    assert header in caplog.messages
    assert f"privacy store={store} run=e1 folder=run log_sha256=" in caplog.text
    assert f"holdouts=0,1 sha256={hashlib.sha256(store.read_bytes()).hexdigest()}" in caplog.text
    assert "privacy read stores=1 holdouts=2" in caplog.text
    assert re.search(r"privacy holdout=1 done=2/2 seconds=\S+ elapsed=\S+", caplog.text)
    (added,) = [r for r in caplog.records if r.getMessage() == "privacy holdout=0 arm=none added=0"]
    assert added.levelno == logging.WARNING


@pytest.mark.usefixtures("reference")
def test_privacy_reads_stores_from_a_runs_dir_outside_the_codes_checkout(tmp_path, caplog):
    elsewhere = tmp_path / "elsewhere" / "runs"
    _, store = _stored(tmp_path, caplog, root=elsewhere)
    caplog.set_level(logging.INFO)
    flags = ("--runs-dir", str(elsewhere))
    code, text = _privacy(tmp_path, store, runs_dir=tmp_path / "empty", flags=flags)
    assert code == 0
    assert text.splitlines()[0] == "holdout,arm,source,measure,scope,value"
    assert f" root={elsewhere.resolve()}" in caplog.text


@pytest.mark.usefixtures("reference")
def test_privacy_without_the_runs_dir_reads_the_contexts_root_and_misses_the_folder(
    tmp_path, caplog
):
    elsewhere = tmp_path / "elsewhere" / "runs"
    _, store = _stored(tmp_path, caplog, root=elsewhere)
    assert _privacy(tmp_path, store, runs_dir=tmp_path / "empty") == (2, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert str(tmp_path / "empty" / "run") in errors[-1]


@pytest.mark.usefixtures("reference")
def test_a_symlinked_run_folder_outside_the_root_is_refused(tmp_path, caplog):
    outside = tmp_path / "outside"
    _stored(tmp_path, caplog, root=outside)
    root = tmp_path / "root"
    root.mkdir()
    (root / "run").symlink_to(outside / "run", target_is_directory=True)
    store = tmp_path / "run.jsonl"
    assert _privacy(tmp_path, store, flags=("--runs-dir", str(root))) == (3, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "not a folder directly under" in errors[-1]


@pytest.mark.usefixtures("reference")
@pytest.mark.parametrize("run", ["e1", "nowhere"], ids=["read", "refused"])
def test_privacys_log_opens_with_its_versions_line(tmp_path, caplog, run):
    _, store = _stored(tmp_path, caplog)
    caplog.set_level(logging.INFO)
    code, _ = _privacy(tmp_path, store, run=run)
    assert code == (0 if run == "e1" else 2)
    logged = [r.getMessage() for r in caplog.records]
    assert logged[0].startswith("versions commit=")


def _rewrite(store: Path, edit) -> None:
    """Rewrite `store`'s lines, parsed, through `edit`, which returns the new list."""
    lines = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()]
    text = "".join(
        json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n" for line in edit(lines)
    )
    store.write_text(text, encoding="utf-8")


def _flip(lines: list[dict]) -> list[dict]:
    line = next(line for line in lines[1:] if line["added"])
    csv = line["csv"]
    at = max(i for i, char in enumerate(csv) if char.isdigit())
    line["csv"] = csv[:at] + str((int(csv[at]) + 1) % 10) + csv[at + 1 :]
    return lines


def _stray(lines: list[dict]) -> list[dict]:
    return [*lines, {**lines[-1], "arm": "ghost"}]


def _header(field: str, value: object):
    def edit(lines: list[dict]) -> list[dict]:
        lines[0][field] = value
        return lines

    return edit


@pytest.mark.usefixtures("reference")
@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_flip, "does not hash to its sha256"),
        (operator.itemgetter(slice(-1)), "missing"),
        (_stray, "extra"),
        (_header("log_sha256", "0" * 64), "log_sha256"),
        (_header("run", "b1-uci"), "not a slice of"),
        (_header("folder", "../run"), "not a folder directly under"),
        (_header("folder", ".."), "not a folder directly under"),
        (_header("folder", ""), "not a folder directly under"),
        (_header("folder", "/run"), "not a folder directly under"),
    ],
    ids=[
        "flipped-byte",
        "missing-line",
        "stray-arm",
        "header-log",
        "other-run",
        "traversal",
        "parent",
        "empty",
        "absolute",
    ],
)
def test_a_store_that_fails_an_identity_check_exits_3(tmp_path, caplog, edit, message):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, edit)
    assert _privacy(tmp_path, store) == (3, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert message in errors[-1]


@pytest.mark.usefixtures("reference")
def test_an_identity_error_echoes_a_long_header_value_cut(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, _header("run", "x" * 5_000))
    assert _privacy(tmp_path, store) == (3, "")
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "x" * membership.SHOWN not in error
    assert "x" * (membership.SHOWN - 1) + "..." in error


QUARTET = (0, 1, 2, 3)


def _sliced_runs(experiment: Experiment) -> dict:
    """`RUNS` with `whole`, `experiment` on holdouts 0 to 3, and its slices of two holdouts."""
    ctx = _context(Path())
    named = {
        "whole": dataclasses.replace(experiment, holdout_seeds=QUARTET),
        "whole-a": dataclasses.replace(experiment, holdout_seeds=QUARTET[:2]),
        "whole-b": dataclasses.replace(experiment, holdout_seeds=QUARTET[2:]),
    }
    found = {name: NamedRun(e, ctx.settings, ctx.dataset) for name, e in named.items()}
    return {**ctx.runs, **found}


def _slices(tmp_path: Path, caplog) -> tuple[Path, Path]:
    """Two stores of two holdouts each, their headers naming `whole`'s slices."""
    stores = []
    for seeds, letter in ((QUARTET[:2], "a"), (QUARTET[2:], "b")):
        experiment = dataclasses.replace(PRIVATE, holdout_seeds=seeds)
        _, store = _stored(tmp_path, caplog, name=letter, experiment=experiment)
        _rewrite(store, _header("run", f"whole-{letter}"))
        stores.append(store)
    return stores[0], stores[1]


@pytest.mark.usefixtures("reference")
def test_the_stores_of_a_runs_slices_read_as_the_run(tmp_path, caplog):
    first, second = _slices(tmp_path, caplog)
    caplog.set_level(logging.INFO)
    code, text = _privacy(tmp_path, first, second, run="whole", runs=_sliced_runs(PRIVATE))
    assert code == 0
    assert {s.holdout for s in parse(text)} == set(QUARTET)
    assert "run=whole-a folder=a" in caplog.text
    assert "run=whole-b folder=b" in caplog.text


@pytest.mark.usefixtures("reference")
def test_a_holdout_no_store_holds_exits_3(tmp_path, caplog):
    first, _ = _slices(tmp_path, caplog)
    assert _privacy(tmp_path, first, run="whole", runs=_sliced_runs(PRIVATE)) == (3, "")
    assert "[0, 1] are not run 'whole''s [0, 1, 2, 3]" in caplog.text


@pytest.mark.usefixtures("reference")
def test_a_store_lacking_the_runs_replica_fills_exits_3(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, _header("run", "replicated"))
    replicated = dataclasses.replace(PRIVATE, replica_arms=("smote",))
    ctx = _context(Path())
    runs = {**ctx.runs, "replicated": NamedRun(replicated, ctx.settings, ctx.dataset)}
    assert _privacy(tmp_path, store, run="replicated", runs=runs) == (3, "")
    assert "smote-replica" in caplog.text


@pytest.mark.usefixtures("reference")
def test_a_store_read_twice_overlaps_its_holdouts_and_exits_3(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    assert _privacy(tmp_path, store, store) == (3, "")
    assert "holdouts" in caplog.text


@pytest.mark.usefixtures("reference")
def test_a_split_line_the_data_does_not_reproduce_exits_3(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog)
    edit_log(folder, "split seed=1", lambda line: re.sub(r"fingerprint=\w+", "fingerprint=0", line))
    log = hashlib.sha256((folder / "scores.log").read_bytes()).hexdigest()
    _rewrite(store, _header("log_sha256", log))
    assert _privacy(tmp_path, store) == (3, "")
    assert f"which the store {store} is bound to" in caplog.text
    assert "privacy holdout=1 failed" in caplog.text


def test_a_distance_that_cannot_be_read_names_its_holdout_and_arm(monkeypatch):
    train, test = _split(91)
    real = membership.distances_and_nndr

    def refuse(rows: pd.DataFrame, *args, **kwargs) -> dict:
        if rows is test:
            return real(rows, *args, **kwargs)
        msg = "no rows"
        raise ValueError(msg)

    monkeypatch.setattr(membership, "distances_and_nndr", refuse)
    monkeypatch.setattr(membership, "held_out_ratio", lambda *_args, **_kwargs: 0.25)
    ablated = SimpleNamespace(train=train, test=test, seed=3)
    ready = cast("Prepared", SimpleNamespace(ablated=ablated, sized=None))
    with pytest.raises(EvaluationError, match="holdout 3's arm smote: no rows"):
        membership.holdout_scores(ready, {"smote": test.iloc[:5]}, cell=CELL, frame=UCI_FRAME)


def test_a_run_with_no_reference_arm_is_refused(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    assert _privacy(tmp_path, store)[0] == 2
    assert "no reference arm" in caplog.text


def test_the_reference_arm_resolves_through_the_runs_a_slice_slices(monkeypatch):
    runs = _sliced_runs(PRIVATE)
    monkeypatch.setattr(membership, "REFERENCE", {"whole": REAL_ARM})
    assert reference_of("whole-a", PRIVATE, runs) == (REAL_ARM, CELL)
    assert reference_of("whole", PRIVATE, runs) == (REAL_ARM, CELL)
    twin = {**runs, "twin": runs["whole"]}
    monkeypatch.setattr(membership, "REFERENCE", {"whole": REAL_ARM, "twin": "smote"})
    with pytest.raises(EvaluationError, match="whose reference arms differ"):
        reference_of("whole-a", PRIVATE, twin)


def test_a_run_lacking_its_reference_arm_or_a_removed_cell_is_refused(monkeypatch):
    monkeypatch.setattr(membership, "REFERENCE", {"e1": REAL_ARM})
    lacking = dataclasses.replace(
        PRIVATE, arms=("none", "smote"), candidate="smote", comparators=("none",)
    )
    with pytest.raises(EvaluationError, match="lacks its reference arm 'real'"):
        reference_of("e1", lacking, {})
    monkeypatch.setattr(membership, "REFERENCE", {"e1": "smote"})
    with pytest.raises(EvaluationError, match="removes no cell"):
        reference_of("e1", dataclasses.replace(lacking, removal=None), {})


def test_the_e6_runs_read_against_real_and_the_e10_runs_against_bko_direct():
    assert dict(membership.REFERENCE) == {
        "e6-thin": REAL_ARM,
        "e6-empty": REAL_ARM,
        "e6-curated": REAL_ARM,
        "e10-pilot-thin": DIRECT_ARM,
        "e10-pilot-empty": DIRECT_ARM,
        "e10-pram-thin": DIRECT_ARM,
        "e10-pram-empty": DIRECT_ARM,
    }
