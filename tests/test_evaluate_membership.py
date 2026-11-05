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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import cell_keys
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import membership
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.experiment import REAL_ARM
from bko.evaluate.membership import (
    GROUPS,
    SCOPES,
    Attack,
    arm_measures,
    attack,
    exposure,
    mi,
    scopes,
)
from bko.evaluate.privacy_rule import Reading, readings
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, parse
from tests.test_evaluate_privacy import random_records
from tests.test_evaluate_rows import BASELINE_ARMS, REAL
from tests.test_evaluate_shared import edit_log, rescorer, run_folder


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
    # One member above every non-member, one tied with the top non-member: the curve runs
    # straight from (0, 0.5) to (0.01, 1.0) across the tie.
    nonmember_d = np.array([0.2, *np.linspace(0.3, 0.9, 99)])
    found = attack(np.array([0.1, 0.2]), nonmember_d)
    assert found.tpr[0.01] == pytest.approx(1.0)
    assert found.tpr[0.001] == pytest.approx(0.55)


def _split(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            _placed(random_records(UCI_FRAME, 100, seed=seed + 2), inside=False),
            _placed(random_records(UCI_FRAME, 5, seed=seed + 3), inside=True),
        ],
        ignore_index=True,
    )
    return train, test


def test_a_copied_test_record_stays_a_non_member():
    train, test = _split(61)
    added = pd.concat([random_records(UCI_FRAME, 50, seed=65), test.iloc[[0]]], ignore_index=True)
    found = arm_measures(added, train=train, test=test, cell=CELL, frame=UCI_FRAME)
    inner, outer = (scopes(x, CELL, edges=UCI_FRAME.edges)["outside"] for x in (train, test))
    member_d = exposure(train, added, train=train, frame=UCI_FRAME)[inner]
    nonmember_d = exposure(test, added, train=train, frame=UCI_FRAME)[outer]
    assert found[mi("auc", "outside")] == attack(member_d, nonmember_d).auc
    assert found[mi("auc", "outside")] != attack(member_d, nonmember_d[1:]).auc


def test_an_arm_measures_every_privacy_measure_and_none_is_read_as_zero():
    train, test = _split(71)
    found = arm_measures(test.iloc[:30], train=train, test=test, cell=CELL, frame=UCI_FRAME)
    assert list(found) == list(membership.ADDED_MEASURES)
    assert all(math.isfinite(value) for value in found.values())
    empty = arm_measures(test.iloc[:0], train=train, test=test, cell=CELL, frame=UCI_FRAME)
    assert list(empty) == list(membership.ADDED_MEASURES)
    assert all(math.isnan(value) for value in empty.values())


def test_the_cell_scope_is_undefined_with_no_member_in_the_cell():
    train, test = _split(81)
    outside = train[scopes(train, CELL, edges=UCI_FRAME.edges)["outside"]]
    found = arm_measures(test.iloc[:30], train=outside, test=test, cell=CELL, frame=UCI_FRAME)
    assert math.isnan(found[mi("auc", "cell")])
    assert math.isnan(found[mi("tpr01", "cell")])
    assert math.isfinite(found[mi("auc", "outside")])
    assert math.isfinite(found[mi("auc", "label")])


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
    assert leak.exceeds_reference
    assert not null.exceeds_reference
    assert (leak.family, leak.against, leak.scope) == (4, REAL_ARM, "outside")
    assert leak.contrast == pytest.approx(0.1)
    assert leak.mean == pytest.approx(0.6)
    assert not any(r.arm == REAL_ARM for r in found)


def test_the_family_size_changes_the_outcome():
    alone = {"arm": [0.05 + w for w in WOBBLE]}
    crowded = {**alone, **{f"null-{n}": WOBBLE for n in range(19)}}
    for arms, family, flagged in ((alone, 2, True), (crowded, 40, False)):
        scores = _scored(arms, MI_OUTSIDE[:1])
        found = readings(scores, GROUPS, arms=[*arms, REAL_ARM], reference=REAL_ARM)
        reading = _reading(found, "arm", mi("auc", "outside"))
        assert (reading.family, reading.exceeds_reference) == (family, flagged)


def test_nan_holdouts_are_counted_and_never_read_as_zero():
    contrasts = [0.1 + w / 3 for w in WOBBLE]
    gappy = [math.nan, math.nan, math.nan, *contrasts[3:]]
    arms = {"gappy": gappy, "empty": [math.nan] * HOLDOUTS}
    found = readings(_scored(arms, MI_OUTSIDE), GROUPS, arms=[*arms, REAL_ARM], reference=REAL_ARM)
    reading = _reading(found, "gappy", mi("auc", "outside"))
    assert (reading.holdouts, reading.undefined) == (HOLDOUTS, 3)
    assert reading.contrast == pytest.approx(float(np.mean(contrasts[3:])))
    assert reading.family == 2
    empty = _reading(found, "empty", mi("auc", "outside"))
    assert (empty.undefined, empty.exceeds_reference) == (HOLDOUTS, False)
    assert math.isnan(empty.contrast)
    assert math.isnan(empty.low)
    assert math.isnan(empty.mean)


def test_a_distance_farther_than_the_reference_exceeds_it():
    measures = [Measure.DCR_P5, Measure.DCR_MEDIAN, Measure.NNDR_P5, Measure.NNDR_MEDIAN]
    arms = {"near": [-0.1 + w / 3 for w in WOBBLE], "far": [0.1 + w / 3 for w in WOBBLE]}
    found = readings(_scored(arms, measures), GROUPS, arms=[*arms, REAL_ARM], reference=REAL_ARM)
    near, far = (_reading(found, arm, Measure.DCR_P5) for arm in ("near", "far"))
    assert far.exceeds_reference
    assert not near.exceeds_reference
    assert (near.family, near.scope) == (8, "all")


# The command.


def _no_send() -> object:
    msg = "the privacy reading made an API call"
    raise AssertionError(msg)


def _context(tmp_path: Path) -> Context:
    ctx = rescorer(experiment=PRIVATE, arms=BASELINE_ARMS)
    return dataclasses.replace(ctx, send=_no_send, runs_dir=tmp_path)


@pytest.fixture
def reference(monkeypatch):
    """`e1`, which the test context's experiment stands in for, read against `real`."""
    monkeypatch.setattr(membership, "REFERENCE", {**membership.REFERENCE, "e1": REAL_ARM})


def _stored(tmp_path: Path, caplog) -> tuple[Path, Path]:
    """`PRIVATE`'s folder `run` under `tmp_path` and its `rows` store."""
    folder = run_folder(tmp_path, "run", caplog, experiment=PRIVATE, arms=BASELINE_ARMS)
    store = tmp_path / "rows.jsonl"
    argv = ["rows", "--folder", str(folder), "--rows", str(store)]
    argv += ["--data-dir", str(tmp_path / "data")]
    assert main(argv, context=_context(tmp_path), out=io.StringIO()) == 0
    caplog.clear()
    return folder, store


def _privacy(tmp_path: Path, *stores: Path) -> tuple[int, str]:
    argv = ["privacy", "--data-dir", str(tmp_path / "data")]
    for store in stores:
        argv += ["--rows", str(store)]
    out = io.StringIO()
    code = main(argv, context=_context(tmp_path), out=out)
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
    assert all(math.isnan(ours[h, "none", Source.ADDED, mi("auc", "outside")]) for h in (0, 1))
    assert all(math.isfinite(ours[h, "smote", Source.ADDED, mi("auc", "cell")]) for h in (0, 1))
    assert text.splitlines()[0] == "holdout,arm,source,measure,scope,value"
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert f"privacy rows={text.count(chr(10)) - 1} sha256={sha}" in caplog.text
    line = re.search(
        r"privacy arm=smote scope=outside measure=mi-auc-outside mean=\S+ against=real"
        r" contrast=\S+ low=\S+ high=\S+ family=(\d+) holdouts=2 undefined=0"
        r" exceeds_reference=(true|false)",
        caplog.text,
    )
    assert line is not None
    assert line[1] == "2"
    assert "privacy arm=none scope=outside measure=mi-auc-outside" in caplog.text
    assert "undefined=2 exceeds_reference=false" in caplog.text
    assert "privacy holdout=1 done=2/2" in caplog.text


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


def _header(lines: list[dict]) -> list[dict]:
    lines[0]["log_sha256"] = "0" * 64
    return lines


def _elsewhere(lines: list[dict]) -> list[dict]:
    lines[0]["run"] = "b1-uci"
    return lines


@pytest.mark.usefixtures("reference")
@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_flip, "does not hash to its sha256"),
        (operator.itemgetter(slice(-1)), "missing"),
        (_stray, "extra"),
        (_header, "log_sha256"),
        (_elsewhere, "not a slice of"),
    ],
    ids=["flipped-byte", "missing-line", "stray-arm", "header-log", "other-run"],
)
def test_a_store_that_fails_an_identity_check_exits_3(tmp_path, caplog, edit, message):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, edit)
    assert _privacy(tmp_path, store) == (3, "")
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert message in error


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

    def resealed(lines: list[dict]) -> list[dict]:
        lines[0]["log_sha256"] = log
        return lines

    _rewrite(store, resealed)
    assert _privacy(tmp_path, store) == (3, "")
    assert "split" in caplog.text


def test_a_run_with_no_reference_arm_is_refused(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    assert _privacy(tmp_path, store)[0] == 2
    assert "no reference arm" in caplog.text


def test_the_e6_runs_read_against_real():
    assert dict(membership.REFERENCE) == {
        "e6-thin": REAL_ARM,
        "e6-empty": REAL_ARM,
        "e6-curated": REAL_ARM,
    }
