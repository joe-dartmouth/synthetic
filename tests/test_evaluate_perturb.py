"""`bko-direct-perturbed`: rewrite calls on `bko-direct`'s rows, merged by `row`, each refusal
counted."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import math
import operator
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from bko.evaluate.__main__ import Context, main
from bko.evaluate.data import digest
from bko.evaluate.direct import DirectArm
from bko.evaluate.experiment import DIRECT_ARM, PAID_ARMS, PERTURBED_ARM, PRAM_ARM
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.perturb import Merged, PerturbedArm, merge
from bko.evaluate.run_folder import BKO
from bko.generate.decode import Reason, Undecoded
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from bko.generate.rewrite import REWRITE, ROW
from tests._datasets_fixtures import FIXED_NOW
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests._generate_fixtures import reply
from tests.test_evaluate_direct import ADDED_DIGEST, SMALL_G2, g2_context, varied_arm
from tests.test_evaluate_pram import FIELDS
from tests.test_evaluate_shared import FORMAT


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ORIGINALS = (
    {
        "age": 50,
        "sex": 0,
        "cp": 4,
        "trestbps": 120,
        "chol": 220,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
        "target": 1,
    },
    {
        "age": 52,
        "sex": 0,
        "cp": 3,
        "trestbps": 130,
        "chol": 240,
        "fbs": 1,
        "restecg": 2,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 2.0,
        "slope": 2,
        "ca": 1,
        "thal": 7,
        "target": 1,
    },
)
"""Two drawn rows of one call."""

BOUNDS = {"trestbps": (94.0, 200.0), "chol": (126.0, 564.0), "thalach": (71.0, 202.0)}
BOUNDS |= {"oldpeak": (0.0, 6.2)}
"""The holdout's training ranges of the measured fields."""


def _asked() -> list[dict]:
    """The reply that rewrites each row: its own fields, each measure moved within range."""
    return [
        {
            ROW: index,
            **{name: row[name] for name in FIELDS},
            "chol": row["chol"] + 5,
            "fbs": 1 - row["fbs"],
        }
        for index, row in enumerate(ORIGINALS)
    ]


def _merge(items: list[object]) -> Merged:
    frame = varied_arm()[1].frame
    return merge(items, list(ORIGINALS), fields=FIELDS, bounds=BOUNDS, frame=frame)


def test_merging_by_row_changes_only_the_rewritten_fields():
    merged = _merge(list(_asked()))
    assert merged.outcomes == ("perturbed", "perturbed")
    assert (merged.reason, merged.returned) == ("none", 2)
    for row, original in zip(merged.rows, ORIGINALS, strict=True):
        assert row["chol"] == original["chol"] + 5
        for name in ("age", "sex", "cp", "target"):
            assert row[name] == original[name]


def test_an_unchanged_row_is_counted():
    items = _asked()
    items[1] = {ROW: 1, **{name: ORIGINALS[1][name] for name in FIELDS}}
    merged = _merge(list(items))
    assert merged.outcomes == ("perturbed", "unchanged")
    assert merged.rows[1] == ORIGINALS[1]


def _keyed(name: str, value: object) -> Callable[[dict], dict]:
    return lambda row: {**row, name: value}


@pytest.mark.parametrize(
    "edit",
    [
        _keyed("age", 30),
        _keyed("sex", 1),
        _keyed("target", 0),
        _keyed("cp", 1),
        _keyed("chol", 999),
        _keyed("chol", 125),
        _keyed("fbs", 7),
        _keyed("thal", 4),
        _keyed("oldpeak", math.nan),
        _keyed("trestbps", math.inf),
        _keyed("chol", True),
        _keyed("chol", "220"),
        lambda row: {key: value for key, value in row.items() if key != "thal"},
    ],
    ids=[
        "age",
        "sex",
        "label",
        "mapped",
        "above-range",
        "below-range",
        "code",
        "unknown-code",
        "nan",
        "inf",
        "bool",
        "text",
        "missing",
    ],
)
def test_a_planted_invalid_row_keeps_its_original_and_is_counted(edit):
    items = _asked()
    items[1] = edit(items[1])
    merged = _merge(list(items))
    assert merged.outcomes == ("perturbed", "invalid")
    assert merged.reason == "invalid"
    assert merged.rows[1] == ORIGINALS[1]


def _plus_one(items: list[dict]) -> list[object]:
    return [*items, {**items[0], ROW: 2}]


@pytest.mark.parametrize(
    "plant",
    [
        _plus_one,
        operator.itemgetter(slice(1)),
        operator.itemgetter(slice(None, None, -1)),
        lambda items: [items[0], {**items[1], ROW: 0}],
        lambda items: [{**items[0], ROW: True}, items[1]],
        lambda items: ["not a row", items[1]],
        lambda items: [Undecoded(Reason.NOT_JSON)] * 2,
    ],
    ids=["asked+1", "asked-1", "permuted", "duplicate", "bool-row", "not-a-mapping", "undecodable"],
)
def test_a_planted_row_set_keeps_every_original_of_the_call(plant):
    merged = _merge(plant(_asked()))
    assert merged.outcomes == ("rows", "rows")
    assert merged.reason == "rows"
    assert merged.rows == ORIGINALS


class Rewriter:
    """A send answering each rewrite call with its rows, each one's measures rotated by a row.

    Rotating a measure within a call keeps its values on the training split, so every rewritten
    row passes and the fill's values stay the drawn ones; `invalid` names the calls whose second
    row carries an `age`, which keeps its values.

    Attributes:
        calls: Every call made, in order.
    """

    def __init__(self, *, rotate: bool = True, invalid: frozenset[int] = frozenset()) -> None:
        self.calls: list[Call] = []
        self._rotate = rotate
        self._invalid = invalid

    def __call__(self, call: Call) -> Reply:
        self.calls.append(call)
        assert REWRITE.match(call.user), call.user[:200]
        rows = [json.loads(line) for line in call.user.splitlines() if line.startswith("{")]
        out = [dict(row) for row in rows]
        planted = len(self.calls) in self._invalid and len(out) > 1
        if planted:
            out[1]["age"] = 30
        # The planted row keeps its values, so the rows kept hold the values drawn.
        moved = [index for index in range(len(out)) if not (planted and index == 1)]
        if self._rotate:
            for name in ("trestbps", "chol", "thalach"):
                values = [rows[index][name] for index in moved]
                for index, value in zip(moved, values[1:] + values[:1], strict=True):
                    out[index][name] = value
        return reply(list[object](out))


def _flown(tmp_path: Path, send: Rewriter) -> tuple[pd.DataFrame, list[Call]]:
    train, request, _ = varied_arm()
    inner = DirectArm(
        (ANGINA, STENOSIS), {}, name=PERTURBED_ARM, budget=3, sink=None, dataset="uci-test"
    )
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, now=lambda: FIXED_NOW, frame="uci")
    with log, Flight(send, log, Limits(2, None), spendable=lambda: None) as flight:
        arm = PerturbedArm(inner, flight=flight, fields=FIELDS, delta=1.0)
        added = arm(train, request).added
    return added, send.calls


def test_the_perturbed_arms_inner_draw_is_bko_directs_rows(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    added, calls = _flown(tmp_path, Rewriter(rotate=False))
    assert digest(added) == ADDED_DIGEST
    assert calls
    for call in calls:
        assert "Rewrite these 8 records' fields: " in call.user
        for name in ("age", "sex", "target", "cp"):
            assert f'"{name}"' not in call.user.split("The records:")[1]
    rows = 8 * len(calls)
    assert (
        f"perturb holdout=-1 rows={rows} perturbed=0 unchanged={rows} refused_rows=0"
        " refused_invalid=0"
    ) in caplog.messages


def test_the_perturbed_arm_logs_each_calls_counts_and_reason(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    added, calls = _flown(tmp_path, Rewriter(invalid=frozenset({1})))
    train, request, direct = varied_arm()
    theirs = direct(train, request).added
    kept = [column for column in theirs.columns if column not in FIELDS]
    pd.testing.assert_frame_equal(added[kept], theirs[kept])
    lines = [m for m in caplog.messages if m.startswith("perturb call=")]
    assert len(lines) == len(calls)
    assert lines[0] == (
        "perturb call=1 holdout=-1 rows=8 returned=8 kept=7 refused=1 reason=invalid"
    )
    assert all(line.endswith(" refused=0 reason=none") for line in lines[1:])
    summary = [m for m in caplog.messages if m.startswith("perturb holdout=")]
    assert len(summary) == 1
    assert re.fullmatch(
        r"perturb holdout=-1 rows=\d+ perturbed=\d+ unchanged=\d+ refused_rows=0 refused_invalid=1",
        summary[0],
    )


def test_the_perturbed_arm_is_a_paid_bko_arm():
    assert PERTURBED_ARM in PAID_ARMS
    assert PERTURBED_ARM in BKO


PERTURBED = dataclasses.replace(
    SMALL_G2,
    arms=("none", DIRECT_ARM, PERTURBED_ARM, PRAM_ARM),
    candidate=PERTURBED_ARM,
    comparators=("none", DIRECT_ARM, PRAM_ARM),
    replica_arms=(),
)
"""Two holdouts, one baseline, `bko-direct` and both perturbation arms."""


def perturbed_folder(tmp_path: Path, caplog, send: Rewriter) -> tuple[Path, Context]:
    """A paid `score g2` folder of `PERTURBED` over `send`, and its context."""
    caplog.set_level(logging.INFO)
    ctx, _ = g2_context()
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(PERTURBED)}, send=lambda: send)
    folder = tmp_path / "run"
    folder.mkdir()
    argv = ["score", "g2", "--log", str(folder / "calls.jsonl")]
    argv += ["--candidates", str(folder / "holdout-candidates.jsonl")]
    out = io.StringIO()
    caplog.clear()
    assert main([*argv, "--data-dir", str(tmp_path / "data")], context=ctx, out=out) == 0
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    return folder, ctx


def test_a_perturbed_rows_line_carries_its_call_and_a_kept_original_none(tmp_path, caplog):
    folder, _ = perturbed_folder(tmp_path, caplog, Rewriter(invalid=frozenset({1})))
    found = read((folder / "holdout-candidates.jsonl").read_bytes())
    mine = [line for line in found.candidates if line.arm == PERTURBED_ARM]
    assert len(mine) == 2 * 28
    assert {line.outcome for line in mine} == {KEPT}
    called = [line for line in mine if line.call is not None]
    assert len(called) == len(mine) - 1
    (original,) = [line for line in mine if line.call is None]
    assert (original.holdout, original.position) == (0, 1)
    assert {line.call for line in called} == {1, 2, 3, 4}
    assert {line.position for line in called if line.call == 1} == set(range(12)) - {1}
    checks = [
        m for m in caplog.messages if m.startswith(f"perturb check holdout=0 arm={PERTURBED_ARM}")
    ]
    assert checks
    assert checks[0].endswith("outcome=accepted")


def test_the_stored_arm_rebuilds_the_perturbed_arms_rows_to_its_arm_sha(tmp_path, caplog):
    folder, ctx = perturbed_folder(tmp_path, caplog, Rewriter(invalid=frozenset({1})))
    argv = ["shared", "g2", "--folder", str(folder), "--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    assert main(argv, context=ctx, out=out) == 0
    assert out.getvalue() == (folder / "scores.csv").read_text(encoding="utf-8")
