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
from typing import Final

import pandas as pd
import pytest

from bko.evaluate.__main__ import Context, main
from bko.evaluate.data import digest
from bko.evaluate.direct import DirectArm
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, PAID_ARMS, PERTURBED_ARM, PRAM_ARM
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.perturb import Delta
from bko.evaluate.perturbed import Merged, PerturbedArm, _guidance, merge
from bko.evaluate.run_folder import BKO
from bko.generate.decode import Reason, Undecoded
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from bko.generate.rewrite import REWRITE, ROW
from bko.ontology.schema import Codes
from tests._datasets_fixtures import FIXED_NOW
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests._generate_fixtures import reply
from tests.test_evaluate_direct import ADDED_DIGEST, SMALL_G2, g2_context, varied_arm
from tests.test_evaluate_launch import Starts
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
    ("edit", "outcome", "checks"),
    [
        (_keyed("age", 30), "keys", ()),
        (_keyed("sex", 1), "keys", ()),
        (_keyed("target", 0), "keys", ()),
        (_keyed("cp", 1), "keys", ()),
        (lambda row: {key: value for key, value in row.items() if key != "thal"}, "keys", ()),
        (_keyed("chol", 999), "validate", ("range:chol",)),
        (_keyed("fbs", 7), "validate", ("code:fbs",)),
        (_keyed("thal", 4), "validate", ("code:thal",)),
        (_keyed("oldpeak", math.nan), "validate", ("number:oldpeak",)),
        (_keyed("trestbps", math.inf), "validate", ("number:trestbps",)),
        (_keyed("chol", True), "validate", ("number:chol",)),
        (_keyed("chol", "220"), "validate", ("number:chol",)),
        (_keyed("chol", 225.5), "validate", ("precision:chol",)),
        (_keyed("chol", 125), "bounds", ()),
        (_keyed("oldpeak", 7.0), "bounds", ()),
    ],
    ids=[
        "age",
        "sex",
        "label",
        "mapped",
        "missing",
        "above-range",
        "code",
        "unknown-code",
        "nan",
        "inf",
        "bool",
        "text",
        "fraction-of-a-whole",
        "below-training",
        "above-training",
    ],
)
def test_a_planted_invalid_row_keeps_its_original_and_is_counted_by_cause(edit, outcome, checks):
    items = _asked()
    items[1] = edit(items[1])
    merged = _merge(list(items))
    assert merged.outcomes == ("perturbed", outcome)
    assert (merged.reason, merged.checks, merged.refused) == (outcome, checks, 1)
    assert merged.rows[1] == ORIGINALS[1]


def test_an_integral_float_of_a_whole_number_field_merges_as_its_int():
    items = _asked()
    items[1] = {**items[1], "chol": 225.0, "oldpeak": 2}
    merged = _merge(list(items))
    assert merged.outcomes == ("perturbed", "perturbed")
    assert (type(merged.rows[1]["chol"]), merged.rows[1]["chol"]) == (int, 225)
    assert (type(merged.rows[1]["oldpeak"]), merged.rows[1]["oldpeak"]) == (float, 2.0)


def _plus_one(items: list[dict]) -> list[object]:
    return [*items, {**items[0], ROW: 2}]


@pytest.mark.parametrize(
    ("plant", "reason"),
    [
        (_plus_one, "short"),
        (lambda items: [*items[:1], Undecoded(Reason.SHORT)], "short"),
        (operator.itemgetter(slice(None, None, -1)), "order"),
        (lambda items: [items[0], {**items[1], ROW: 0}], "order"),
        (lambda items: [{**items[0], ROW: True}, items[1]], "undecodable"),
        (lambda items: [items[0], {**items[1], ROW: True}], "undecodable"),
        (lambda items: ["not a row", items[1]], "undecodable"),
        (lambda items: [Undecoded(Reason.NOT_JSON)] * 2, "empty"),
    ],
    ids=[
        "asked+1",
        "asked-1",
        "permuted",
        "duplicate",
        "bool-row",
        "bool-later-row",
        "not-a-mapping",
        "empty",
    ],
)
def test_a_planted_row_set_keeps_every_original_of_the_call(plant, reason):
    merged = _merge(plant(_asked()))
    assert merged.outcomes == (reason, reason)
    assert (merged.reason, merged.refused) == (reason, 2)
    assert merged.rows == ORIGINALS


class Rewriter:
    """A send answering each rewrite call with its rows, each one's measures rotated by a row.

    Rotating a measure within a call keeps its values on the training split, so every rewritten
    row passes and the fill's values stay the drawn ones; `invalid` names the calls whose second
    row carries an `age`, which keeps its values, `short` the calls whose last row is left out,
    and `skew` sets every row's measures to the call's least, which the check refuses.

    Attributes:
        calls: Every call made, in order.
    """

    def __init__(
        self,
        *,
        rotate: bool = True,
        invalid: frozenset[int] = frozenset(),
        short: frozenset[int] = frozenset(),
        skew: bool = False,
    ) -> None:
        self.calls: list[Call] = []
        self._rotate = rotate
        self._invalid = invalid
        self._short = short
        self._skew = skew

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
        for name in ("trestbps", "chol", "thalach"):
            values = [rows[index][name] for index in moved]
            if self._skew:
                values = [min(values)] * len(values)
            elif self._rotate:
                values = values[1:] + values[:1]
            for index, value in zip(moved, values, strict=True):
                out[index][name] = value
        if len(self.calls) in self._short:
            out = out[:-1]
        return reply(list[object](out))


def _flown(
    tmp_path: Path, send: Rewriter, delta: float = 1.0
) -> tuple[pd.DataFrame, list[Call], PerturbedArm]:
    train, request, _ = varied_arm()
    inner = DirectArm(
        (ANGINA, STENOSIS), {}, name=PERTURBED_ARM, budget=3, sink=None, dataset="uci-test"
    )
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, now=lambda: FIXED_NOW, frame="uci")
    with log, Flight(send, log, Limits(2, None), spendable=lambda: None) as flight:
        arm = PerturbedArm(inner, flight=flight, fields=FIELDS, delta=Delta(delta, "table", None))
        added = arm(train, request).added
    return added, send.calls, arm


NONE_REFUSED: Final = (
    " refused_empty=0 refused_short=0 refused_undecodable=0 refused_order=0 refused_keys=0"
    " refused_validate=0 refused_bounds=0"
)
"""The holdout line's refusal counts when no row was refused."""


def test_the_perturbed_arms_inner_draw_is_bko_directs_rows(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    added, calls, _ = _flown(tmp_path, Rewriter(rotate=False))
    assert digest(added) == ADDED_DIGEST
    assert calls
    for call in calls:
        assert "Rewrite these 8 records' fields: " in call.user
        for name in ("age", "sex", "target", "cp"):
            assert f'"{name}"' not in call.user.split("The records:")[1]
    rows = 8 * len(calls)
    assert (
        f"perturb holdout=-1 rows={rows} perturbed=0 unchanged={rows}{NONE_REFUSED}"
    ) in caplog.messages


def test_a_rewrite_call_sends_only_the_unmapped_fields_and_their_training_ranges(tmp_path):
    train, _, _ = varied_arm()
    _, calls, _ = _flown(tmp_path, Rewriter())
    for call in calls:
        sent = [json.loads(line) for line in call.user.splitlines() if line.startswith("{")]
        assert sent
        assert all(set(row) == {ROW, *FIELDS} for row in sent)
        for name in ("trestbps", "chol", "thalach", "oldpeak"):
            low, high = float(train[name].min()), float(train[name].max())
            assert f"{name} {low:g} to {high:g} " in call.user, name


def test_the_perturbed_arm_logs_each_calls_counts_and_reason(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    added, calls, arm = _flown(tmp_path, Rewriter(invalid=frozenset({1}), short=frozenset({2})))
    train, request, direct = varied_arm()
    theirs = direct(train, request).added
    kept = [column for column in theirs.columns if column not in FIELDS]
    pd.testing.assert_frame_equal(added[kept], theirs[kept])
    lines = [m for m in caplog.messages if m.startswith("perturb call=")]
    assert len(lines) == len(calls)
    assert lines[0] == (
        "perturb call=1 holdout=-1 rows=8 returned=8 accepted=7 refused=1 reason=keys"
        " stop_reason=end_turn checks=none"
    )
    assert lines[1] == (
        "perturb call=2 holdout=-1 rows=8 returned=7 accepted=0 refused=8 reason=short"
        " stop_reason=end_turn checks=none"
    )
    assert all(
        line.endswith(" refused=0 reason=none stop_reason=end_turn checks=none")
        for line in lines[2:]
    )
    summary = [m for m in caplog.messages if m.startswith("perturb holdout=")]
    assert len(summary) == 1
    assert re.fullmatch(
        r"perturb holdout=-1 rows=\d+ perturbed=\d+ unchanged=\d+ refused_empty=0 refused_short=8"
        r" refused_undecodable=0 refused_order=0 refused_keys=1 refused_validate=0"
        r" refused_bounds=0",
        summary[0],
    )
    generated = [m for m in caplog.messages if m.startswith("generate call=")]
    assert len(generated) == len(calls)
    assert all(" rewrite=true " in m and "level=" not in m for m in generated)
    (short,) = [m for m in caplog.messages if m.startswith("short call=")]
    assert short.endswith(" rewrite=true")
    (configured,) = [m for m in caplog.messages if m.startswith("perturb arm=")]
    assert configured == (
        f"perturb arm={PERTURBED_ARM} fields={','.join(FIELDS)} delta=1.000000"
        f" delta_source=table fraction=none per_call={SETTINGS.per_call} model={SETTINGS.model}"
    )
    caplog.clear()
    arm.summary()
    (line,) = [m for m in caplog.messages if m.startswith("perturb summary ")]
    assert re.fullmatch(
        rf"perturb summary arm={PERTURBED_ARM} holdouts=1 accepted=1 refused=0 empty=0"
        r" rows=\d+ rows_changed=\d+ perturbed=\d+ unchanged=\d+ refused_empty=0"
        r" refused_short=8 refused_undecodable=0 refused_order=0 refused_keys=1"
        r" refused_validate=0 refused_bounds=0",
        line,
    )


def test_a_check_below_the_reference_keeps_every_drawn_row(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    added, calls, arm = _flown(tmp_path, Rewriter(skew=True), delta=0.0)
    assert calls
    assert digest(added) == ADDED_DIGEST
    (check,) = [m for m in caplog.messages if m.startswith("perturb check ")]
    assert check.endswith(
        " delta=0.000000 pair_trends="
        + check.split("pair_trends=")[1].split(" ")[0]
        + " outcome=refused reason=below_reference rows_changed=0 final=original"
    )
    assert (arm.counts["refused"], arm.counts["rows_changed"]) == (1, 0)


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
    for call in (2, 3, 4):
        positions = sorted(line.position for line in called if line.call == call)
        assert positions == list(range(len(positions))), call
    checks = [
        m for m in caplog.messages if m.startswith(f"perturb check holdout=0 arm={PERTURBED_ARM}")
    ]
    assert checks
    assert " outcome=accepted reason=none rows_changed=" in checks[0]
    assert checks[0].endswith(" final=perturbed")


def test_a_refused_holdouts_rows_lines_carry_no_call(tmp_path, caplog):
    folder, _ = perturbed_folder(tmp_path, caplog, Rewriter(skew=True))
    found = read((folder / "holdout-candidates.jsonl").read_bytes())
    mine = [line for line in found.candidates if line.arm == PERTURBED_ARM]
    assert len(mine) == 2 * 28
    assert {line.call for line in mine} == {None}
    checks = [m for m in caplog.messages if m.startswith("perturb check holdout=")]
    mine_checks = [m for m in checks if f" arm={PERTURBED_ARM} " in m]
    assert len(mine_checks) == 2
    assert all(
        m.endswith(" outcome=refused reason=below_reference rows_changed=0 final=original")
        for m in mine_checks
    ), mine_checks


def test_the_stored_arm_rebuilds_the_perturbed_arms_rows_to_its_arm_sha(tmp_path, caplog):
    folder, ctx = perturbed_folder(tmp_path, caplog, Rewriter(invalid=frozenset({1})))
    argv = ["shared", "g2", "--folder", str(folder), "--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    assert main(argv, context=ctx, out=out) == 0
    assert out.getvalue() == (folder / "scores.csv").read_text(encoding="utf-8")


def test_compare_names_the_level_of_a_run_whose_only_paid_arm_is_perturbed(tmp_path, caplog):
    folder, ctx = perturbed_folder(tmp_path, caplog, Rewriter())
    caplog.clear()
    argv = ["compare", "g2", str(folder / "scores.csv")]
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    (header,) = [m for m in caplog.messages if m.startswith("compare candidate=")]
    assert f" level={ctx.settings.guidance} " in header


def test_launch_takes_a_run_whose_only_paid_arm_is_perturbed(tmp_path):
    start = Starts()
    runs = {"g2": NamedRun(PERTURBED)}
    context = Context(power=lambda: False, runs_dir=tmp_path, start=start, runs=runs)
    assert main(["launch", "g2-3", "--dry-run"], context=context, out=io.StringIO()) == 0
    assert (tmp_path / "g2-3" / "launch.log").is_file()


def test_nested_field_rows_refuse_the_perturbed_arm_before_any_download(tmp_path, caplog):
    ctx, _ = g2_context()
    chain = tuple(
        dataclasses.replace(ANGINA, concept=f"HP:900000{index}", when=(Codes("cp", codes),))
        for index, codes in enumerate((frozenset({1}), frozenset({1, 2}), frozenset({1, 2, 3})))
    )
    alone = dataclasses.replace(
        PERTURBED,
        arms=("none", PERTURBED_ARM),
        candidate=PERTURBED_ARM,
        comparators=("none",),
    )
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(alone)}, field_rows=chain)
    argv = ["score", "g2", "--log", str(tmp_path / "calls.jsonl")]
    argv += ["--candidates", str(tmp_path / "c.jsonl"), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert f"the direct arms {[PERTURBED_ARM]} cannot draw dataset uci's rows: " in caplog.text
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("arm", [PERTURBED_ARM, PRAM_ARM])
def test_trace_refuses_a_perturbation_arms_fill(tmp_path, caplog, arm):
    folder, ctx = perturbed_folder(tmp_path, caplog, Rewriter())
    argv = ["trace", "g2", "--folder", str(folder), "--holdout", "0", "--arm", arm]
    argv += ["--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert f"trace cannot explain {arm}" in caplog.text
    assert "cleanup:evaluate/trace-logic-arm" in caplog.text


def test_a_rewrite_call_for_a_cell_no_sex_code_names_is_refused():
    with pytest.raises(EvaluationError, match="no sex code names cell"):
        _guidance(("6-7", "unknown", 1))
