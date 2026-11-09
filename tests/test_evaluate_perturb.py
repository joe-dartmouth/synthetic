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
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from bko.datasets.diabetes import CHANGE_CODES, DIABETES_MED_CODES, DRUG_CODES, DRUGS
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import perturbed as perturbed_module
from bko.evaluate.__main__ import Context, main
from bko.evaluate.data import digest
from bko.evaluate.direct import DirectArm
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, PAID_ARMS, PERTURBED_ARM, PRAM_ARM
from bko.evaluate.fidelity import quality_report
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.perturb import POSITIVE, incoherent, label_rows
from bko.evaluate.perturbed import Merged, PerturbedArm, _guidance, merge
from bko.evaluate.quotas import allot, categories, category_of
from bko.evaluate.run_folder import BKO
from bko.generate.decode import Reason, Undecoded
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from bko.generate.rewrite import COUNTS_HEADER, REWRITE, ROW
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
    and `skew` sets every row's measures to the call's least, which reads below the unperturbed
    rows.

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
    tmp_path: Path, send: Rewriter, train: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[Call], PerturbedArm]:
    varied, request, _ = varied_arm()
    train = varied if train is None else train
    inner = DirectArm(
        (ANGINA, STENOSIS), {}, name=PERTURBED_ARM, budget=3, sink=None, dataset="uci-test"
    )
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, now=lambda: FIXED_NOW, frame="uci")
    with log, Flight(send, log, Limits(2, None), spendable=lambda: None) as flight:
        arm = PerturbedArm(inner, flight=flight, fields=FIELDS)
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


def _bounds(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        name: (float(train[name].min()), float(train[name].max()))
        for name in FIELDS
        if name not in UCI_FRAME.categorical
    }


def _block(call: Call) -> dict[str, dict[str, int]]:
    """Each field's counts in `call`'s quota block, keyed by the value or range stated."""
    block = call.user.split(COUNTS_HEADER + "\n", 1)[1].split("\n\nThe records:", 1)[0]
    found: dict[str, dict[str, int]] = {}
    for line in block.splitlines():
        name, counts = line.removeprefix("- ").split(": ", 1)
        found[name] = {
            stated: int(count) for stated, count in (s.split(" x") for s in counts.split(", "))
        }
    return found


def _stated(low: float, high: float) -> str:
    return f"{low}" if low == high else f"{low}-{high}"


def _quota_off(call: Call, train: pd.DataFrame, rows: list[dict]) -> int:
    """Each quota's shortfall in `rows`, summed over the fields and their categories."""
    found = categories(label_rows(train, UCI_FRAME, call.label), FIELDS, UCI_FRAME, _bounds(train))
    stated = _block(call)
    off = 0
    for name in FIELDS:
        actual = Counter(category_of(row[name], found[name]) for row in rows)
        for index, category in enumerate(found[name]):
            quota = stated[name].get(_stated(category.low, category.high), 0)
            off += max(0, quota - actual[index])
    return off


def _sent(call: Call) -> list[dict]:
    return [json.loads(line) for line in call.user.splitlines() if line.startswith("{")]


def test_a_rewrite_call_sends_only_the_unmapped_fields_their_ranges_and_quotas(tmp_path):
    train, _, _ = varied_arm()
    _, calls, _ = _flown(tmp_path, Rewriter())
    for call in calls:
        sent = _sent(call)
        assert sent
        assert all(set(row) == {ROW, *FIELDS} for row in sent)
        for name in ("trestbps", "chol", "thalach", "oldpeak"):
            low, high = float(train[name].min()), float(train[name].max())
            assert f"{name} {low:g} to {high:g} " in call.user, name
        block = _block(call)
        assert tuple(block) == FIELDS
        assert all(sum(counts.values()) == len(sent) for counts in block.values())


def test_a_rewrite_call_sends_the_label_groups_quotas(tmp_path):
    train, _, _ = varied_arm()
    train = train.reset_index(drop=True)
    ones = train.index.to_series().groupby(train["target"]).cumcount() % 10
    train["fbs"] = ((train["target"] == 1) == (ones != 0)).astype(int)
    _, calls, _ = _flown(tmp_path, Rewriter(), train=train)
    assert {call.label for call in calls} == {0, 1}
    for call in calls:
        found = categories(
            label_rows(train, UCI_FRAME, call.label), FIELDS, UCI_FRAME, _bounds(train)
        )
        (quotas,) = allot(found, [len(_sent(call))])
        expected = {
            _stated(c.low, c.high): n for c, n in zip(found["fbs"], quotas["fbs"], strict=True) if n
        }
        assert _block(call)["fbs"] == expected
    by_label = {call.label: _block(call)["fbs"] for call in calls}
    assert by_label == {1: {"0": 1, "1": 7}, 0: {"0": 7, "1": 1}}


def test_a_cell_whose_label_group_has_no_training_row_is_refused_before_any_call(
    tmp_path, monkeypatch
):
    def no_label_zero(train: pd.DataFrame, frame: object, label: int) -> pd.DataFrame:
        return train.iloc[:0] if label == 0 else label_rows(train, UCI_FRAME, label)

    monkeypatch.setattr(perturbed_module, "label_rows", no_label_zero)
    send = Rewriter()
    with pytest.raises(EvaluationError, match="no training row"):
        _flown(tmp_path, send)
    assert send.calls == []
    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["kind"] for line in lines} <= {"run", "end"}


def test_each_call_logs_its_quota_off(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    train, _, _ = varied_arm()
    _, calls, _ = _flown(tmp_path, Rewriter())
    lines = [m for m in caplog.messages if m.startswith("perturb call=")]
    assert len(lines) == len(calls)
    offs = [_quota_off(call, train, _sent(call)) for call in calls]
    assert any(offs)
    for index, (line, off) in enumerate(zip(lines, offs, strict=True)):
        assert line.startswith(f"perturb call={index + 1} "), line
        assert line.endswith(f" checks=none quota_off={off}"), line


def test_a_calls_quota_off_reads_the_replys_rows_not_the_sent_ones(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    train, _, _ = varied_arm()
    _, calls, _ = _flown(tmp_path, Rewriter(skew=True))
    lines = [m for m in caplog.messages if m.startswith("perturb call=")]
    assert len(lines) == len(calls)
    differs = 0
    for line, call in zip(lines, calls, strict=True):
        sent = _sent(call)
        replied = [dict(row) for row in sent]
        for name in ("trestbps", "chol", "thalach"):
            least = min(row[name] for row in sent)
            for row in replied:
                row[name] = least
        off = _quota_off(call, train, replied)
        assert line.endswith(f" quota_off={off}"), line
        differs += off != _quota_off(call, train, sent)
    assert differs


def test_the_perturbed_arm_logs_each_calls_counts_and_reason(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    added, calls, arm = _flown(tmp_path, Rewriter(invalid=frozenset({1}), short=frozenset({2})))
    train, request, direct = varied_arm()
    theirs = direct(train, request).added
    kept = [column for column in theirs.columns if column not in FIELDS]
    pd.testing.assert_frame_equal(added[kept], theirs[kept])
    lines = [m for m in caplog.messages if m.startswith("perturb call=")]
    assert len(lines) == len(calls)
    offs = [_quota_off(call, train, _sent(call)) for call in calls]
    assert lines[0] == (
        "perturb call=1 holdout=-1 rows=8 returned=8 accepted=7 refused=1 reason=keys"
        f" stop_reason=end_turn checks=none quota_off={offs[0]}"
    )
    assert lines[1] == (
        "perturb call=2 holdout=-1 rows=8 returned=7 accepted=0 refused=8 reason=short"
        f" stop_reason=end_turn checks=none quota_off={offs[1]}"
    )
    assert all(
        line.endswith(f" refused=0 reason=none stop_reason=end_turn checks=none quota_off={off}")
        for line, off in zip(lines[2:], offs[2:], strict=True)
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
        f"perturb arm={PERTURBED_ARM} fields={','.join(FIELDS)}"
        f" per_call={SETTINGS.per_call} model={SETTINGS.model}"
    )
    caplog.clear()
    arm.summary()
    (line,) = [m for m in caplog.messages if m.startswith("perturb summary ")]
    assert re.fullmatch(
        rf"perturb summary arm={PERTURBED_ARM} holdouts=1 read=1 empty=0 rows=\d+"
        r" rows_changed=\d+ values_changed=\d+ mean_shapes=\d\.\d{6} mean_unperturbed=\d\.\d{6}"
        r" perturbed=\d+ unchanged=\d+ refused_empty=0"
        r" refused_short=8 refused_undecodable=0 refused_order=0 refused_keys=1"
        r" refused_validate=0 refused_bounds=0",
        line,
    )


def test_a_perturbed_reading_below_the_unperturbed_keeps_the_arms_rows(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    added, calls, arm = _flown(tmp_path, Rewriter(skew=True))
    assert calls
    assert digest(added) != ADDED_DIGEST
    assert arm.counts["rows_changed"] > 0
    (line,) = [m for m in caplog.messages if m.startswith("perturb shapes ")]
    assert float(line.split(" difference=")[1].split(" ")[0]) < 0, line


def test_the_reading_logs_perturbed_and_unperturbed_column_lines(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    added, _, arm = _flown(tmp_path, Rewriter(skew=True))
    train, request, direct = varied_arm()
    drawn = direct(train, request).added
    reference = label_rows(train, UCI_FRAME, POSITIVE)
    lines = [m for m in caplog.messages if m.startswith("perturb columns ")]
    assert [line.split(" kind=")[1].split(" ")[0] for line in lines] == ["perturbed", "unperturbed"]
    for line, rows in zip(lines, (added, drawn), strict=True):
        head, _, tail = line.partition(" kind=")
        assert head == f"perturb columns holdout=-1 arm={arm.name}"
        tokens = tail.split(" ")[1:]
        assert [token.split("=")[0] for token in tokens] == list(FIELDS)
        for token in tokens:
            name, value = token.split("=")
            report = quality_report(reference, rows, frame=UCI_FRAME, columns=[name])
            shape = float(report.get_properties().set_index("Property")["Score"]["Column Shapes"])
            assert value == f"{shape:.6f}", name


def test_incoherence_counts_rows_breaking_any_of_the_three_rules():
    no, steady, up = DRUG_CODES["No"], DRUG_CODES["Steady"], DRUG_CODES["Up"]
    rows = pd.DataFrame(
        [
            {
                **dict.fromkeys(DRUGS, no),
                "insulin": steady,
                "change": CHANGE_CODES["No"],
                "diabetesMed": DIABETES_MED_CODES["No"],
            },
            {
                **dict.fromkeys(DRUGS, no),
                "change": CHANGE_CODES["Ch"],
                "diabetesMed": DIABETES_MED_CODES["No"],
            },
            {
                **dict.fromkeys(DRUGS, no),
                "metformin": up,
                "change": CHANGE_CODES["No"],
                "diabetesMed": DIABETES_MED_CODES["Yes"],
            },
            {
                **dict.fromkeys(DRUGS, no),
                "metformin": up,
                "change": CHANGE_CODES["Ch"],
                "diabetesMed": DIABETES_MED_CODES["Yes"],
            },
        ]
    )
    assert incoherent(rows) == pytest.approx(0.75)
    assert incoherent(rows.iloc[[3]]) == pytest.approx(0.0)


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
    (shapes,) = [
        m for m in caplog.messages if m.startswith(f"perturb shapes holdout=0 arm={PERTURBED_ARM} ")
    ]
    assert " incoherent=none unperturbed_incoherent=none rows_changed=" in shapes


def _results(folder: Path, arm: str) -> list[tuple[int, int | None, str | None]]:
    """Each of `arm`'s candidates lines' holdout, call and `perturb` result, in file order."""
    found = read((folder / "holdout-candidates.jsonl").read_bytes())
    return [(line.holdout, line.call, line.perturb) for line in found.candidates if line.arm == arm]


def test_every_rewrite_line_carries_its_perturb_result(tmp_path, caplog):
    folder, _ = perturbed_folder(tmp_path, caplog, Rewriter(invalid=frozenset({1})))
    results = _results(folder, PERTURBED_ARM)
    assert len(results) == 2 * 28
    assert {result for _, _, result in results} == {"perturbed", "unchanged", "refused_keys"}
    found = read((folder / "holdout-candidates.jsonl").read_bytes()).candidates
    assert {line.perturb for line in found if line.arm not in {PERTURBED_ARM, PRAM_ARM}} == {None}


def test_the_refused_row_reads_refused_keys(tmp_path, caplog):
    folder, _ = perturbed_folder(tmp_path, caplog, Rewriter(invalid=frozenset({1})))
    results = _results(folder, PERTURBED_ARM)
    refused = [line for line in results if line[2] not in {"perturbed", "unchanged"}]
    assert refused == [(0, None, "refused_keys")]


def test_rewrite_results_count_as_the_holdout_line_does(tmp_path, caplog):
    folder, _ = perturbed_folder(tmp_path, caplog, Rewriter(invalid=frozenset({1})))
    results = _results(folder, PERTURBED_ARM)
    for holdout in {holdout for holdout, _, _ in results}:
        counts = Counter(result for at, _, result in results if at == holdout)
        (line,) = [m for m in caplog.messages if m.startswith(f"perturb holdout={holdout} rows=")]
        logged = dict(field.split("=") for field in line.split()[2:])
        assert {name: int(count) for name, count in logged.items() if int(count)} == {
            "rows": sum(counts.values()),
            **counts,
        }


def test_every_perturbed_row_carries_its_call(tmp_path, caplog):
    folder, _ = perturbed_folder(tmp_path, caplog, Rewriter(skew=True))
    found = read((folder / "holdout-candidates.jsonl").read_bytes())
    mine = [line for line in found.candidates if line.arm == PERTURBED_ARM]
    assert len(mine) == 2 * 28
    assert None not in {line.call for line in mine}
    shapes = [m for m in caplog.messages if m.startswith("perturb shapes holdout=")]
    mine_shapes = [m for m in shapes if f" arm={PERTURBED_ARM} " in m]
    assert len(mine_shapes) == 2
    assert all(float(m.split(" difference=")[1].split(" ")[0]) < 0 for m in mine_shapes), (
        mine_shapes
    )


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


def test_rewrite_rows_changed_is_above_zero_exactly_when_values_changed_is(tmp_path, caplog):
    perturbed_folder(tmp_path, caplog, Rewriter(invalid=frozenset({1})))
    lines = [
        m
        for m in caplog.messages
        if m.startswith("perturb shapes holdout=") and f" arm={PERTURBED_ARM} " in m
    ]
    assert lines
    for line in lines:
        rows = int(line.split(" rows_changed=")[1].split(" ")[0])
        values = int(line.split(" values_changed=")[1])
        assert (rows > 0) == (values > 0), line
