"""`attribution`: each trace line's fields counted by what they can be attributed to."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.attribution import (
    ArmKind,
    Attribution,
    arm_kind,
    attribution,
    classify,
)
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, LOGIC_ARM, NOMAP_ARM, PERTURBED_ARM, PRAM_ARM
from tests.test_evaluate_trace import _scored
from tests.test_evaluate_trace_direct import _context, _trace


MAPPED = frozenset({"cp", "thal"})
"""The fields a mapping row's `when` clause reads in these hand-written lines."""

RECORD: Mapping[str, object] = {"age": 61, "sex": 1, "cp": 4, "thal": 7, "chol": 240, "num": 1}
SOURCES: Mapping[str, int] = {"age": 11, "cp": 11, "thal": 12, "chol": 13}
REQUEST: Mapping[str, object] = {"attempt": 0, "cell": "60-69/male/1", "quota": 2, "rows": 4}


def _direct(arm: str = DIRECT_ARM, **extra: object) -> dict[str, object]:
    return {
        "holdout": 0,
        "arm": arm,
        "fill": "gap",
        "record_id": 0,
        "record": dict(RECORD),
        "request": {**REQUEST, "seed": 5},
        "sources": dict(SOURCES),
        "untraced": None,
        **extra,
    }


def _messages(arm: str = "bko", *, untraced: str | None = None) -> dict[str, object]:
    request = None if untraced else {"attempt": 0, "cell": "60-69/male/1", "guidance": "summary"}
    return {
        "holdout": 0,
        "arm": arm,
        "fill": "gap",
        "record_id": 0,
        "record": dict(RECORD),
        "request": request,
        "sources": {},
        "call": 3,
        "untraced": untraced,
    }


def _baseline(arm: str = "smote") -> dict[str, object]:
    return {
        "holdout": 0,
        "arm": arm,
        "fill": "gap",
        "record_id": None,
        "record": None,
        "request": {"method": arm, "seed": 5},
        "sources": {},
        "untraced": None,
    }


def _write(path: Path, lines: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def test_a_direct_record_is_knowledge_on_mapped_fields_and_donor_elsewhere():
    line = _direct()
    assert arm_kind(line) is ArmKind.DIRECT
    assert classify(line, mapped=MAPPED) == {
        "age": Attribution.DONOR,
        "sex": Attribution.SEED,
        "cp": Attribution.KNOWLEDGE,
        "thal": Attribution.KNOWLEDGE,
        "chol": Attribution.DONOR,
        "num": Attribution.SEED,
    }


def test_a_replica_fill_of_a_direct_arm_reads_its_arms_spec():
    found = classify(_direct(f"{NOMAP_ARM}-replica"), mapped=MAPPED)
    assert Attribution.KNOWLEDGE not in found.values()


def test_a_nomap_record_is_donor_and_never_knowledge():
    found = classify(_direct(NOMAP_ARM), mapped=MAPPED)
    assert found["cp"] is found["thal"] is found["chol"] is Attribution.DONOR
    assert Attribution.KNOWLEDGE not in found.values()
    assert classify(_direct(LOGIC_ARM), mapped=MAPPED)["cp"] is Attribution.KNOWLEDGE


def test_a_messages_record_is_prompt_on_every_field():
    line = _messages()
    assert arm_kind(line) is ArmKind.MESSAGES
    assert set(classify(line, mapped=MAPPED).values()) == {Attribution.PROMPT}
    assert set(RECORD) == set(classify(line, mapped=MAPPED))


def test_a_pram_changed_field_is_method_seed_and_a_rewritten_one_is_prompt():
    perturb = {"result": "changed", "fields": ["chol"], "drawn": {"chol": 250}}
    pram = _direct(PRAM_ARM, perturb=perturb)
    assert arm_kind(pram) is ArmKind.PERTURBATION
    found = classify(pram, mapped=MAPPED)
    assert found["chol"] is Attribution.SEED
    assert found["cp"] is Attribution.KNOWLEDGE
    assert found["age"] is Attribution.DONOR
    rewritten = classify(_direct(PERTURBED_ARM, perturb=perturb), mapped=MAPPED)
    assert rewritten["chol"] is Attribution.PROMPT
    assert rewritten["cp"] is Attribution.KNOWLEDGE


def test_an_untraced_record_is_method_seed_and_counted_in_the_log(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    line = _messages(untraced="no calls.jsonl")
    assert arm_kind(line) is ArmKind.MESSAGES
    assert set(classify(line, mapped=MAPPED).values()) == {Attribution.SEED}
    path = _write(tmp_path / "trace.jsonl", [line, _messages()])
    text = attribution([path], mapped=MAPPED)
    assert "0,bko,messages,method-seed,field,6\n" in text
    assert "0,bko,messages,prompt,field,6\n" in text
    (logged,) = [m for m in caplog.messages if m.startswith("attribution trace=")]
    assert logged.endswith(" lines=2 untraced=1")


def test_a_baseline_fill_counts_once_as_method_seed(tmp_path):
    line = _baseline()
    assert arm_kind(line) is ArmKind.BASELINE
    assert classify(line, mapped=MAPPED) == {}
    text = attribution([_write(tmp_path / "trace.jsonl", [line])], mapped=MAPPED)
    assert text.splitlines()[1:] == ["0,smote,baseline,method-seed,fill,1"]


@pytest.mark.parametrize(
    "line",
    [
        {"holdout": 0, "arm": "x"},
        {"holdout": 0, "arm": "x", "record": [1]},
        {"holdout": 0, "arm": "x", "record": {"a": 1}, "perturb": []},
        {"holdout": 0, "arm": "x", "record": {"a": 1}, "request": "no"},
    ],
)
def test_any_other_line_shape_is_refused(line):
    with pytest.raises(EvaluationError):
        arm_kind(line)


def test_a_direct_line_of_an_arm_outside_the_direct_arms_is_refused():
    with pytest.raises(EvaluationError, match="not a direct arm"):
        classify(_direct("bko"), mapped=MAPPED)


def test_a_line_that_is_not_a_json_object_is_refused(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="not a JSON object"):
        attribution([path], mapped=MAPPED)
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="not a JSON object"):
        attribution([path], mapped=MAPPED)


def test_one_fill_of_two_kinds_is_refused(tmp_path):
    path = _write(tmp_path / "trace.jsonl", [_direct("bko"), _messages()])
    with pytest.raises(EvaluationError):
        attribution([path], mapped=MAPPED)


GOLDEN = (
    "holdout,arm,kind,class,unit,count\n"
    "0,bko-direct,direct,knowledge,field,4\n"
    "0,bko-direct,direct,donor,field,4\n"
    "0,bko-direct,direct,prompt,field,0\n"
    "0,bko-direct,direct,method-seed,field,4\n"
    "0,bko,messages,knowledge,field,0\n"
    "0,bko,messages,donor,field,0\n"
    "0,bko,messages,prompt,field,6\n"
    "0,bko,messages,method-seed,field,0\n"
    "0,smote,baseline,method-seed,fill,1\n"
)


def test_the_csv_is_the_golden_string(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    direct = _write(tmp_path / "direct.jsonl", [_direct(), _direct()])
    messages = _write(tmp_path / "messages.jsonl", [_messages()])
    baseline = _write(tmp_path / "baseline.jsonl", [_baseline()])
    text = attribution([direct, messages, baseline], mapped=MAPPED)
    assert text == GOLDEN
    closing = caplog.messages[-1]
    assert closing.startswith("attribution rows=9 sha256=")
    assert closing.endswith(" files=3")


def test_the_command_reads_trace_files_and_makes_no_api_call(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    paths = [
        _write(tmp_path / "direct.jsonl", [_direct()]),
        _write(tmp_path / "baseline.jsonl", [_baseline()]),
    ]
    out = io.StringIO()
    argv = ["attribution", "g2", "--trace", *map(str, paths)]
    assert main(argv, context=_context(), out=out) == 0
    rows = out.getvalue().splitlines()
    assert rows[0] == "holdout,arm,kind,class,unit,count"
    assert rows[-1] == "0,smote,baseline,method-seed,fill,1"
    assert caplog.messages[0].startswith("versions commit=")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("7\n", encoding="utf-8")
    assert main(["attribution", "g2", "--trace", str(bad)], context=_context()) == 2


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `g2` folder of `bko-direct`, `bko-logic` and `none`, its data root beside it."""
    base = tmp_path_factory.mktemp("attribution")
    folder = base / "run"
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    return folder


def test_every_field_of_every_record_gets_exactly_one_class(scored, tmp_path):
    data = scored.parent / "data"
    paths, lines = [], []
    for k, arm in enumerate(("none", DIRECT_ARM, LOGIC_ARM)):
        code, found = _trace(scored, data, "--arm", arm)
        assert code == 0
        assert found
        paths.append(_write(tmp_path / f"trace-{k}.jsonl", found))
        lines.extend(found)
    out = io.StringIO()
    argv = ["attribution", "g2", "--trace", *map(str, paths)]
    assert main(argv, context=_context(), out=out) == 0
    counted: dict[tuple[str, str], int] = {}
    for row in out.getvalue().splitlines()[1:]:
        _, arm, _, kind, unit, count = row.split(",")
        counted[arm, unit] = counted.get((arm, unit), 0) + int(count)
        if arm == DIRECT_ARM and kind == Attribution.KNOWLEDGE:
            assert int(count) > 0
    for arm in (DIRECT_ARM, LOGIC_ARM):
        records = [line["record"] for line in lines if line["arm"] == arm]
        assert counted[arm, "field"] == sum(len(record) for record in records)
    assert counted["none", "fill"] == 1
