"""`attribution`: each trace line's fields counted by what they can be attributed to."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import re
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
    mapped_fields,
    run_attribution,
)
from bko.evaluate.data import Dataset
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, LOGIC_ARM, NOMAP_ARM, PERTURBED_ARM, PRAM_ARM
from tests.test_evaluate_trace import _scored
from tests.test_evaluate_trace_direct import _context, _trace


MAPPED = frozenset({"cp", "thal"})
"""The fields a mapping row's `when` clause reads in these hand-written lines."""

RECORD: Mapping[str, object] = {
    "age": 61,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 7,
    "target": 1,
}
"""A record in the UCI frame's columns, which `g2` reads."""

COLUMNS = frozenset(RECORD)
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


def _messages(
    arm: str = "bko", *, untraced: str | None = None, record_id: int = 0
) -> dict[str, object]:
    request = None if untraced else {"attempt": 0, "cell": "60-69/male/1", "guidance": "summary"}
    return {
        "holdout": 0,
        "arm": arm,
        "fill": "gap",
        "record_id": record_id,
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


def _attribution(paths: list[Path]) -> str:
    return attribution(paths, run="g2", mapped=MAPPED, columns=COLUMNS)


def test_a_direct_record_is_knowledge_on_mapped_fields_and_donor_elsewhere():
    line = _direct()
    assert arm_kind(line) is ArmKind.DIRECT
    assert classify(line, mapped=MAPPED) == {
        "age": Attribution.DONOR,
        "cp": Attribution.KNOWLEDGE,
        "thal": Attribution.KNOWLEDGE,
        "chol": Attribution.DONOR,
        **dict.fromkeys(COLUMNS - {"age", "cp", "thal", "chol"}, Attribution.SEED),
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


def test_a_changed_mapped_field_takes_the_changed_fields_class_not_knowledge():
    perturb = {"result": "changed", "fields": ["cp"], "drawn": {"cp": 3}}
    assert classify(_direct(PRAM_ARM, perturb=perturb), mapped=MAPPED)["cp"] is Attribution.SEED
    rewritten = classify(_direct(PERTURBED_ARM, perturb=perturb), mapped=MAPPED)
    assert rewritten["cp"] is Attribution.PROMPT
    assert rewritten["thal"] is Attribution.KNOWLEDGE


def test_a_perturbation_arm_outside_the_direct_arms_logs_its_spec_once_per_fill(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    perturb = {"result": "perturbed", "fields": ["chol"], "drawn": {"chol": 250}}
    lines = [_direct(PERTURBED_ARM, perturb=perturb, record_id=k) for k in range(2)]
    pram = _direct(PRAM_ARM, perturb=perturb)
    _attribution([_write(tmp_path / "trace.jsonl", [*lines, pram])])
    logged = [m for m in caplog.messages if m.startswith("attribution spec")]
    assert logged == [
        (
            'attribution spec={"presence":"shares","mapped":true,"min_donors":1,"reuse_cap":null,'
            '"tree":false}'
            f" holdout=0 arm={PERTURBED_ARM}"
        )
    ]


def test_an_untraced_record_is_method_seed_and_counted_in_the_log(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    line = _messages(untraced="no calls.jsonl")
    assert arm_kind(line) is ArmKind.MESSAGES
    assert set(classify(line, mapped=MAPPED).values()) == {Attribution.SEED}
    path = _write(tmp_path / "trace.jsonl", [line, _messages(record_id=1)])
    text = _attribution([path])
    assert "0,bko,messages,method-seed,field,14\n" in text
    assert "0,bko,messages,prompt,field,14\n" in text
    (logged,) = [m for m in caplog.messages if m.startswith("attribution trace=")]
    assert logged.endswith(" lines=2 untraced=1")


def test_a_baseline_fill_counts_once_as_method_seed(tmp_path):
    line = _baseline()
    assert arm_kind(line) is ArmKind.BASELINE
    assert classify(line, mapped=MAPPED) == {}
    text = _attribution([_write(tmp_path / "trace.jsonl", [line])])
    assert text.splitlines()[1:] == ["0,smote,baseline,method-seed,fill,1"]


def test_a_repeated_baseline_line_is_refused(tmp_path):
    path = _write(tmp_path / "trace.jsonl", [_baseline(), _baseline()])
    with pytest.raises(EvaluationError, match="smote record None recurs"):
        _attribution([path])


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


@pytest.mark.parametrize("fields", ["x", [1], ["chol", None]])
def test_perturb_fields_that_are_not_a_list_of_strings_are_refused(fields):
    with pytest.raises(EvaluationError, match="string fields"):
        arm_kind(_direct(PRAM_ARM, perturb={"result": "changed", "fields": fields}))


@pytest.mark.parametrize("arm", [DIRECT_ARM, "bko", f"{DIRECT_ARM}-replica"])
def test_a_perturb_key_under_an_arm_that_is_not_a_perturbation_arm_is_refused(arm):
    with pytest.raises(EvaluationError, match="not a perturbation arm"):
        arm_kind(_direct(arm, perturb={"result": "changed", "fields": ["chol"]}))


def test_a_perturb_key_under_a_perturbation_arms_replica_is_read():
    line = _direct(f"{PRAM_ARM}-replica", perturb={"result": "changed", "fields": ["chol"]})
    assert arm_kind(line) is ArmKind.PERTURBATION


@pytest.mark.parametrize("sources", [[1], "age", None])
def test_sources_that_are_not_an_object_are_refused(sources):
    with pytest.raises(EvaluationError, match="sources is not a JSON object"):
        arm_kind(_direct(sources=sources))


def test_a_direct_line_of_an_arm_outside_the_direct_arms_is_refused():
    with pytest.raises(EvaluationError, match="not a direct arm"):
        classify(_direct("bko"), mapped=MAPPED)


def test_a_line_that_is_not_a_json_object_is_refused_at_its_file_and_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match=f"^{re.escape(str(path))}:1: .*not a JSON object"):
        _attribution([path])
    path.write_text(json.dumps(_direct()) + "\n\n{not json\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match=f"^{re.escape(str(path))}:3: .*not a JSON object"):
        _attribution([path])


def test_a_file_that_is_not_utf8_is_refused(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(json.dumps(_direct()).encode() + b"\n\xff\n")
    with pytest.raises(EvaluationError, match=":2: a line is not UTF-8"):
        _attribution([path])


def test_lines_split_on_newline_alone(tmp_path):
    path = tmp_path / "trace.jsonl"
    line = _messages(untraced="no\u2028calls.jsonl")
    path.write_text(json.dumps(line, ensure_ascii=False) + "\n", encoding="utf-8")
    assert "0,bko,messages,method-seed,field,14\n" in _attribution([path])


@pytest.mark.parametrize(
    "line",
    [
        {**_baseline(), "holdout": True},
        {**_baseline(), "holdout": "0"},
        {**_baseline(), "arm": 3},
        _baseline("=HYPERLINK(1)"),
        _baseline("Smote"),
        _baseline("smote replica"),
        {**_baseline(), "record_id": "0"},
        {**_baseline(), "record_id": False},
    ],
)
def test_a_line_naming_no_holdout_arm_or_record_id_of_their_shapes_is_refused(tmp_path, line):
    path = _write(tmp_path / "trace.jsonl", [line])
    with pytest.raises(EvaluationError, match=f"^{re.escape(str(path))}:1: "):
        _attribution([path])


def test_a_record_named_twice_across_the_files_is_refused(tmp_path):
    path = _write(tmp_path / "trace.jsonl", [_direct()])
    with pytest.raises(EvaluationError, match=f"{DIRECT_ARM} record 0 recurs"):
        _attribution([path, path])
    other = _write(tmp_path / "other.jsonl", [_direct()])
    with pytest.raises(EvaluationError, match=f"^{re.escape(str(other))}:1: "):
        _attribution([path, other])
    parts = [_write(tmp_path / f"part-{k}.jsonl", [_direct(record_id=k)]) for k in range(2)]
    assert "0,bko-direct,direct,knowledge,field,4\n" in _attribution(parts)


def test_a_record_whose_fields_are_not_the_runs_columns_is_refused(tmp_path):
    record = {name: value for name, value in RECORD.items() if name != "target"}
    path = _write(tmp_path / "trace.jsonl", [_direct(record={**record, "num": 1})])
    with pytest.raises(
        EvaluationError, match="not the run g2's columns: missing target, extra num"
    ):
        _attribution([path])


def test_one_fill_of_two_kinds_is_refused(tmp_path):
    path = _write(tmp_path / "trace.jsonl", [_direct(), _messages(DIRECT_ARM, record_id=1)])
    with pytest.raises(EvaluationError, match="bko-direct fill has direct and messages lines"):
        _attribution([path])


GOLDEN = (
    "holdout,arm,kind,class,unit,count\n"
    "0,bko-direct,direct,knowledge,field,4\n"
    "0,bko-direct,direct,donor,field,4\n"
    "0,bko-direct,direct,prompt,field,0\n"
    "0,bko-direct,direct,method-seed,field,20\n"
    "0,bko,messages,knowledge,field,0\n"
    "0,bko,messages,donor,field,0\n"
    "0,bko,messages,prompt,field,14\n"
    "0,bko,messages,method-seed,field,0\n"
    "0,smote,baseline,method-seed,fill,1\n"
)


def test_the_csv_is_the_golden_string(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    direct = _write(tmp_path / "direct.jsonl", [_direct(), _direct(record_id=1)])
    messages = _write(tmp_path / "messages.jsonl", [_messages()])
    baseline = _write(tmp_path / "baseline.jsonl", [_baseline()])
    text = _attribution([direct, messages, baseline])
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


def test_the_mapped_fields_leave_out_the_label():
    assert mapped_fields(Dataset.UCI, _context()) == frozenset({"cp"})


@pytest.mark.parametrize(
    "dataset", [Dataset.DIABETES_CODES, Dataset.TABLESHIFT_ID, Dataset.TABLESHIFT_OOD]
)
def test_attribution_refuses_a_dataset_whose_frame_needs_its_records(tmp_path, dataset):
    named = dataclasses.replace(_context(), dataset=dataset)
    why = f"the {dataset} frame's codebook is built from its loaded records"
    with pytest.raises(EvaluationError, match=why):
        mapped_fields(dataset, named)
    args = argparse.Namespace(traces=[tmp_path / "absent.jsonl"], run="g2")
    with pytest.raises(EvaluationError, match=why):
        run_attribution(args, named)


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
    mapped = mapped_fields(Dataset.UCI, _context())
    for line in lines:
        if line["arm"] == DIRECT_ARM:
            assert classify(line, mapped=mapped)["target"] is Attribution.SEED
