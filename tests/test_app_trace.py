"""The app's record traces through AppTest: a run folder's `trace.jsonl` and a demo fill's rows."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd
from streamlit.testing.v1 import AppTest

from bko.app.page import OUTCOME_KEY
from bko.app.steps import Outcome, analyze, fill_and_score
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.trace import UNTRACED, Trace
from tests._app_fixtures import analyzed, app_context, frame_with, launch, never, settled
from tests._generate_fixtures import ScriptedSend, reply
from tests._ontology_fixtures import RoutingOpener
from tests._validate_fixtures import planted


NAME: Final = "e1-3-trace"
"""The trace folder's name, as `docs/DEVELOPMENT.md` (Evaluation, `trace`) writes one."""

CELL: Final = "<55/female/1"
SUMMARY: Final = "Summary of the training records in this cell: 4 records, ages 41 to 52."
FINDINGS: Final = ("Angina pectoris (HP:9000003) implies coronary stenosis (MONDO:9000004).",)


def _concept(concept: str, label: str, share: float, *, present: bool) -> dict[str, object]:
    return {
        "concept": concept,
        "label": label,
        "present": present,
        "share": share,
        "cell_prior": share / 2,
        "group_prior": share * 2,
        "cell_records": 4,
        "cell_holders": 1,
        "group_records": 10,
        "group_holders": 3,
        "group_weight": 20,
        "snodes": [
            {
                "head": [concept, "a"],
                "prob": share,
                "tail": [[f"__Source__[{concept}]", "uci/target=1"]],
            }
        ],
    }


CONCEPTS: Final = (
    _concept("HP:9000003", "Angina pectoris", 0.25, present=True),
    _concept("MONDO:9000004", "coronary stenosis", 0.125, present=False),
)


def _age(holdout: int, record_id: int) -> int:
    return 41 + record_id + 10 * holdout


def direct(holdout: int, record_id: int) -> Trace:
    """Record `record_id` of a `bko-direct` fill on `holdout`, its age naming both."""
    return Trace(
        holdout,
        "bko-direct",
        "gap",
        record_id,
        CELL,
        {"age": _age(holdout, record_id), "sex": 0, "target": 1},
        KEPT,
        (),
        {"attempt": 1, "cell": CELL, "quota": 3, "rows": 3, "seed": 7},
        CONCEPTS,
        {"age": 100 + record_id, "sex": 200 + record_id, "target": 300 + record_id},
        ("sex",),
        ("HP:9000003", "HP:9000001"),
    )


def messages(holdout: int, record_id: int, *, failed: tuple[str, ...] = ()) -> Trace:
    """Record `record_id` of a `bko` fill on `holdout`, with its call's prompt sections."""
    shown = {
        "opening": f"Generate 2 records for the cell {CELL}.",
        "bounds": "Keep each field inside these bounds: age 18 to 120.",
        "coherence": False,
        "basis": "ontology",
        "findings": list(FINDINGS),
        "profile": None,
        "summary": SUMMARY,
        "prescribed": None,
        "shown": None,
    }
    return Trace(
        holdout,
        "bko",
        "gap",
        record_id,
        CELL,
        {"age": 50 + record_id, "sex": 0, "target": 1},
        KEPT,
        failed,
        {"attempt": 1, "cell": CELL, "guidance": "ontology", "sections": shown},
        call=3 + record_id,
    )


LINES: Final = (
    *(direct(0, k) for k in range(3)),
    *(direct(1, k) for k in range(2)),
    *(messages(0, k) for k in range(2)),
    *(messages(1, k) for k in range(2)),
)
"""Two arms on two holdouts; `bko-direct` on holdout 0 has three records."""


def _write(folder: Path, lines: Sequence[Trace] | str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "".join(line.line() for line in lines)
    path = folder / "trace.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def _opened(tmp_path: Path, lines: Sequence[Trace] | str) -> tuple[AppTest, RoutingOpener]:
    """The Earlier runs view over one trace folder, opened; the folder is outside `every_kind`."""
    runs = tmp_path / "runs"
    _write(runs / NAME, lines)
    context, opener = app_context(tmp_path, send=never, runs_dir=runs)
    at = settled(launch(context).radio(key="bko.view").set_value("Earlier runs").run())
    return settled(at.selectbox(key="bko.run").select(NAME).run()), opener


def _picked(at: AppTest, arm: str, holdout: int, record: int = 0) -> AppTest:
    at = settled(at.selectbox(key=f"bko.trace.arm.{NAME}").select(arm).run())
    at = settled(at.selectbox(key=f"bko.trace.holdout.{NAME}").select(holdout).run())
    return settled(at.number_input(key=f"bko.trace.record.{NAME}").set_value(record).run())


def _fields(at: AppTest) -> dict[str, str]:
    frame = frame_with(at, {"field", "value"})
    return dict(zip(frame["field"], frame["value"], strict=True))


def test_a_trace_file_gets_a_record_trace_section(tmp_path):
    at, _ = _opened(tmp_path, LINES)
    assert "Record trace" in {subheader.value for subheader in at.subheader}
    assert not at.error


def test_the_pickers_list_the_files_arms_and_holdouts(tmp_path):
    at, _ = _opened(tmp_path, LINES)
    assert list(at.selectbox(key=f"bko.trace.arm.{NAME}").options) == ["bko-direct", "bko"]
    assert list(at.selectbox(key=f"bko.trace.holdout.{NAME}").options) == ["0", "1"]


def test_the_record_picker_spans_the_fill_and_shows_the_chosen_record(tmp_path):
    at, _ = _opened(tmp_path, LINES)
    picker = at.number_input(key=f"bko.trace.record.{NAME}")
    assert (picker.min, picker.max) == (0, 2)
    at = settled(picker.set_value(1).run())
    assert _fields(at) == {"age": str(_age(0, 1)), "sex": "0", "target": "1"}


def test_a_direct_record_shows_its_concepts_sources_and_fallback(tmp_path):
    at = _picked(_opened(tmp_path, LINES)[0], "bko-direct", 0, 2)
    concepts = frame_with(at, {"concept", "label", "share", "cell_prior", "group_prior"})
    assert concepts["concept"].tolist() == [c["concept"] for c in CONCEPTS]
    assert concepts["share"].tolist() == [c["share"] for c in CONCEPTS]
    assert concepts["label"].tolist() == ["Angina pectoris", "coronary stenosis"]
    snodes = frame_with(at, {"head", "tail", "prob"})
    assert snodes["head"].tolist() == ["HP:9000003 = a", "MONDO:9000004 = a"]
    sources = frame_with(at, {"field", "source_record"})
    assert dict(zip(sources["field"], sources["source_record"], strict=True)) == {
        "age": 102,
        "sex": 202,
        "target": 302,
    }
    assert sources.loc[sources["fallback"], "field"].tolist() == ["sex"]
    assert {"Request", "Concepts", "Sources", "Validation"} <= {s.value for s in at.subheader}


def test_a_messages_record_shows_its_calls_gap_statements_and_stored_summary(tmp_path):
    at = _picked(_opened(tmp_path, LINES)[0], "bko", 1, 1)
    assert SUMMARY in [block.value for block in at.code]
    assert frame_with(at, {"statement"})["statement"].tolist() == list(FINDINGS)
    origin = frame_with(at, {"call", "cell"})
    assert origin[["call", "cell"]].values.tolist() == [[4, CELL]]
    assert {"Request", "Gap statements", "Prompt", "Validation"} <= {s.value for s in at.subheader}
    assert "Concepts" not in {s.value for s in at.subheader}


def test_the_validation_section_shows_the_outcome_and_its_failed_checks(tmp_path):
    lines = [messages(0, 0), messages(0, 1, failed=("range:age", "code:sex"))]
    at, _ = _opened(tmp_path, lines)
    at = settled(at.number_input(key=f"bko.trace.record.{NAME}").set_value(1).run())
    validation = frame_with(at, {"outcome", "failed"})
    assert validation.values.tolist() == [[KEPT, "range:age, code:sex"]]


def test_a_file_without_trace_fields_says_so_and_still_shows_the_record(tmp_path):
    bare = {"holdout": 0, "arm": "bko", "fill": "gap", "record_id": 0, "record": {"age": 44}}
    untraced = dataclasses.replace(messages(0, 1), request=None, untraced=UNTRACED)
    at, _ = _opened(tmp_path, json.dumps(bare) + "\n" + untraced.line())
    assert any("no trace is stored in this file" in info.value for info in at.info)
    assert _fields(at) == {"age": "44"}
    assert not at.error
    at = settled(at.number_input(key=f"bko.trace.record.{NAME}").set_value(1).run())
    assert UNTRACED in [block.value for block in at.code]
    assert not at.error


def test_a_truncated_file_shows_its_error_in_place(tmp_path):
    text = "".join(line.line() for line in LINES)
    at, _ = _opened(tmp_path, text[: len(text) - 40])
    assert [error.value for error in at.error] == ["trace.jsonl could not be shown."]
    assert any(block.value.startswith("JSONDecodeError: ") for block in at.code)


def test_the_view_makes_no_call_and_fetches_nothing(tmp_path):
    at, opener = _opened(tmp_path, LINES)
    at = _picked(at, "bko", 0, 1)
    assert opener.calls == []
    assert not (tmp_path / "data").exists()


def test_the_pick_survives_reruns_and_a_rewrite_rebuilds_the_index(tmp_path):
    at = _picked(_opened(tmp_path, LINES)[0], "bko-direct", 1, 1)
    at = settled(settled(at.run()).run())
    assert at.selectbox(key=f"bko.trace.arm.{NAME}").value == "bko-direct"
    assert at.selectbox(key=f"bko.trace.holdout.{NAME}").value == 1
    assert at.number_input(key=f"bko.trace.record.{NAME}").value == 1
    assert _fields(at)["age"] == str(_age(1, 1))
    path = tmp_path / "runs" / NAME / "trace.jsonl"
    before = path.stat().st_mtime_ns
    changed = [
        dataclasses.replace(line, record={"age": 99, "sex": 0, "target": 1}) for line in LINES
    ]
    _write(path.parent, changed)
    os.utime(path, ns=(before + 10**9, before + 10**9))
    at = settled(at.run())
    assert _fields(at)["age"] == "99"


def varied() -> ScriptedSend:
    """The default split's two calls, 12 without the condition then 16 with it, no two records of
    one call alike: each differs by age, inside the training range of 40 to 63, or by chest pain."""
    return ScriptedSend(
        reply([planted(age=40 + i, sex=0, cp=4, target=0) for i in range(12)]),
        reply([planted(age=40 + i % 8, sex=0, cp=1 + 3 * (i // 8), target=1) for i in range(16)]),
    )


def _key(record: Mapping[Any, object]) -> tuple[int, ...]:
    return tuple(int(str(record[name]).split(".")[0]) for name in ("age", "cp", "target"))


def _row(outcome: Outcome, k: int) -> dict[str, float]:
    row = outcome.baseline.added.iloc[k]
    return {str(key): float(value) for key, value in row.items()}


def _kept_line(outcome: Outcome, k: int) -> list[object]:
    """The call and cell of the kept candidate line in the fill's candidates file that added row
    `k` came from."""
    lines = read((outcome.folder / "holdout-candidates.jsonl").read_bytes()).candidates
    wanted = _key(outcome.baseline.added.iloc[k].to_dict())
    (line,) = [
        line
        for line in lines
        if line.outcome == KEPT
        and isinstance(line.candidate, dict)
        and _key(line.candidate) == wanted
    ]
    return [line.call, line.cell]


def _trace_frames(at: AppTest) -> tuple[dict[str, float], list[list[object]]]:
    """The shown record's values as numbers, and its call and cell."""
    origin = frame_with(at, {"call", "cell"})
    fields = {key: float(value) for key, value in _fields(at).items()}
    return fields, origin[["call", "cell"]].values.tolist()


def _paid(tmp_path: Path) -> tuple[AppTest, ScriptedSend, Outcome]:
    send = varied()
    at = analyzed(app_context(tmp_path, send=lambda: send)[0])
    at = settled(at.selectbox(key="bko.method").select("bko").run())
    at = settled(at.checkbox(key="bko.confirm.bko.0").check().run())
    at = settled(at.button(key="bko.fill").click().run())
    return at, send, at.session_state[OUTCOME_KEY].value


def test_a_paid_fill_picks_each_added_row_and_shows_its_kept_line(tmp_path):
    at, send, outcome = _paid(tmp_path)
    key = f"bko.trace.added.{outcome.folder.name}"
    assert len(at.selectbox(key=key).options) == len(outcome.baseline.added)
    calls = len(send.calls)
    for k in (1, len(outcome.baseline.added) - 1):
        at = settled(at.selectbox(key=key).select(k).run())
        fields, origin = _trace_frames(at)
        assert fields == _row(outcome, k)
        assert origin == [_kept_line(outcome, k)]
    at = settled(at.run())
    assert len(send.calls) == calls
    assert not at.error


def test_a_free_fill_shows_no_trace_picker(tmp_path):
    at = analyzed(app_context(tmp_path, send=never)[0])
    at = settled(at.selectbox(key="bko.method").select("smote").run())
    at = settled(at.button(key="bko.fill").click().run())
    assert not [box for box in at.selectbox if str(box.key).startswith("bko.trace.")]


def test_a_replayed_fill_shows_the_same_trace_without_a_call(tmp_path):
    send = varied()
    context, _ = app_context(tmp_path, send=lambda: send)
    recorded = fill_and_score(analyze(context), "bko", context)
    at = analyzed(dataclasses.replace(context, send=never))
    at = settled(at.selectbox(key="bko.method").select("bko").run())
    at = settled(at.radio(key="bko.answers").set_value("replay").run())
    at = settled(at.selectbox(key="bko.log").select(recorded.folder.name).run())
    at = settled(at.button(key="bko.fill").click().run())
    replayed: Outcome = at.session_state[OUTCOME_KEY].value
    assert replayed.replayed is True
    at = settled(at.selectbox(key=f"bko.trace.added.{replayed.folder.name}").select(2).run())
    fields, origin = _trace_frames(at)
    assert fields == _row(recorded, 2)
    assert origin == [_kept_line(recorded, 2)]


def test_the_download_carries_no_source_record_id(tmp_path):
    _, _, outcome = _paid(tmp_path)
    columns = pd.read_csv(outcome.folder / "training.csv", nrows=0).columns
    assert not [column for column in columns if "source" in column]
