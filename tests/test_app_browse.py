"""The Earlier runs view through AppTest over run folders under `tmp_path`."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from bko.evaluate.scores import render
from bko.generate.candidates import render_lines
from tests._app_fixtures import analyzed, app_context, launch, never, settled
from tests._fusion_fixtures import young
from tests._runs_fixtures import DEMO, SCORES, every_kind


NAMES = [DEMO, "c1-2", "c1-2-closure", "e1-3", "e2-5", "r1-2"]

SECTIONS = {
    DEMO: {"Means over the holdouts", "Training set"},
    "c1-2": {"Call log totals", "Candidates"},
    "c1-2-closure": {"Gaps before and after the join"},
    "e1-3": {"Call log totals", "Means over the holdouts", "E1's criteria", "Validity"},
    "e2-5": {"UCI gap listing", "E2 gap readings"},
    "r1-2": {"Reproduction"},
}


def _earlier(at: AppTest) -> AppTest:
    return settled(at.radio(key="bko.view").set_value("Earlier runs").run())


def _opened(tmp_path: Path, runs: Path, name: str) -> AppTest:
    at = _earlier(launch(app_context(tmp_path, send=never, runs_dir=runs)[0]))
    return settled(at.selectbox(key="bko.run").select(name).run())


def test_the_view_switch_defaults_to_the_demo(tmp_path):
    at = launch(app_context(tmp_path, send=never, runs_dir=every_kind(tmp_path / "runs"))[0])
    assert at.radio(key="bko.view").value == "Demo"
    assert "1. Dataset" in [header.value for header in at.header]


def test_earlier_runs_lists_every_run_folder(tmp_path):
    at = launch(app_context(tmp_path, send=never, runs_dir=every_kind(tmp_path / "runs"))[0])
    at = _earlier(at)
    assert list(at.selectbox(key="bko.run").options) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_each_role_renders_its_section(tmp_path, name):
    runs = every_kind(tmp_path / "runs")
    at = _opened(tmp_path, runs, name)
    assert {subheader.value for subheader in at.subheader} == SECTIONS[name]
    roles = next(
        frame.value for frame in at.dataframe if list(frame.value.columns) == ["role", "bytes"]
    )
    present = [path.name for path in sorted((runs / name).iterdir()) if path.name != "stray.txt"]
    assert sorted(roles["role"]) == sorted(present)
    assert not at.error


def test_no_run_folder_shows_a_message(tmp_path):
    at = _earlier(launch(app_context(tmp_path, send=never, runs_dir=tmp_path / "absent")[0]))
    assert any("No run folder on this machine" in info.value for info in at.info)


def test_an_unreadable_artifact_shows_its_message(tmp_path):
    runs = every_kind(tmp_path / "runs")
    (runs / "e1-3" / "scores.csv").write_text("not,a,scores,file\n", encoding="utf-8")
    at = _opened(tmp_path, runs, "e1-3")
    assert [error.value for error in at.error] == [
        "scores.csv could not be shown.",
        "scores.csv could not be shown.",
    ]
    assert any(block.value.startswith("ScoresFormatError: ") for block in at.code)
    assert {"Call log totals", "E1's criteria"} <= {subheader.value for subheader in at.subheader}


def test_a_partial_e1_run_shows_its_validity_refusal(tmp_path):
    runs = every_kind(tmp_path / "runs")
    partial = render([score for score in SCORES if score.holdout != 0])
    (runs / "e1-3" / "scores.csv").write_text(partial, encoding="utf-8")
    at = _opened(tmp_path, runs, "e1-3")
    assert "Means over the holdouts" in [subheader.value for subheader in at.subheader]
    assert len(at.error) == 1
    assert "holdouts" in at.code[-1].value


def test_a_demo_fill_shows_up_among_the_runs(tmp_path):
    runs = every_kind(tmp_path / "runs")
    at = analyzed(app_context(tmp_path, send=never, runs_dir=runs)[0])
    at = settled(at.selectbox(key="bko.method").select("smote").run())
    at = settled(at.button(key="bko.fill").click().run())
    assert not at.error
    options = list(_earlier(at).selectbox(key="bko.run").options)
    assert len(options) == len(NAMES) + 1
    assert any(name.startswith("app-2026110") and name.endswith("-smote") for name in options)


def test_a_call_log_with_no_end_line_says_so(tmp_path):
    runs = every_kind(tmp_path / "runs")
    log = runs / "c1-2" / "calls.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    log.write_text("".join(lines[:-1]), encoding="utf-8")
    at = _opened(tmp_path, runs, "c1-2")
    assert any("no end line" in info.value for info in at.info)
    assert not at.error


def test_a_fill_survives_a_trip_to_earlier_runs(tmp_path):
    at = analyzed(app_context(tmp_path, send=never, runs_dir=tmp_path / "runs")[0])
    at = settled(at.selectbox(key="bko.method").select("smote").run())
    at = settled(at.button(key="bko.fill").click().run())
    at = _earlier(at)
    at = settled(at.radio(key="bko.view").set_value("Demo").run())
    assert "4. Results" in [header.value for header in at.header]


def test_a_headerless_candidates_file_shows_its_error_in_place(tmp_path):
    runs = every_kind(tmp_path / "runs")
    (runs / "c1-2" / "candidates.jsonl").write_text(
        render_lines(1, ("<55", "female", 1), [young(1)]), encoding="utf-8"
    )
    at = _opened(tmp_path, runs, "c1-2")
    assert "candidates.jsonl could not be shown." in [error.value for error in at.error]
    assert any(block.value.startswith("CandidatesFormatError: ") for block in at.code)


def test_a_first_flag_scan_is_listed_and_not_read(tmp_path):
    runs = every_kind(tmp_path / "runs")
    folder = runs / "e2-8"
    folder.mkdir()
    (folder / "first-flag.csv").write_text("not a first-flag table\n", encoding="utf-8")
    (folder / "first-flag.log").write_text("INFO bko.gaps: gaps rows=0\n", encoding="utf-8")
    at = _opened(tmp_path, runs, "e2-8")
    roles = next(
        frame.value for frame in at.dataframe if list(frame.value.columns) == ["role", "bytes"]
    )
    assert sorted(roles["role"]) == ["first-flag.csv", "first-flag.log"]
    assert not at.subheader
    assert not at.error
