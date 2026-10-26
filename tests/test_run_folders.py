"""The run-folder layout: `bko.app.runs` and the docs that write run folders agree."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from bko.app import runs
from bko.app.runs import ROLES, call_totals, discover
from bko.datasets.run_folders import RUN_ID
from tests._corpus import REPO_ROOT
from tests._runs_fixtures import DEMO, call_log, every_kind


DEVELOPMENT: Final = REPO_ROOT / "docs" / "DEVELOPMENT.md"
AGENTS: Final = REPO_ROOT / "AGENTS.md"
EVIDENCE: Final = REPO_ROOT / "docs" / "EVIDENCE.md"
RUN_PATH: Final = re.compile(r"runs/<run-id>(?:-[a-z]+)?/([\w.-]+)")
ROW: Final = re.compile(r"^\| `([\w.-]+)` \| (.+?) \|$", re.MULTILINE)
REGISTER_ID: Final = re.compile(r"(?:^|<br>|\| )`([a-z]\d+-[\w-]+)` · \d{4}-", re.MULTILINE)


def _section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def test_the_development_roles_table_mirrors_roles():
    assert ROW.findall(_section(DEVELOPMENT, "Runs")) == list(ROLES.items())


def test_every_run_path_in_the_docs_names_a_role():
    for path in (DEVELOPMENT, AGENTS):
        named = set(RUN_PATH.findall(path.read_text(encoding="utf-8")))
        assert named, f"{path.name} names no run path, so the scan is broken"
        assert named <= set(ROLES), f"{path.name} names {sorted(named - set(ROLES))}"


def test_no_doc_command_writes_a_scratchpad_run_file():
    for path in (DEVELOPMENT, AGENTS):
        assert "scratchpad/<run-id>" not in path.read_text(encoding="utf-8"), path.name


def test_every_register_run_id_is_a_run_folder_name():
    ids = set(REGISTER_ID.findall(EVIDENCE.read_text(encoding="utf-8")))
    assert {"e1-3", "e1-3-replay", "c1-2-closure", "c1-1-closure-2"} <= ids
    assert [name for name in sorted(ids) if RUN_ID.fullmatch(name) is None] == []


def test_run_id_reads_an_app_fill_and_refuses_other_names():
    app = RUN_ID.fullmatch("app-20261101T000000000000Z-bko-population")
    assert app is not None
    assert app["entry"] is None
    closure = RUN_ID.fullmatch("c1-2-closure")
    assert closure is not None
    assert closure["entry"] == "c1"
    for name in ("scratchpad", "e1", "e1-3.csv", "app-bko", "E1-3", "app-2026-bko", "e\u0661-3"):
        assert RUN_ID.fullmatch(name) is None, name


def test_run_folders_are_gitignored():
    assert "/runs/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_discover_lists_run_folders_with_their_roles(tmp_path):
    runs = discover(every_kind(tmp_path))
    assert [run.name for run in runs] == [DEMO, "c1-2", "c1-2-closure", "e1-3", "e2-5", "r1-2"]
    assert [run.entry for run in runs] == ["app", "c1", "c1", "e1", "e2", "r1"]
    e1 = next(run for run in runs if run.name == "e1-3")
    assert list(e1.roles) == ["calls.jsonl", "scores.csv", "criteria.csv"]
    assert e1.roles["scores.csv"] == tmp_path / "e1-3" / "scores.csv"


def test_discover_finds_nothing_without_a_root(tmp_path):
    assert discover(tmp_path / "absent") == ()


def test_call_totals_reads_the_end_line(tmp_path):
    totals = call_totals(call_log(tmp_path / "x.jsonl"))
    assert totals is not None
    assert totals["status"] == "complete"
    assert totals["calls"] == 1


def test_call_totals_is_none_without_an_end_line(tmp_path):
    lines = call_log(tmp_path / "x.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    stopped = tmp_path / "stopped.jsonl"
    stopped.write_text("".join(lines[:-1]), encoding="utf-8")
    cut = tmp_path / "cut.jsonl"
    cut.write_text("".join(lines)[:-20], encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert [call_totals(path) for path in (stopped, cut, empty)] == [None, None, None]


def test_call_totals_reads_only_the_tail_and_skips_blank_lines(tmp_path, monkeypatch):
    log = call_log(tmp_path / "x.jsonl")
    end = log.read_text(encoding="utf-8").splitlines()[-1]
    blanks = "\n  \n"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(blanks)
    # The window holds the end line, its newline and the blanks, and nothing before them.
    monkeypatch.setattr(runs, "TAIL", len(end) + 1 + len(blanks))
    assert log.stat().st_size > runs.TAIL
    totals = call_totals(log)
    assert totals is not None
    assert totals["status"] == "complete"


def test_discover_skips_links_and_entries_of_the_wrong_kind(tmp_path):
    root = every_kind(tmp_path / "runs")
    (root / "e1-4").symlink_to(root / "e1-3")
    (root / "e2-5" / "gaps-uci.log").mkdir()
    (root / "r1-2" / "reproduction.log").symlink_to(root / "r1-2" / "reproduction.csv")
    (root / "g1-14" / "pytest").mkdir(parents=True)
    (root / "c1-2" / "pytest").write_text("not a directory", encoding="utf-8")
    found = {run.name: list(run.roles) for run in discover(root)}
    assert "e1-4" not in found
    assert "gaps-uci.log" not in found["e2-5"]
    assert found["r1-2"] == ["reproduction.csv"]
    assert found["g1-14"] == ["pytest"]
    assert "pytest" not in found["c1-2"]


def test_discover_orders_run_numbers_as_numbers(tmp_path):
    for name in ("e1-10", "e1-9", "e1-3-replay", "e1-3"):
        (tmp_path / name).mkdir()
    assert [run.name for run in discover(tmp_path)] == ["e1-3", "e1-3-replay", "e1-9", "e1-10"]
