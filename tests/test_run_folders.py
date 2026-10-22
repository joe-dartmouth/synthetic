"""The run-folder layout: `bko.app.runs` and the docs that write run folders agree."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from bko.app.runs import ROLES, RUN_ID
from tests._corpus import REPO_ROOT


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
