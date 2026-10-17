"""`tasks/CLEANUP.md` keeps its contract, and the controls that keep the guard honest.

Each entry declares a `cleanup:<topic>/<slug>` key under the heading of its topic, names what
discharges it and its trigger, and, when blocked, links to where the blocker is tracked. Every key
cited anywhere in the repository resolves to a declared entry, so deleting a paid entry without
scrubbing its citations fails here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pytest

from tests._corpus import REPO_ROOT, markdown_under, numbered_prose_lines, python_under


BACKLOG = Path("tasks/CLEANUP.md")

SURFACES = frozenset({"docs", "tests", "tools", "tasks"})
"""The topics that are not stages under `src/bko/`."""

DECLARATION = re.compile(r"^- \*\*`(cleanup:[^`]*)`")
KEY = re.compile(r"cleanup:([a-z0-9]+(?:-[a-z0-9]+)*)/([a-z0-9]+(?:-[a-z0-9]+)*)")
CITATION = re.compile(r"cleanup:[a-z0-9-]+/[a-z0-9-]+")
TOPIC_HEADING = re.compile(r"^## ([a-z0-9-]+)$")
HEADING = re.compile(r"^#{1,6} ")
DISCHARGE = "**What discharges it:**"
TRIGGER = "**Trigger:**"
BLOCKED = "**Blocked on"
LINK = re.compile(r"\]\([^)\s]+\)")


def topics(root: Path = REPO_ROOT) -> frozenset[str]:
    """The topics a key may name: every stage under `src/bko/`, and `SURFACES`."""
    stages = {path.parent.name for path in (root / "src" / "bko").glob("*/__init__.py")}
    return frozenset(stages) | SURFACES


def backlog_problems(text: str, allowed: frozenset[str]) -> list[str]:
    """Every way `text`, a backlog, breaks the contract; empty when it holds."""
    problems: list[str] = []
    seen: set[str] = set()
    heading: str | None = None
    heading_line = 0
    entries_under_heading = 0
    entry: tuple[int, str, list[str]] | None = None

    def close_entry() -> None:
        if entry is not None:
            problems.extend(_entry_problems(*entry))

    def close_heading() -> None:
        if heading is not None and entries_under_heading == 0:
            problems.append(f"line {heading_line}: heading `{heading}` holds no entry")

    for number, line in numbered_prose_lines(text):
        if HEADING.match(line):
            close_entry()
            entry = None
            if (m := TOPIC_HEADING.match(line)) is None:
                continue
            close_heading()
            heading, heading_line, entries_under_heading = m.group(1), number, 0
            if heading not in allowed:
                problems.append(f"line {number}: heading `{heading}` is not a topic")
            continue
        if (m := DECLARATION.match(line)) is None:
            if entry is not None:
                entry[2].append(line)
            continue
        close_entry()
        key = m.group(1)
        entry = (number, key, [line])
        entries_under_heading += 1
        if key in seen:
            problems.append(f"line {number}: `{key}` is declared twice")
        seen.add(key)
        if (k := KEY.fullmatch(key)) is None:
            problems.append(f"line {number}: `{key}` is not `cleanup:<topic>/<slug>`")
            continue
        if k.group(1) not in allowed:
            problems.append(f"line {number}: `{key}` names no known topic")
        if heading != k.group(1):
            problems.append(f"line {number}: `{key}` sits under heading `{heading}`")
    close_entry()
    close_heading()
    return problems


def _entry_problems(number: int, key: str, lines: list[str]) -> list[str]:
    body = " ".join(line.strip() for line in lines)
    problems = [
        f"line {number}: `{key}` does not say {marker}"
        for marker in (DISCHARGE, TRIGGER)
        if marker not in body
    ]
    if BLOCKED in body and LINK.search(body[body.index(BLOCKED) :]) is None:
        problems.append(f"line {number}: `{key}` is blocked but links to no tracker")
    return problems


def declared(text: str) -> frozenset[str]:
    """The keys `text` declares."""
    return frozenset(
        m.group(1) for _, line in numbered_prose_lines(text) if (m := DECLARATION.match(line))
    )


def dangling(files: Iterable[Path], keys: frozenset[str], root: Path = REPO_ROOT) -> list[str]:
    """`path: key` for every cited key that no entry declares; a key holds no space, so no wrap."""
    return sorted(
        f"{path.relative_to(root)}: {cited}"
        for path in files
        for cited in CITATION.findall(path.read_text(encoding="utf-8"))
        if cited not in keys
    )


def _citing_files() -> list[Path]:
    files = markdown_under(".", "docs", "tasks", "src", "tools")
    files += python_under("src") + python_under("tests") + python_under("tools")
    return [path for path in files if path != Path(__file__).resolve()]


def _read() -> str:
    return (REPO_ROOT / BACKLOG).read_text(encoding="utf-8")


def test_the_backlog_keeps_its_contract() -> None:
    assert backlog_problems(_read(), topics()) == []


def test_every_cited_key_resolves() -> None:
    assert dangling(_citing_files(), declared(_read())) == []


def test_every_stage_is_a_topic() -> None:
    assert {"ontology", "datasets", "bkb"} <= topics()


GOOD = """\
## ontology

- **`cleanup:ontology/a-gap`: something is wrong.** Evidence.

  **What discharges it:** a fix.

  **Trigger:** the next change.
"""


def test_a_well_formed_backlog_passes() -> None:
    assert backlog_problems(GOOD, frozenset({"ontology"})) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (GOOD.replace("a-gap", "A_Gap"), "is not `cleanup:<topic>/<slug>`"),
        (GOOD.replace("ontology/", "nowhere/"), "names no known topic"),
        (GOOD + GOOD.split("\n", 2)[2], "is declared twice"),
        (GOOD.replace("**What discharges it:**", "Fix:"), "does not say **What discharges it:**"),
        (GOOD.replace("**Trigger:**", "When:"), "does not say **Trigger:**"),
        (GOOD + "\n  **Blocked on** a decision.\n", "is blocked but links to no tracker"),
        (GOOD.replace("## ontology", "## ontology\n\n## datasets"), "sits under heading"),
        (GOOD.replace("## ontology", "## ontology\n\n## datasets"), "holds no entry"),
        (GOOD.replace("## ontology", "## misc"), "is not a topic"),
    ],
)
def test_each_break_is_refused(text: str, expected: str) -> None:
    problems = backlog_problems(text, frozenset({"ontology", "datasets"}))
    assert any(expected in problem for problem in problems), problems


def test_a_blocked_entry_with_a_tracker_passes() -> None:
    text = GOOD + "\n  **Blocked on** a decision. Tracked in [the index](proposal-index.md).\n"
    assert backlog_problems(text, frozenset({"ontology"})) == []


def test_a_dangling_citation_is_reported(tmp_path: Path) -> None:
    cited = tmp_path / "x.md"
    cited.write_text("see `cleanup:ontology/a-gap` and `cleanup:ontology/gone`\n", encoding="utf-8")
    keys = frozenset({"cleanup:ontology/a-gap"})
    assert dangling([cited], keys, root=tmp_path) == ["x.md: cleanup:ontology/gone"]
