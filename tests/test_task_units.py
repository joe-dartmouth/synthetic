"""Every task unit under `tasks/` carries the unit schema, and the guard refuses each break.

A unit is one pull request's worth of planned work. This module checks every set:

- the frontmatter keys, once each and in order, an `id` matching the file and a known `status`;
- the three sections a unit must have;
- a `blocks` list that mirrors `depends_on`, every dependency a unit, and an acyclic graph;
- a status that agrees with the graph: no unit `blocked` once its dependencies have landed, and none
  `in-progress` while one has not;
- each unit on exactly one index board, whose columns agree with the unit's frontmatter;
- the index diagrams carrying exactly the declared edges;
- every unit's milestone naming a heading of `docs/ROADMAP.md`.

A landed unit's file is deleted (`AGENTS.md`), so an empty `tasks/` is a legitimate state and the
repository case does not demand a unit; the synthetic cases plant one defect each, so a guard that
found nothing cannot pass by reading nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import pytest

from tests._corpus import REPO_ROOT, anchors_of


UNIT_KEYS = (
    "id",
    "title",
    "status",
    "depends_on",
    "blocks",
    "parallel_safe",
    "owner",
    "milestone",
    "paid",
    "adrs",
)
"""The frontmatter keys a unit declares, in this order."""

UNIT_STATUSES = frozenset({"ready", "blocked", "in-progress"})
"""A unit's possible states. A landed unit's file is deleted, never marked done (`AGENTS.md`)."""

BOOLEANS = frozenset({"true", "false"})
"""The values `parallel_safe` and `paid` take."""

ROADMAP = REPO_ROOT / "docs" / "ROADMAP.md"
"""The document whose milestone headings a unit's `milestone` names."""

REQUIRED_SECTIONS = ("## Why this unit exists", "## Scope", "## Acceptance")
"""The sections every unit carries; a unit may add more."""

_BOARD_ROW = re.compile(
    r"^\|\s*\[`([a-z0-9-]+)`\]\(([a-z0-9-]+)\.md\)\s*\|"
    r"\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|",
    re.MULTILINE,
)
"""One board row: the linked unit, then its Status, Depends on, Owner and Milestone cells."""

_MERMAID = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
"""A fenced diagram in an index."""

_NODE = re.compile(r"^\s*([A-Za-z0-9_]+)(?:\[([^\]]+)\])?\s*$")
"""One end of a diagram edge: an alias, optionally with its label."""

type Units = dict[str, dict[str, str]]


def _frontmatter(text: str) -> tuple[list[str], dict[str, str]] | None:
    """The keys in order and their values between the leading `---` fences, or ``None``."""
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return None
    try:
        end = lines.index("---", 1)
    except ValueError:
        return None
    keys: list[str] = []
    fields: dict[str, str] = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        keys.append(key.strip())
        fields[key.strip()] = value.strip()
    return keys, fields


def _names(value: str) -> list[str]:
    """A list written bare (`a, b`), bracketed, backticked, or empty (blank or `--`)."""
    inner = value.strip().strip("[]")
    names = [part.strip().strip("`") for part in inner.split(",")]
    return [name for name in names if name and name != "--"]


def unit_problems(tasks: Path) -> list[str]:
    """Every way the units under ``tasks`` break the schema, one sentence each.

    Args:
        tasks: The directory holding the units and their indexes.

    Returns:
        The problems found, empty when every unit holds.
    """
    problems: list[str] = []
    units: Units = {}
    for path in sorted(tasks.glob("*.md")):
        if path.name.endswith("-index.md"):
            continue
        fields = _file_problems(path, problems)
        if fields is not None:
            units[path.stem] = fields
    problems.extend(_graph_problems(units))
    problems.extend(_board_problems(tasks, units))
    problems.extend(_diagram_problems(tasks, units))
    return problems


def _file_problems(path: Path, problems: list[str]) -> dict[str, str] | None:
    """Append one unit file's own problems, returning its frontmatter when it has one."""
    text = path.read_text(encoding="utf-8")
    parsed = _frontmatter(text)
    if parsed is None:
        problems.append(f"{path.name}: no frontmatter")
        return None
    keys, fields = parsed
    if tuple(keys) != UNIT_KEYS:
        problems.append(f"{path.name}: frontmatter keys {tuple(keys)} are not {UNIT_KEYS}")
    if fields.get("id") != path.stem:
        problems.append(f"{path.name}: id {fields.get('id')!r} is not the file stem")
    status = fields.get("status")
    if status not in UNIT_STATUSES:
        problems.append(f"{path.name}: status {status!r} is not one of {sorted(UNIT_STATUSES)}")
    problems.extend(
        f"{path.name}: {key} {fields.get(key)!r} is not one of {sorted(BOOLEANS)}"
        for key in ("parallel_safe", "paid")
        if fields.get(key) not in BOOLEANS
    )
    headings = {line.strip() for line in text.splitlines() if line.startswith("## ")}
    problems.extend(
        f"{path.name}: missing section {section!r}"
        for section in REQUIRED_SECTIONS
        if section not in headings
    )
    return fields


def _graph_problems(units: Units) -> list[str]:
    """Dependencies naming no unit, unmirrored `blocks`, contradicted statuses, and cycles."""
    problems: list[str] = []
    graph = {name: _names(fields.get("depends_on", "")) for name, fields in units.items()}
    for name, fields in units.items():
        for dependency in graph[name]:
            if dependency not in units:
                problems.append(f"{name}: depends on {dependency!r}, which is not a unit")
            elif name not in _names(units[dependency].get("blocks", "")):
                problems.append(f"{dependency}: blocks does not name {name!r}, which depends on it")
        problems.extend(
            f"{name}: blocks {blocked!r}, which does not depend on it"
            for blocked in _names(fields.get("blocks", ""))
            if blocked not in units or name not in graph.get(blocked, [])
        )
        open_dependencies = [d for d in graph[name] if d in units]
        if fields.get("status") == "blocked" and not open_dependencies:
            problems.append(f"{name}: is blocked although every dependency has landed")
        if fields.get("status") == "in-progress" and open_dependencies:
            problems.append(f"{name}: is in progress while {open_dependencies} have not landed")
    problems.extend(_cycles(graph))
    return problems


def _cycles(graph: dict[str, list[str]]) -> list[str]:
    """One sentence per unit that sits on a `depends_on` cycle."""
    on_cycle: list[str] = []
    for start, dependencies in graph.items():
        stack: list[str] = list(dependencies)
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node == start:
                on_cycle.append(f"{start}: sits on a depends_on cycle")
                break
            if node in seen or node not in graph:
                continue
            seen.add(node)
            stack.extend(graph[node])
    return on_cycle


def _board_problems(tasks: Path, units: Units) -> list[str]:
    """Units on no board or on several, rows with no unit, and cells that disagree with a unit."""
    problems: list[str] = []
    rows: dict[str, list[str]] = {}
    for index in sorted(tasks.glob("*-index.md")):
        for row_id, row_file, status, depends, owner, milestone in _BOARD_ROW.findall(
            index.read_text(encoding="utf-8")
        ):
            if row_id != row_file:
                problems.append(f"{index.name}: row {row_id!r} links {row_file}.md")
            rows.setdefault(row_id, []).append(index.name)
            fields = units.get(row_id)
            if fields is None:
                continue
            cells = {
                "Status": (status, fields.get("status", "")),
                "Depends on": (sorted(_names(depends)), sorted(_names(fields["depends_on"]))),
                "Owner": (owner, fields.get("owner", "")),
                "Milestone": (milestone, fields.get("milestone", "")),
            }
            problems.extend(
                f"{index.name}: {row_id}'s {column} cell {shown!r} disagrees with {declared!r}"
                for column, (shown, declared) in cells.items()
                if shown != declared
            )
    for name in units:
        listed = rows.get(name, [])
        if len(listed) != 1:
            problems.append(f"{name}: on {len(listed)} board rows ({listed}), not exactly one")
    problems.extend(
        f"{name}: on an index board with no unit file" for name in rows if name not in units
    )
    return problems


def _diagram_problems(tasks: Path, units: Units) -> list[str]:
    """Edges the index diagrams draw that no unit declares, and declared edges they omit."""
    drawn: set[tuple[str, str]] = set()
    for index in sorted(tasks.glob("*-index.md")):
        for diagram in _MERMAID.findall(index.read_text(encoding="utf-8")):
            labels: dict[str, str] = {}
            for line in diagram.splitlines():
                ends = [_NODE.match(part) for part in line.split("-->")]
                if len(ends) < 2:
                    continue
                aliases: list[str] = []
                for match in ends:
                    if match is None:
                        break
                    alias, label = match.groups()
                    labels.setdefault(alias, label or alias)
                    aliases.append(alias)
                drawn.update((labels[a], labels[b]) for a, b in pairwise(aliases))
    declared = {
        (dependency, name)
        for name, fields in units.items()
        for dependency in _names(fields.get("depends_on", ""))
    }
    extra = [f"the index diagrams draw {edge} but no unit declares it" for edge in drawn - declared]
    missing = [f"the index diagrams omit the declared edge {edge}" for edge in declared - drawn]
    return extra + missing


def test_every_unit_in_the_repository_holds_the_schema() -> None:
    """The committed units carry the schema; an empty `tasks/` is legitimate between sets."""
    assert (REPO_ROOT / "tasks").is_dir()
    assert unit_problems(REPO_ROOT / "tasks") == []


def test_every_milestone_is_a_roadmap_anchor() -> None:
    """A unit's `milestone` names a `docs/ROADMAP.md` heading, so the board links resolve."""
    anchors = anchors_of(ROADMAP)
    stray = {
        path.name: fields.get("milestone")
        for path in sorted((REPO_ROOT / "tasks").glob("*.md"))
        if not path.name.endswith("-index.md")
        and (parsed := _frontmatter(path.read_text(encoding="utf-8"))) is not None
        and (fields := parsed[1]).get("milestone") not in anchors
    }
    assert not stray, f"milestones that name no heading of {ROADMAP.name}: {stray}"


_GOOD_UNIT = """---
id: {id}
title: A unit
status: {status}
depends_on: {depends}
blocks: {blocks}
parallel_safe: false
owner: Joe
milestone: final-report
paid: false
adrs:
---

# A unit

## Why this unit exists

Because.

## Scope

All of it.

## Acceptance

It holds.
"""


def _status(depends: str) -> str:
    """The status a well-formed unit holds: blocked while it has a dependency, ready otherwise."""
    return "blocked" if depends else "ready"


def _plant(tmp_path: Path, units: dict[str, tuple[str, str]], board: list[str]) -> Path:
    """Write ``units`` (id -> depends, blocks) and one index whose board lists ``board``."""
    for name, (depends, blocks) in units.items():
        text = _GOOD_UNIT.format(id=name, status=_status(depends), depends=depends, blocks=blocks)
        (tmp_path / f"{name}.md").write_text(text, encoding="utf-8")
    rows = []
    for name in board:
        depends = units[name][0] if name in units else ""
        rows.append(
            f"| [`{name}`]({name}.md) | {_status(depends)} | {depends or '--'} | Joe | final-report |"
        )
    edges = [f"    {d} --> {name}" for name, (depends, _) in units.items() for d in _names(depends)]
    diagram = "```mermaid\nflowchart LR\n" + "\n".join(edges) + "\n```\n" if edges else ""
    table = "| Unit | Status | Depends on | Owner | Milestone |\n|---|---|---|---|---|\n"
    text = f"# Set\n\n{table}" + "\n".join(rows) + f"\n\n{diagram}"
    (tmp_path / "set-index.md").write_text(text, encoding="utf-8")
    return tmp_path


def _problems_after(tmp_path: Path, path: str, edit: Callable[[str], str]) -> list[str]:
    """Plant a well-formed two-unit set, apply ``edit`` to one file, and return the problems."""
    root = _plant(tmp_path, {"a": ("", "b"), "b": ("a", "")}, ["a", "b"])
    target = root / path
    target.write_text(edit(target.read_text(encoding="utf-8")), encoding="utf-8")
    return unit_problems(root)


def test_a_well_formed_synthetic_set_passes(tmp_path: Path) -> None:
    """The control: two units that hold the schema produce no problem."""
    root = _plant(tmp_path, {"a": ("", "b"), "b": ("a", "")}, ["a", "b"])
    assert unit_problems(root) == []


_DEFECTS: list[tuple[str, str, Callable[[str], str], str]] = [
    ("status", "a.md", lambda t: t.replace("status: ready", "status: done"), "status 'done'"),
    ("missing-key", "a.md", lambda t: t.replace("owner: Joe\n", ""), "frontmatter keys"),
    (
        "duplicate-key",
        "a.md",
        lambda t: t.replace("adrs:\n", "adrs:\nstatus: blocked\n"),
        "frontmatter keys",
    ),
    ("paid-not-boolean", "a.md", lambda t: t.replace("paid: false", "paid: maybe"), "paid 'maybe'"),
    ("id", "a.md", lambda t: t.replace("id: a\n", "id: z\n"), "is not the file stem"),
    ("section", "a.md", lambda t: t.replace("## Acceptance", "## Accept"), "missing section"),
    ("no-frontmatter", "a.md", lambda t: t.replace("---\nid", "id", 1), "no frontmatter"),
    (
        "blocked-with-nothing-open",
        "a.md",
        lambda t: t.replace("status: ready", "status: blocked"),
        "every dependency",
    ),
    (
        "in-progress-with-open-dependency",
        "b.md",
        lambda t: t.replace("status: blocked", "status: in-progress"),
        "in progress",
    ),
    (
        "dependency-not-a-unit",
        "b.md",
        lambda t: t.replace("depends_on: a", "depends_on: a, gone"),
        "not a unit",
    ),
    (
        "blocks-a-non-dependent",
        "a.md",
        lambda t: t.replace("blocks: b", "blocks: b, gone"),
        "does not depend",
    ),
    ("row-links-another-file", "set-index.md", lambda t: t.replace("(a.md)", "(b.md)"), "links"),
    ("status-cell", "set-index.md", lambda t: t.replace("| blocked |", "| ready |"), "Status cell"),
    (
        "depends-cell",
        "set-index.md",
        lambda t: t.replace("| a | Joe", "| -- | Joe"),
        "Depends on cell",
    ),
    ("owner-cell", "set-index.md", lambda t: t.replace("| Joe |", "| Krish |", 1), "Owner cell"),
    (
        "milestone-cell",
        "set-index.md",
        lambda t: t.replace("| final-report |", "| status-update-1 |", 1),
        "Milestone cell",
    ),
    ("diagram-extra-edge", "set-index.md", lambda t: t.replace("a --> b", "b --> a"), "draw"),
    ("diagram-missing-edge", "set-index.md", lambda t: t.replace("    a --> b\n", ""), "omit"),
]
"""One planted defect per check: (id, file edited, the edit, a phrase the problem must carry)."""


@pytest.mark.parametrize(
    ("path", "edit", "expected"),
    [defect[1:] for defect in _DEFECTS],
    ids=[defect[0] for defect in _DEFECTS],
)
def test_each_defect_is_refused(
    tmp_path: Path, path: str, edit: Callable[[str], str], expected: str
) -> None:
    """A planted defect in one file is named."""
    problems = _problems_after(tmp_path, path, edit)
    assert any(expected in problem for problem in problems), problems


def test_a_blocks_list_that_does_not_mirror_depends_on_is_refused(tmp_path: Path) -> None:
    """`b` depends on `a`, and `a` does not say it blocks `b`."""
    root = _plant(tmp_path, {"a": ("", ""), "b": ("a", "")}, ["a", "b"])
    assert any("blocks does not name 'b'" in p for p in unit_problems(root))


@pytest.mark.parametrize(
    "units",
    [
        {"a": ("b", "b"), "b": ("a", "a")},
        {"a": ("a", "a")},
        {"a": ("c", "b"), "b": ("a", "c"), "c": ("b", "a")},
    ],
    ids=["two-cycle", "self-loop", "three-cycle"],
)
def test_a_dependency_cycle_is_refused(tmp_path: Path, units: dict[str, tuple[str, str]]) -> None:
    """A cycle of any length, a unit depending on itself included, is named."""
    root = _plant(tmp_path, units, list(units))
    assert any("depends_on cycle" in p for p in unit_problems(root))


def test_a_diamond_is_not_a_cycle(tmp_path: Path) -> None:
    """Two paths to one shared dependency are a DAG, and the `seen` set must not flag them."""
    units = {"a": ("", "b, c"), "b": ("a", "d"), "c": ("a", "d"), "d": ("b, c", "")}
    root = _plant(tmp_path, units, list(units))
    assert unit_problems(root) == []


def test_a_unit_on_no_board_is_refused(tmp_path: Path) -> None:
    """A unit missing from every index board is named."""
    root = _plant(tmp_path, {"a": ("", "")}, [])
    assert any("on 0 board rows" in p for p in unit_problems(root))


def test_a_unit_on_two_boards_is_refused(tmp_path: Path) -> None:
    """A unit listed by a second index is named."""
    root = _plant(tmp_path, {"a": ("", "")}, ["a"])
    board = (root / "set-index.md").read_text(encoding="utf-8")
    (root / "other-index.md").write_text(board, encoding="utf-8")
    assert any("on 2 board rows" in p for p in unit_problems(root))


def test_a_board_row_with_no_unit_is_refused(tmp_path: Path) -> None:
    """A board row whose unit file was deleted is named."""
    root = _plant(tmp_path, {"a": ("", "")}, ["a", "gone"])
    assert any("gone: on an index board with no unit file" in p for p in unit_problems(root))
