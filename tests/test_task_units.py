"""Every task unit under `tasks/` carries the unit schema, and the guard refuses each break.

A unit is one pull request's worth of planned work. This module checks every set:

- the frontmatter keys, once each and in order, an `id` matching the file and a known `status`;
- an `owner` from the `docs/ROADMAP.md` Workload members and `adrs` naming existing ADR topics;
- the three sections a unit must have, each with a body;
- a `blocks` list that mirrors `depends_on`, every dependency a unit, and an acyclic graph;
- a status that agrees with the graph: no unit `blocked` once its dependencies have landed, and none
  `ready` or `in-progress` while one has not;
- each unit on exactly one index board, whose columns agree with the unit's frontmatter, and no
  board-looking row that does not parse;
- the index diagrams carrying exactly the declared edges, in a form the check can read;
- every unit's milestone naming a milestone heading of `docs/ROADMAP.md`.

A landed unit's file is deleted (`AGENTS.md`), so an empty `tasks/` is a legitimate state and the
repository case does not demand a unit; the synthetic cases plant one defect each, so a guard that
found nothing cannot pass by reading nothing.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import pytest

from tests._corpus import REPO_ROOT, numbered_prose_lines, slug


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
"""The document whose milestone headings a unit's `milestone` names, and whose members own units."""

ADR_ROOT = REPO_ROOT / "docs" / "adr"
"""The directory whose topic subdirectories a unit's `adrs` ids name."""

WORKLOAD = "## Workload"
"""The ROADMAP heading that ends the milestones and holds the member table."""

REQUIRED_SECTIONS = ("## Why this unit exists", "## Scope", "## Acceptance")
"""The sections every unit carries, each with a body; a unit may add more."""

_UNIT_CELL = re.compile(r"\[`([a-z0-9-]+)`\]\(([a-z0-9-]+)\.md\)")
"""A board row's first cell: the unit's id in a code span, linking its file."""

_ADR_ID = re.compile(r"([a-z0-9-]+)/([a-z0-9-]+)")
"""A unit's `adrs` entry: ``topic/slug``."""

_NODE = re.compile(r"^\s*([A-Za-z0-9_]+)(?:\[([^\]]+)\])?\s*$")
"""One end of a diagram edge: an alias, optionally with its label."""

_UNSUPPORTED_ARROWS = ("-.->", "==>", "---", "-->|", "&")
"""Mermaid edge forms the diagram check does not read, so a line using one is reported."""

type Units = dict[str, dict[str, str]]
type BoardRow = tuple[int, str, str, str, str, str, str]


def roadmap_milestones(text: str) -> frozenset[str]:
    """The anchors of the ROADMAP's milestones: every ``## `` heading before the Workload."""
    found: set[str] = set()
    for _, line in numbered_prose_lines(text):
        if line.strip() == WORKLOAD:
            break
        if line.startswith("## "):
            found.add(slug(line[3:]))
    return frozenset(found)


def roadmap_owners(text: str) -> frozenset[str]:
    """The first names in the ROADMAP Workload table, the "All members" row excluded."""
    lines = [line for _, line in numbered_prose_lines(text)]
    if WORKLOAD not in (line.strip() for line in lines):
        return frozenset()
    section = lines[[line.strip() for line in lines].index(WORKLOAD) + 1 :]
    rows: list[str] = []
    for line in section:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            rows.append(line)
    members = (row.strip().strip("|").split("|")[0].strip() for row in rows[2:])
    return frozenset(m.split()[0] for m in members if m and not m.lower().startswith("all "))


MILESTONES = roadmap_milestones(ROADMAP.read_text(encoding="utf-8"))
"""The milestone anchors `docs/ROADMAP.md` defines today."""

OWNERS = roadmap_owners(ROADMAP.read_text(encoding="utf-8"))
"""The member first names `docs/ROADMAP.md` lists today."""


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


def unit_problems(
    tasks: Path,
    *,
    milestones: frozenset[str] = MILESTONES,
    owners: frozenset[str] = OWNERS,
    adr_root: Path = ADR_ROOT,
) -> list[str]:
    """Every way the units under ``tasks`` break the schema, one sentence each.

    Args:
        tasks: The directory holding the units and their indexes.
        milestones: The anchors a unit's `milestone` may name.
        owners: The names a unit's `owner` may be.
        adr_root: The directory whose subdirectories are the ADR topics.

    Returns:
        The problems found, sorted, empty when every unit holds.
    """
    problems: list[str] = []
    units: Units = {}
    for path in sorted(tasks.glob("*.md")):
        if path.name.endswith("-index.md"):
            continue
        fields = _file_problems(path, problems, owners=owners, adr_root=adr_root)
        if fields is not None:
            units[path.stem] = fields
    problems.extend(milestone_problems(milestones, units))
    problems.extend(_graph_problems(units))
    problems.extend(_board_problems(tasks, units))
    problems.extend(_diagram_problems(tasks, units))
    return sorted(problems)


def milestone_problems(milestones: frozenset[str], units: Units) -> list[str]:
    """Units whose `milestone` is not one of ``milestones``, the ROADMAP's milestone anchors."""
    return [
        f"{name}: milestone {fields.get('milestone')!r} is not a ROADMAP milestone "
        f"{sorted(milestones)}"
        for name, fields in units.items()
        if fields.get("milestone") not in milestones
    ]


def _section_bodies(text: str) -> dict[str, str]:
    """Each ``## `` heading's body, up to the next ``#`` or ``##`` heading."""
    bodies: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = bodies.setdefault(line.strip(), [])
        elif line.startswith("# "):
            current = None
        elif current is not None:
            current.append(line)
    return {heading: "\n".join(lines).strip() for heading, lines in bodies.items()}


def _adr_problems(name: str, value: str, adr_root: Path) -> list[str]:
    """An `adrs` value that is not `--`, empty, or a list of `topic/slug` with a known topic."""
    problems: list[str] = []
    for entry in _names(value):
        match = _ADR_ID.fullmatch(entry)
        if match is None:
            problems.append(f"{name}: adrs entry {entry!r} is not a topic/slug id")
        elif not (adr_root / match.group(1)).is_dir():
            problems.append(f"{name}: adrs entry {entry!r} names no topic under {adr_root.name}/")
    return problems


def _file_problems(
    path: Path, problems: list[str], *, owners: frozenset[str], adr_root: Path
) -> dict[str, str] | None:
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
    if fields.get("owner") not in owners:
        problems.append(
            f"{path.name}: owner {fields.get('owner')!r} is not a ROADMAP member {sorted(owners)}"
        )
    problems.extend(_adr_problems(path.name, fields.get("adrs", ""), adr_root))
    bodies = _section_bodies(text)
    for section in REQUIRED_SECTIONS:
        if section not in bodies:
            problems.append(f"{path.name}: missing section {section!r}")
        elif not bodies[section]:
            problems.append(f"{path.name}: section {section!r} is empty")
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
        if fields.get("status") == "ready" and open_dependencies:
            problems.append(f"{name}: is ready while {open_dependencies} have not landed")
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


def board_row(line: str) -> tuple[str, str, str, str, str, str] | None:
    """One board row's unit, linked file and Status, Depends on, Owner and Milestone cells.

    Split on ``|`` rather than matched by one pattern: a pattern of lazy cells between optional
    whitespace backtracks cubically on a long run of spaces with no closing ``|``.

    Returns:
        ``None`` when the row does not have exactly five cells or its first is not a unit link.
    """
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    cells = [cell.strip() for cell in stripped[1:-1].split("|")]
    if len(cells) != 5 or (unit := _UNIT_CELL.fullmatch(cells[0])) is None:
        return None
    row_id, row_file = unit.groups()
    return row_id, row_file, cells[1], cells[2], cells[3], cells[4]


def _board_rows(index: Path, problems: list[str]) -> list[BoardRow]:
    """The board rows of one index, outside fences, appending a problem per malformed row.

    A line that starts with ``|`` and links a ``.md`` file looks like a row; one that does not
    parse is reported rather than skipped, so a typo cannot hide a unit from the board check.
    """
    rows: list[BoardRow] = []
    for number, line in numbered_prose_lines(index.read_text(encoding="utf-8")):
        if not (line.startswith("|") and ".md)" in line):
            continue
        if (row := board_row(line)) is None:
            problems.append(f"{index.name}: line {number}: malformed board row {line.strip()!r}")
        else:
            rows.append((number, *row))
    return rows


def _board_problems(tasks: Path, units: Units) -> list[str]:
    """Units on no board or on several, rows with no unit, and cells that disagree with a unit."""
    problems: list[str] = []
    rows: dict[str, list[str]] = {}
    for index in sorted(tasks.glob("*-index.md")):
        for number, row_id, row_file, status, depends, owner, milestone in _board_rows(
            index, problems
        ):
            if row_id != row_file:
                problems.append(f"{index.name}: line {number}: row {row_id!r} links {row_file}.md")
            rows.setdefault(row_id, []).append(index.name)
            fields = units.get(row_id)
            if fields is None:
                continue
            declared_depends = sorted(_names(fields.get("depends_on", "")))
            cells = {
                "Status": (status, fields.get("status", "")),
                "Depends on": (sorted(_names(depends)), declared_depends),
                "Owner": (owner, fields.get("owner", "")),
                "Milestone": (milestone, fields.get("milestone", "")),
            }
            problems.extend(
                f"{index.name}: line {number}: {row_id}'s {column} cell {shown!r} "
                f"disagrees with {declared!r}"
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


def _mermaid_blocks(text: str) -> list[list[tuple[int, str]]]:
    """Each fenced mermaid diagram's lines, numbered as in ``text``."""
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] | None = None
    for number, line in enumerate(text.splitlines(), 1):
        if current is None:
            if line.strip() == "```mermaid":
                current = []
                blocks.append(current)
        elif line.strip().startswith("```"):
            current = None
        else:
            current.append((number, line))
    return blocks


def _diagram_problems(tasks: Path, units: Units) -> list[str]:
    """Unreadable diagram lines, drawn edges no unit declares, and declared edges not drawn."""
    problems: list[str] = []
    drawn: set[tuple[str, str]] = set()
    for index in sorted(tasks.glob("*-index.md")):
        for block in _mermaid_blocks(index.read_text(encoding="utf-8")):
            labels: dict[str, str] = {}
            for number, line in block:
                edge = line.strip()
                where = f"{index.name}: line {number}"
                if any(form in edge for form in _UNSUPPORTED_ARROWS):
                    problems.append(f"{where}: unsupported edge form {edge!r}")
                    continue
                if "-->" not in edge:
                    continue
                ends = [_NODE.match(part) for part in edge.split("-->")]
                if any(match is None for match in ends):
                    problems.append(f"{where}: unparseable edge {edge!r}")
                    continue
                aliases: list[str] = []
                for match in ends:
                    assert match is not None
                    alias, label = match.groups()
                    labels.setdefault(alias, label or alias)
                    aliases.append(alias)
                drawn.update((labels[a], labels[b]) for a, b in pairwise(aliases))
    declared = {
        (dependency, name)
        for name, fields in units.items()
        for dependency in _names(fields.get("depends_on", ""))
    }
    problems.extend(
        f"the index diagrams draw {edge} but no unit declares it" for edge in drawn - declared
    )
    problems.extend(
        f"the index diagrams omit the declared edge {edge}" for edge in declared - drawn
    )
    return problems


def test_every_unit_in_the_repository_holds_the_schema() -> None:
    """The committed units carry the schema; an empty `tasks/` is legitimate between sets."""
    assert (REPO_ROOT / "tasks").is_dir()
    problems = unit_problems(REPO_ROOT / "tasks")
    assert problems == [], "task units break the schema:\n" + "\n".join(problems)


def test_the_roadmap_still_yields_milestones_and_owners() -> None:
    """Both vocabularies are read from `docs/ROADMAP.md`; an empty read would refuse every unit,
    or, with `tasks/` empty, pass silently while the doc lost its structure."""
    assert MILESTONES, f"no `## ` milestone heading before {WORKLOAD!r} in {ROADMAP.name}"
    assert OWNERS, f"no member row under {WORKLOAD!r} in {ROADMAP.name}"


def test_the_milestone_check_refuses_a_name_outside_the_given_set() -> None:
    """Synthetic control: the check reads the anchors it is given, not the live ROADMAP."""
    units: Units = {"a": {"milestone": "final-report"}, "b": {"milestone": "workload"}}
    problems = milestone_problems(frozenset({"final-report"}), units)
    assert problems == ["b: milestone 'workload' is not a ROADMAP milestone ['final-report']"]


def test_the_roadmap_readers_take_only_their_sections() -> None:
    """Headings after the Workload are not milestones, and "All members" owns nothing."""
    text = (
        "# Roadmap\n\n## Status Update 1\n\nx\n\n## Final Report\n\n## Workload\n\n"
        "| Member | Workstream |\n|---|---|\n| Ada Lovelace | Engines. |\n"
        "| All members | Reviews. |\n\n## Beyond the proposal\n"
    )
    assert roadmap_milestones(text) == {"status-update-1", "final-report"}
    assert roadmap_owners(text) == {"Ada"}


def test_the_board_row_parser_does_not_backtrack_on_a_long_run() -> None:
    """A guard over hand-edited text must not have a length at which it stops returning.

    The threshold is loose by orders of magnitude, so it fails on a reintroduced polynomial
    pattern and not on a busy machine.
    """
    subject = "| [`a`](a.md) |" + " " * 10_000 + "x"
    started = time.perf_counter()
    assert board_row(subject) is None
    assert time.perf_counter() - started < 0.1, "the board row parser is backtracking"


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
adrs:{adrs}
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
        # Both spellings of "no ADR": `--` on the first unit, blank on the rest.
        adrs = " --" if name == next(iter(units)) else ""
        text = _GOOD_UNIT.format(
            id=name, status=_status(depends), depends=depends, blocks=blocks, adrs=adrs
        )
        (tmp_path / f"{name}.md").write_text(text, encoding="utf-8")
    rows = []
    for name in board:
        depends = units[name][0] if name in units else ""
        rows.append(
            f"| [`{name}`]({name}.md) | {_status(depends)} | {depends or '--'} "
            "| Joe | final-report |"
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
        lambda t: t.replace("adrs: --\n", "adrs: --\nstatus: blocked\n"),
        "frontmatter keys",
    ),
    (
        "depends-on-missing",
        "b.md",
        lambda t: t.replace("depends_on: a\n", ""),
        "frontmatter keys",
    ),
    (
        "empty-section",
        "a.md",
        lambda t: t.replace("All of it.\n", ""),
        "section '## Scope' is empty",
    ),
    (
        "adr-topic-typo",
        "a.md",
        lambda t: t.replace("adrs: --", "adrs: tooling/not-yet-written, reserch/x"),
        "'reserch/x' names no topic",
    ),
    (
        "adr-not-an-id",
        "a.md",
        lambda t: t.replace("adrs: --", "adrs: uv-ruff-ty"),
        "is not a topic/slug id",
    ),
    (
        "owner-not-a-member",
        "a.md",
        lambda t: t.replace("owner: Joe", "owner: Nobody"),
        "owner 'Nobody' is not a ROADMAP member",
    ),
    (
        "milestone-not-a-roadmap-milestone",
        "a.md",
        lambda t: t.replace("milestone: final-report", "milestone: workload"),
        "milestone 'workload' is not a ROADMAP milestone",
    ),
    (
        "ready-with-open-dependency",
        "b.md",
        lambda t: t.replace("status: blocked", "status: ready"),
        "is ready while ['a'] have not landed",
    ),
    (
        "malformed-board-row",
        "set-index.md",
        lambda t: t.replace("| [`a`](a.md) | ready |", "| [`a`](a.md) ready |"),
        "set-index.md: line 5: malformed board row",
    ),
    (
        "unparseable-edge",
        "set-index.md",
        lambda t: t.replace("    a --> b\n", "    a --> b\n    a[x --> b\n"),
        "set-index.md: line 11: unparseable edge 'a[x --> b'",
    ),
    (
        "dotted-edge",
        "set-index.md",
        lambda t: t.replace("    a --> b\n", "    a --> b\n    a -.-> b\n"),
        "set-index.md: line 11: unsupported edge form",
    ),
    (
        "labelled-edge",
        "set-index.md",
        lambda t: t.replace("    a --> b\n", "    a --> b\n    a -->|x| b\n"),
        "unsupported edge form",
    ),
    (
        "open-link",
        "set-index.md",
        lambda t: t.replace("    a --> b\n", "    a --> b\n    a --- b\n"),
        "unsupported edge form",
    ),
    (
        "chained-ampersand",
        "set-index.md",
        lambda t: t.replace("    a --> b\n", "    a --> b\n    a & a --> b\n"),
        "unsupported edge form",
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


def test_a_board_looking_line_inside_a_fence_is_not_a_row(tmp_path: Path) -> None:
    """An index may show a row as an example; a fenced line is neither a row nor malformed."""
    root = _plant(tmp_path, {"a": ("", "")}, ["a"])
    index = root / "set-index.md"
    example = "```markdown\n| [`gone`](gone.md) broken |\n```\n"
    index.write_text(index.read_text(encoding="utf-8") + example, encoding="utf-8")
    assert unit_problems(root) == []
