"""The local gate, held so its pieces cannot silently stop existing.

There is no CI workflow and there are no git hooks: `make check` is run by hand before a pull
request, and it is the only gate this repository has. These tests hold that the target still means
every step, in order, and that the configuration behind each step agrees with itself. They are
cheap assertions over config files on purpose; actually running the gate is the gate.
"""

from __future__ import annotations

import json
import re
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from tests._corpus import REPO_ROOT
from tools._git import require_toplevel, run
from tools.check_lock_drift import normalize


MAKEFILE = REPO_ROOT / "Makefile"
UV_LOCK = REPO_ROOT / "uv.lock"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CLAUDE_SETTINGS = REPO_ROOT / ".claude" / "settings.json"

TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".toml", ".yaml", ".yml", ".html", ".css", ".jsonl", ".json", ".cfg", ".txt"}
)
"""Suffixes whose files are text a conflict marker could survive in. PDFs are skipped."""

TEXT_NAMES = frozenset({".gitignore", ".gitattributes", ".python-version", "Makefile"})
"""Tracked text files whose names carry no scanned suffix."""


def tracked_files() -> list[str]:
    """Every tracked path, from ``git ls-files`` so a new file is covered without an edit."""
    require_toplevel(REPO_ROOT)
    listed = run(["ls-files", "-z"], root=REPO_ROOT, check=True).stdout.split("\0")
    return [name for name in listed if name]


def tracked_text_files() -> list[Path]:
    """Every tracked text file."""
    paths = [REPO_ROOT / name for name in tracked_files()]
    return [p for p in paths if (p.suffix in TEXT_SUFFIXES or p.name in TEXT_NAMES) and p.is_file()]


TRACKED_TEXT = tracked_text_files()


def makefile_text() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _sole_match(pattern: str, body: str, *, join: str = "") -> str:
    """The one capture for `pattern`, or a failure naming which file lost its pin.

    Strict about the count: a second match means the quantity is declared twice and a reader
    cannot tell which one is load-bearing.
    """
    found = re.findall(pattern, body)
    assert len(found) == 1, f"expected exactly one {pattern!r}, found {found}"
    only = found[0]
    return join.join(only) if isinstance(only, tuple) else only


def test_check_is_a_make_target_covering_every_gate():
    """`make check` is hand-run, so the target itself is the contract.

    Equality, not membership, so an added or dropped prerequisite fails, and the order is held
    too: `rebased` before `synced` because a rebase can move `uv.lock`; `sizes` before `test`
    because it is cheaper; `rebased-final` last because the trunk can move while the tests run
    (ADR `tooling/refuse-a-stale-trunk-ref`).
    """
    check_line = next(
        (line for line in makefile_text().splitlines() if line.startswith("check:")), None
    )
    assert check_line is not None, "there is no `make check` target"
    expected = ["rebased", "synced", "lint", "sizes", "typecheck", "test", "rebased-final"]
    assert check_line.split(":", 1)[1].split() == expected, (
        f"`make check`'s dependencies are not the gates in order: {check_line!r}"
    )


def _recipe(target: str) -> list[str]:
    """The recipe lines of one Makefile target, without their leading tab."""
    lines = makefile_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"{target}:")), None)
    assert start is not None, f"there is no `make {target}` target"
    recipe: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        recipe.append(line[1:])
    assert recipe, f"`make {target}` has no recipe, so it runs nothing"
    return recipe


def _recipes() -> list[str]:
    """Every recipe line in the Makefile, without its leading tab."""
    return [line[1:] for line in makefile_text().splitlines() if line.startswith("\t")]


def test_the_final_rebased_step_runs_the_same_check_as_the_first():
    """`rebased-final` asks the same question as `rebased`, and both are phony.

    A separate target rather than a repeated prerequisite, because Make runs a target once per
    invocation. A file named `rebased-final` in the tree would otherwise make the step a silent
    no-op, which `.PHONY` prevents.
    """
    first = [line for line in _recipe("rebased") if not line.startswith("@echo")]
    final = [line for line in _recipe("rebased-final") if not line.startswith("@echo")]
    assert first == final, f"`rebased-final` runs {final}, not what `rebased` runs: {first}"
    assert any("tools.check_rebased" in line for line in final)
    phony = next(line for line in makefile_text().splitlines() if line.startswith(".PHONY:"))
    assert {"rebased", "rebased-final"} <= set(phony.split()[1:])


def test_no_gate_step_can_relock_or_install():
    """Every `uv run` carries `--no-sync`, and every `uv sync` carries `--locked`.

    A bare `uv run` re-locks and re-syncs when `pyproject.toml` is ahead of `uv.lock`, so a gate
    step could rewrite the lock and install packages after `synced` measured the environment, and
    the gate would still pass. `--locked` makes an install refuse a stale lock instead of fixing it.
    """
    lines = _recipes()
    unsynced = [line for line in lines if "uv run" in line and "--no-sync" not in line]
    unlocked = [line for line in lines if "uv sync" in line and "--locked" not in line]
    assert any("uv run" in line for line in lines), "no recipe runs `uv run`; the scan is broken"
    assert not unsynced, f"recipes that let `uv run` re-lock or install: {unsynced}"
    assert not unlocked, f"`uv sync` recipes that can install a stale lock: {unlocked}"


def test_the_test_target_measures_coverage_against_the_configured_floor():
    """The floor lives in `pyproject.toml`, and the recipe is what activates it.

    `make test` without `--cov` runs the suite and enforces no floor at all, which reads
    identically in a terminal; a deleted `fail_under`, `branch` or `source` does the same.
    """
    recipe = "\n".join(_recipe("test"))
    assert re.search(r"--cov(\s|$)", recipe), "`make test` does not measure coverage"
    coverage = pyproject()["tool"]["coverage"]
    assert {"src/bko", "tools"} <= set(coverage["run"]["source"]), coverage["run"]
    assert coverage["run"]["branch"] is True, "coverage measures lines only"
    assert coverage["report"]["fail_under"] >= 90, coverage["report"]


def test_the_test_target_keeps_a_dynamic_scheduler():
    """xdist's default `load` scheduler assigns work in chunks up front.

    On a suite with a few dominant tests the wall clock then turns on whether two long tests land
    on the same worker. A dynamic scheduler hands out work on demand.
    """
    recipe = "\n".join(_recipe("test"))
    assert any(f"--dist {name}" in recipe for name in ("worksteal", "loadgroup")), (
        f"the `make test` recipe is back on xdist's chunked `load` scheduler: {recipe!r}"
    )


def test_every_interpreter_pin_agrees():
    """Six files name the Python version, and a bump that misses one splits the toolchain.

    ruff and ty read their target from this repository's config rather than from the interpreter,
    so lint and typecheck pass whichever Python is installed. ADR
    `tooling/python-314-without-synthcity` lists the six homes.
    """
    declared = {
        ".python-version": (REPO_ROOT / ".python-version").read_text().strip(),
        "pyproject.toml:requires-python": _sole_match(
            r'requires-python\s*=\s*">=(\d+\.\d+)"', PYPROJECT.read_text()
        ),
        "ruff.toml:target-version": _sole_match(
            r'target-version\s*=\s*"py(\d)(\d+)"', (REPO_ROOT / "ruff.toml").read_text(), join="."
        ),
        "ty.toml:python-version": _sole_match(
            r'python-version\s*=\s*"(\d+\.\d+)"', (REPO_ROOT / "ty.toml").read_text()
        ),
        "pyrightconfig.json:pythonVersion": _sole_match(
            r'"pythonVersion"\s*:\s*"(\d+\.\d+)"', (REPO_ROOT / "pyrightconfig.json").read_text()
        ),
        "Makefile:uv --python": _sole_match(r"uv sync[^\n]*--python\s+(\d+\.\d+)", makefile_text()),
    }
    assert len(set(declared.values())) == 1, f"the interpreter pins disagree: {declared}"


def test_the_running_interpreter_matches_the_declared_pin():
    """A pin nothing runs on is a comment. This is the half the string checks cannot see."""
    import sys

    declared = (REPO_ROOT / ".python-version").read_text().strip()
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert running == declared, (
        f"`.python-version` declares {declared} but this suite is running on {running}"
    )


def test_coverage_measures_branches_with_the_sysmon_core():
    """Can this interpreter measure branches under `sys.monitoring` at all?

    On interpreters before 3.14 coverage falls back to its C tracer whenever `branch = true`, and
    it does so without a warning. The two attributes are coverage internals: a coverage release
    that renames them fails here loudly, which is the signal this test exists to give.
    """
    from coverage import env

    assert env.PYBEHAVIOR.branch_right_left, (
        "this interpreter cannot measure branches under sys.monitoring"
    )
    assert env.SYSMON_DEFAULT, "coverage would not choose sysmon by default on this interpreter"


def test_the_covered_run_is_actually_using_the_sysmon_core():
    """Which core did *this* run resolve to?

    The capability assertions above read import-time constants, so they pass on a run that fell
    back through `COVERAGE_CORE`, `core =`, `timid`, `dynamic_context` or `concurrency`. This reads
    the resolved core from the public `Coverage.current()` API. It skips without coverage, so a
    `--no-cov` run still works.
    """
    import coverage

    current = coverage.Coverage.current()
    if current is None:
        pytest.skip("not running under coverage, so there is no resolved core to check")
    core = dict(current.sys_info()).get("core")
    assert core == "SysMonitor", (
        f"this run is measuring coverage on {core!r} rather than SysMonitor; check COVERAGE_CORE "
        f"in the environment and `core`/`timid`/`dynamic_context`/`concurrency` under "
        f"[tool.coverage.run]"
    )


def _requirement(spec: str, extra: str | None) -> tuple[str, str | None, str]:
    """``(normalized name, extra, specifier)`` for one PEP 508 string from `pyproject.toml`."""
    match = re.fullmatch(r"\s*([A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*([^;]*?)\s*", spec)
    assert match is not None, f"unparseable requirement in pyproject.toml: {spec!r}"
    return normalize(match.group(1)), extra, match.group(2).replace(" ", "")


def declared_requirements() -> set[tuple[str, str | None, str]]:
    """Every requirement `pyproject.toml` declares, core and optional."""
    project = pyproject()["project"]
    found = {_requirement(spec, None) for spec in project.get("dependencies", [])}
    for extra, specs in project.get("optional-dependencies", {}).items():
        found |= {_requirement(spec, extra) for spec in specs}
    return found


def locked_requirements() -> set[tuple[str, str | None, str]]:
    """The requirements `uv.lock` recorded for the project when it was last locked."""
    packages = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))["package"]
    root = [p for p in packages if p.get("source", {}).get("editable") == "."]
    assert len(root) == 1, f"expected one editable root package in uv.lock, found {len(root)}"
    found: set[tuple[str, str | None, str]] = set()
    for entry in root[0]["metadata"]["requires-dist"]:
        extra = re.fullmatch(r"extra == '([^']+)'", entry.get("marker", ""))
        found.add(
            (
                normalize(entry["name"]),
                extra.group(1) if extra else None,
                entry.get("specifier", ""),
            )
        )
    return found


def test_the_lock_records_exactly_the_declared_requirements():
    """`uv.lock` was locked against the `pyproject.toml` in this tree.

    The gate installs with `--locked` and runs with `--no-sync`, so nothing in it re-locks. A
    requirement edited without `uv lock` therefore shows up here, offline, as a difference between
    what `pyproject.toml` declares and what the lock recorded.
    """
    declared = declared_requirements()
    assert declared, "parsed no requirements out of pyproject.toml, so this test proves nothing"
    locked = locked_requirements()
    assert declared == locked, (
        f"pyproject.toml and uv.lock disagree; run `uv lock`. Only declared: "
        f"{sorted(declared - locked, key=str)}; only locked: {sorted(locked - declared, key=str)}"
    )


def stray_tests(paths: list[str]) -> list[str]:
    """The tracked test files outside `tests/`, ignoring dot-directories and `scratchpad/`."""
    return sorted(
        path
        for path in paths
        if "tests" not in Path(path).parts
        and not any(part.startswith(".") or part == "scratchpad" for part in Path(path).parts)
    )


def test_no_test_file_lives_outside_the_tests_tree():
    """`ruff.toml` carries one test glob, so a test file elsewhere is held to library rules."""
    require_toplevel(REPO_ROOT)
    listed = run(["ls-files", "--", "*test_*.py", "*_test.py"], root=REPO_ROOT, check=True)
    lines = listed.stdout.splitlines()
    assert "tests/test_repo_gate.py" in lines, "the listing found no test file; it is broken"
    stray = stray_tests(lines)
    assert not stray, f"test files outside `tests/` escape `ruff.toml`'s test glob: {stray}"


def test_the_stray_test_filter_reports_a_test_outside_the_tests_tree():
    """The positive control the committed tree cannot supply."""
    paths = ["tests/test_a.py", "tools/test_b.py", "scratchpad/test_c.py", ".venv/x/test_d.py"]
    assert stray_tests(paths) == ["tools/test_b.py"]


PINNED_LINTERS = ("ruff", "ty")
"""The two tools `make lint` and `make typecheck` run. Both must be exact."""


def pinned_version(tool: str) -> str:
    """The exact version the `dev` extra pins for ``tool``.

    Args:
        tool: Distribution name as it appears in the dependency list.

    Returns:
        The pinned version, without the ``==``.
    """
    specs = [
        requirement for requirement in declared_requirements() if requirement[0] == normalize(tool)
    ]
    assert len(specs) == 1, f"`{tool}` is declared {len(specs)} times in pyproject.toml: {specs}"
    _, extra, specifier = specs[0]
    assert extra == "dev", f"`{tool}` belongs in the `dev` extra, not {extra!r}"
    exact = re.fullmatch(r"==([0-9][^,]*)", specifier)
    assert exact is not None, f"`{tool}` is not pinned exactly: {specifier!r}"
    return exact.group(1)


def test_the_linters_are_pinned_exactly_rather_than_floated():
    """A floating linter can turn the gate red with no commit in between."""
    for tool in PINNED_LINTERS:
        pinned_version(tool)


def test_the_pinned_linters_are_the_ones_actually_installed():
    """`make lint` runs the environment's ruff, not `pyproject.toml`'s; the two must agree.

    Fix a failure with `make setup`, which installs what `uv.lock` records.
    """
    for tool in PINNED_LINTERS:
        pinned = pinned_version(tool)
        installed = version(tool)
        assert installed == pinned, (
            f"`{tool}` {installed} is installed while pyproject.toml pins {pinned}; "
            "run `make setup`"
        )


# Written as character classes so this module does not match itself.
CONFLICT_MARKER = re.compile(r"^(?:[<]{7} |[=]{7}$|[>]{7} )")


def test_no_tracked_file_carries_a_merge_conflict_marker():
    """A committed marker is never correct, and in a non-Python file no other gate sees it."""
    found = [
        f"{path.relative_to(REPO_ROOT)}:{n}: {line[:40]}"
        for path in TRACKED_TEXT
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
        if CONFLICT_MARKER.match(line)
    ]
    assert not found, f"tracked file(s) carry merge conflict markers: {found}"


def test_the_conflict_marker_scan_reaches_the_files_it_is_for():
    """A `git ls-files` that stopped matching would make the check above a silent no-op."""
    assert len(TRACKED_TEXT) >= 30, f"only {len(TRACKED_TEXT)} tracked text file(s) discovered"
    covered = {p.relative_to(REPO_ROOT).as_posix() for p in TRACKED_TEXT}
    witnesses = (
        "pyproject.toml",
        "src/bko/__init__.py",
        "tools/README.md",
        "tests/conftest.py",
        ".gitignore",
        "Makefile",
    )
    for witness in witnesses:
        assert witness in covered, f"{witness} is not scanned, so its root is not being walked"

    marker = "<" * 7 + " HEAD"
    assert CONFLICT_MARKER.match(marker), "the pattern no longer matches a real marker"
    assert not CONFLICT_MARKER.match("<<< not a marker"), "the pattern is too loose"


ROOT_WITHOUT_SUFFIX = frozenset({"Makefile"})
"""The only top-level tracked files that legitimately carry no extension.

A closed set rather than a pattern, because the failure this catches is a name nobody chose: a
shell redirect that got committed.
"""


def test_no_unnamed_scratch_file_is_tracked_at_the_repository_root():
    """An extension-less root file is a committed redirect unless it is named here."""
    top_level = [name for name in tracked_files() if "/" not in name]
    assert top_level, "no top-level tracked files discovered; the listing is broken"
    unexpected = sorted(
        name for name in top_level if "." not in name and name not in ROOT_WITHOUT_SUFFIX
    )
    assert not unexpected, (
        f"{unexpected} are tracked at the repository root with no extension. If one is a stray "
        "redirect, `git rm` it; if it is deliberate, add it to ROOT_WITHOUT_SUFFIX"
    )
    for expected in ROOT_WITHOUT_SUFFIX:
        assert expected in top_level, f"{expected} is allowed but no longer tracked"


AGENT_IMPORTS = (("CLAUDE.md", "@AGENTS.md"), ("GEMINI.md", "@./AGENTS.md"))
"""Each tool file and the import line it must open with (ADR `tooling/agent-config-single-home`)."""


@pytest.mark.parametrize(("name", "line"), AGENT_IMPORTS, ids=[name for name, _ in AGENT_IMPORTS])
def test_claude_and_gemini_import_agents_md(name: str, line: str):
    """`AGENTS.md` is the single home; each tool file reaches it through its first line."""
    lines = (REPO_ROOT / name).read_text(encoding="utf-8").splitlines()
    assert lines, f"{name} is empty, so it imports nothing"
    assert lines[0] == line, f"{name} opens with {lines[0]!r}, not the import {line!r}"


ALLOWED_COMMANDS = frozenset(
    f"Bash(make {target})"
    for target in ("lint", "format", "typecheck", "test", "sizes", "synced", "rebased", "check")
)
"""The committed Claude Code allow list (ADR `tooling/agent-config-single-home`)."""


def test_the_committed_allow_list_is_exactly_the_make_targets():
    """A committed grant pre-approves a command for everyone who clones the repository.

    Exact equality, so a widened entry, a wildcard or an install target (`setup`, `sync`) fails
    here rather than landing as a standing grant.
    """
    settings = json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))
    allowed = settings["permissions"]["allow"]
    assert len(allowed) == len(set(allowed)), f"duplicate allow entries: {allowed}"
    assert set(allowed) == ALLOWED_COMMANDS, f"unexpected allow list: {sorted(allowed)}"
    assert not any("*" in entry for entry in allowed), "a wildcard grant is in the allow list"
