"""The file-size cap, and the checker that enforces it.

`tools/check_file_size.py` is the only gate in this repository whose subject is the repository
itself rather than the library, and its failure mode is a scan that matches nothing: it exits 0 and
reads exactly like a clean result. So every clause below either drives the predicate over a
synthesized tree or pins the exit code a Makefile depends on.

**No clause here restates a cap.** Every fixture derives its length from `BY_PREFIX`, so a cap
change moves the tree and the tests together instead of leaving the tests asserting a boundary the
checker no longer has. The one property worth pinning about the values, that tests get a looser cap
than source, is asserted as an inequality. Prefer `BY_PREFIX[prefix] + n` to a figure, and where a
fixture needs an ordering between two caps, assert the ordering as well so the fixture cannot
silently stop posing its question.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests._corpus import REPO_ROOT
from tools.check_file_size import (
    CAPS,
    DISCOVERY_FLOOR,
    UnmeasuredPathError,
    cap_for,
    line_count,
    main,
    offenders,
    tracked_python_files,
)


def test_the_repository_is_within_its_own_caps():
    """The rule the checker exists to make true, asserted on the real tree."""
    found, _ = offenders(REPO_ROOT)
    assert not found, (
        f"{[f'{name}: {lines} > {cap}' for name, lines, cap in found]} are over cap. Split them, "
        "or change the cap in tools/check_file_size.py and say why."
    )


def test_the_scan_finds_the_tracked_python_tree():
    """Anti-vacuity. Every clause here is `assert not offenders`, which an empty scan satisfies.

    Compared against `DISCOVERY_FLOOR` on purpose, which makes this clause unable to catch a change
    to the floor itself: both sides move together. It is here to catch a broken scan, not a mis-set
    constant; `test_main_returns_two_when_the_scan_finds_almost_nothing` holds the floor's own
    behaviour, by driving the refusal rather than re-reading the number.
    """
    found = tracked_python_files(REPO_ROOT)
    assert len(found) > DISCOVERY_FLOOR, (
        f"only {len(found)} tracked Python files were discovered, down at the floor. The scan is "
        "broken, not the tree small. This clause is deliberately one stricter than the checker, "
        f"which refuses below {DISCOVERY_FLOOR} rather than at it."
    )
    assert all(name.endswith(".py") for name in found), "the listing is filtered to Python"


def test_the_listing_is_repo_relative():
    """Every name joins onto the root, so `root / name` is the file the cap applies to."""
    found = tracked_python_files(REPO_ROOT)
    assert all(not name.startswith(("..", "/")) for name in found)
    assert (REPO_ROOT / found[0]).is_file()


def test_a_root_that_is_not_the_git_toplevel_is_refused():
    """`git ls-files` reports relative to the toplevel, so another root would misjoin every path.

    Clone this repository inside another one and git's toplevel is the outer repo; the listing
    then describes a tree this checker is not measuring. Refusing names that cause, where the
    alternative reported every path as matching no rule and blamed `CAPS`.
    """
    with pytest.raises(RuntimeError, match="not this git repository's toplevel"):
        tracked_python_files(REPO_ROOT / "src" / "bko")


def test_the_scan_ignores_a_hijacked_git_environment(monkeypatch):
    """`GIT_INDEX_FILE` and friends outrank `cwd`, and a hijack shrinks the scan silently.

    With `GIT_INDEX_FILE` pointing at a copy of the index with two files removed, the checker would
    scan two fewer files and print its success line: a gate exiting 0 while not measuring part of
    the tree. `DISCOVERY_FLOOR` cannot catch that, because it guards the empty end and a hijack
    produces a *partial* shrink. Stripping every `GIT_*` key from the child environment closes it.

    Args:
        monkeypatch: To set the hijacking variables for this test only.
    """
    honest = tracked_python_files(REPO_ROOT)
    monkeypatch.setenv("GIT_DIR", "/nonexistent/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/nonexistent")
    assert tracked_python_files(REPO_ROOT) == honest, (
        "a GIT_* variable changed what the gate measured, so an operator's environment decides "
        "which files are checked"
    )


BY_PREFIX = dict(CAPS)


@pytest.mark.parametrize(
    ("path", "prefix"),
    [
        ("tests/test_example.py", "tests/"),
        ("tests/nested/test_example.py", "tests/"),
        ("src/bko/example.py", "src/"),
        ("tools/check_file_size.py", "tools/"),
    ],
)
def test_each_tree_selects_the_rule_for_its_own_prefix(path: str, prefix: str):
    """One case per prefix in `CAPS`, reading the expected cap out of `CAPS` rather than writing it.

    So this pins the *selection* and leaves the values with their single owner. It does not pin the
    table's order: the prefixes are disjoint, so first-match-wins has no observable effect and
    reversing `CAPS` survives this file. Ordering becomes load-bearing the day a nested prefix is
    added, and the case for it belongs in that change.

    Args:
        path: A repo-relative path.
        prefix: The `CAPS` entry that must claim it.
    """
    assert cap_for(path) == BY_PREFIX[prefix]


def test_tests_get_a_looser_cap_than_source():
    """The one thing worth asserting about the values, and it is a relation rather than a number.

    A test file's length tracks how many cases it holds rather than how many ideas, which is the
    reason there are two caps. Written as an inequality so it says that without becoming a second
    home for either figure.
    """
    assert BY_PREFIX["tests/"] > BY_PREFIX["src/"]
    assert BY_PREFIX["src/"] == BY_PREFIX["tools/"], (
        "source and the tooling share one cap; splitting them needs a stated reason"
    )


def test_a_prefix_matches_only_at_the_start_of_a_path():
    """`startswith`, not `in`. A substring test would claim a path no rule is meant to cover.

    Every path `git ls-files` returns is anchored at position 0, so nothing in this tree is named
    to collide; this is the case that keeps the anchoring deliberate rather than incidental.
    """
    assert cap_for("vendor/tests/thing.py") is None
    assert cap_for("docs/src/example.py") is None
    assert cap_for("srcfoo/mod.py") is None, "a prefix is a directory, not a character run"


def test_the_checker_is_under_its_own_cap():
    """A size checker that exempts itself would do so invisibly."""
    cap = cap_for("tools/check_file_size.py")
    assert cap is not None, "the checker is under no rule, so it measures everything but itself"
    assert line_count(REPO_ROOT / "tools" / "check_file_size.py") <= cap


def test_a_path_under_no_rule_is_a_structural_failure_not_a_pass():
    """An unrecognized tracked directory must fail loudly rather than go unmeasured."""
    assert cap_for("docs/conf.py") is None
    assert cap_for("conftest.py") is None


def test_the_boundary_is_at_the_cap_not_one_either_side(tmp_path, monkeypatch):
    """`> cap`, so a file exactly at the cap passes and one line more does not.

    Driven through `offenders`, not through `line_count`: asserting only the counting helper would
    leave the comparison untested, and mutating `offenders`' `>` to `>=` would turn every at-cap
    file in the tree into a violation.

    Args:
        tmp_path: A directory to synthesize files in.
        monkeypatch: To replace the `git ls-files` call with a fixed listing.
    """
    cap = BY_PREFIX["src/"]
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "at_cap.py").write_bytes(b"\n" * cap)
    (tmp_path / "src" / "under.py").write_bytes(b"\n" * (cap - 1))
    monkeypatch.setattr(
        "tools.check_file_size.tracked_python_files",
        lambda root: ["src/at_cap.py", "src/under.py"],
    )
    found, _ = offenders(tmp_path)
    assert found == [], "a file exactly at the cap is within it"

    (tmp_path / "src" / "at_cap.py").write_bytes(b"\n" * (cap + 1))
    found, _ = offenders(tmp_path)
    assert found == [("src/at_cap.py", cap + 1, cap)], "one line more is not"


def test_the_count_matches_wc_l_on_a_file_with_no_trailing_newline(tmp_path):
    """`data.count(b"\\n")`, not `len(splitlines())`: the two disagree here and `wc -l` decides.

    `ruff format` guarantees a trailing newline across this tree, so nothing under the cap
    distinguishes them. That is a fact about the tree, not about the counting, and this stops
    someone "simplifying" the implementation into the other one.

    Args:
        tmp_path: A directory to synthesize a file in.
    """
    ragged = tmp_path / "ragged.py"
    ragged.write_bytes(b"one\ntwo\nthree")

    assert len(ragged.read_text(encoding="utf-8").splitlines()) == 3
    assert line_count(ragged) == 2, "wc -l counts newlines, and the last line has none"

    # Fixed argv, and the only interpolated word is a `tmp_path` this test just wrote.
    measured = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["wc", "-l", str(ragged)], capture_output=True, text=True, check=True
    ).stdout
    assert int(measured.split()[0]) == line_count(ragged)


def test_an_over_cap_file_is_reported_worst_first(tmp_path, monkeypatch):
    """The predicate itself, driven over a synthesized tree rather than waiting for a violation.

    Args:
        tmp_path: A directory to synthesize a tree in.
        monkeypatch: To replace the `git ls-files` call with a fixed listing.
    """
    src_cap = BY_PREFIX["src/"]
    tests_cap = BY_PREFIX["tests/"]
    big, huge = src_cap + 200, src_cap + 700
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "small.py").write_bytes(b"\n" * 10)
    (tmp_path / "src" / "big.py").write_bytes(b"\n" * big)
    (tmp_path / "src" / "huge.py").write_bytes(b"\n" * huge)
    # Longer than either src offender and still within the looser tests cap, which is what makes
    # this a test of per-tree selection rather than of one global number. Sized from `CAPS` so the
    # relation survives a change to either cap.
    long_test = tests_cap - 100
    (tmp_path / "tests" / "long.py").write_bytes(b"\n" * long_test)
    assert huge > big > src_cap, "neither src fixture is over its cap"
    assert tests_cap > long_test > big, (
        "the test fixture must out-length an src offender while staying inside its own cap, or "
        "this stops distinguishing per-tree selection from one global number"
    )

    listing = ["src/small.py", "src/big.py", "src/huge.py", "tests/long.py"]
    monkeypatch.setattr("tools.check_file_size.tracked_python_files", lambda root: list(listing))

    found, scanned = offenders(tmp_path)
    assert scanned == 4
    assert found == [("src/huge.py", huge, src_cap), ("src/big.py", big, src_cap)], (
        "the worst offender comes first, and a longer test file is not one at all"
    )


def test_a_tracked_file_under_no_rule_raises_rather_than_passing(tmp_path, monkeypatch):
    """A new top-level directory must stop the gate, not slip through it.

    Args:
        tmp_path: A directory to synthesize a tree in.
        monkeypatch: To replace the `git ls-files` call.
    """
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "mod.py").write_text("\n" * 5, encoding="utf-8")
    monkeypatch.setattr(
        "tools.check_file_size.tracked_python_files", lambda root: ["elsewhere/mod.py"]
    )

    with pytest.raises(UnmeasuredPathError, match="under no rule in CAPS"):
        offenders(tmp_path)


def test_a_tracked_symlink_is_refused_rather_than_followed(tmp_path, monkeypatch):
    """A symlink's length is a property of its target, which may be outside the repository.

    `read_bytes` follows it, so the gate would report a number measured somewhere it does not
    govern, or block forever on a fifo. The tree holds no tracked symlink, so this is the only
    clause that drives the refusal.

    Args:
        tmp_path: A directory to synthesize a tree in.
        monkeypatch: To replace the listing.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "outside.py").write_bytes(b"\n" * 10)
    (tmp_path / "src" / "link.py").symlink_to(tmp_path / "outside.py")
    monkeypatch.setattr("tools.check_file_size.tracked_python_files", lambda root: ["src/link.py"])

    with pytest.raises(UnmeasuredPathError, match="tracked symlink"):
        offenders(tmp_path)


def test_main_returns_one_when_a_file_is_over_cap(tmp_path, monkeypatch, capsys):
    """Exit 1 is the whole point of the gate and the real tree never exercises it.

    `make sizes` reads only the process exit code, so a `main` that found offenders and returned 0
    would be a gate that reports and does not fail.

    Args:
        tmp_path: A directory to synthesize a tree in.
        monkeypatch: To replace the listing and point the checker at the synthetic root.
        capsys: To read the offender lines.
    """
    cap = BY_PREFIX["src/"]
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "big.py").write_bytes(b"\n" * (cap + 200))
    monkeypatch.setattr("tools.check_file_size.REPO_ROOT", tmp_path)
    monkeypatch.setattr("tools.check_file_size.tracked_python_files", lambda root: ["src/big.py"])
    monkeypatch.setattr("tools.check_file_size.DISCOVERY_FLOOR", 1)

    assert main() == 1
    out = capsys.readouterr().out
    assert "src/big.py" in out
    assert "over by 200" in out


def test_main_returns_two_when_the_scan_finds_almost_nothing(tmp_path, monkeypatch, capsys):
    """The anti-vacuity branch, which is the reason `DISCOVERY_FLOOR` exists.

    The floor's own clause elsewhere in this file compares a discovered count against the constant
    it imports, so it moves in lockstep and no value of the floor can fail it. This is the control
    that exercises the refusal: a scan matching almost nothing must exit 2, not 0.

    Args:
        tmp_path: A directory to synthesize a tree in.
        monkeypatch: To replace the listing and the root.
        capsys: To read the refusal.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_bytes(b"\n" * 10)
    monkeypatch.setattr("tools.check_file_size.REPO_ROOT", tmp_path)
    monkeypatch.setattr("tools.check_file_size.tracked_python_files", lambda root: ["src/one.py"])

    assert main() == 2, "one file found is a broken scan, not a small repository"
    assert "below the floor" in capsys.readouterr().out


def test_main_returns_two_when_a_tracked_file_is_under_no_rule(tmp_path, monkeypatch, capsys):
    """A new top-level directory stops the gate rather than going unmeasured.

    Args:
        tmp_path: A directory to synthesize a tree in.
        monkeypatch: To replace the listing and the root.
        capsys: To read the refusal.
    """
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "mod.py").write_bytes(b"\n" * 5)
    monkeypatch.setattr("tools.check_file_size.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "tools.check_file_size.tracked_python_files", lambda root: ["elsewhere/mod.py"]
    )

    assert main() == 2
    assert "under no rule in CAPS" in capsys.readouterr().out


def test_main_returns_two_when_git_cannot_list(tmp_path, monkeypatch, capsys):
    """A failed `git ls-files` must not read as a clean tree.

    Args:
        tmp_path: An empty directory to point the checker at.
        monkeypatch: To make the listing raise as `git` failing would.
        capsys: To read the refusal.
    """

    def boom(root):
        raise subprocess.CalledProcessError(128, ["git", "ls-files"])

    monkeypatch.setattr("tools.check_file_size.REPO_ROOT", tmp_path)
    monkeypatch.setattr("tools.check_file_size.tracked_python_files", boom)

    assert main() == 2
    assert "could not list tracked files" in capsys.readouterr().out


def test_main_returns_two_when_the_root_is_not_the_git_toplevel(tmp_path, monkeypatch, capsys):
    """The toplevel guard raises, and `main` must turn that into a refusal rather than a pass.

    A separate branch from the `CalledProcessError` one beside it: git succeeds here and the
    checker rejects git's answer, which is the case a clone inside another repository produces.

    Args:
        tmp_path: A directory to point the checker at.
        monkeypatch: To make the listing raise as the guard would.
        capsys: To read the refusal.
    """

    def boom(root):
        raise RuntimeError("not this git repository's toplevel")

    monkeypatch.setattr("tools.check_file_size.REPO_ROOT", tmp_path)
    monkeypatch.setattr("tools.check_file_size.tracked_python_files", boom)

    assert main() == 2
    assert "could not list tracked files" in capsys.readouterr().out


def test_the_documented_exit_codes_are_what_the_process_returns():
    """The Makefile reads these and nothing else does, so they are the contract.

    Run as a subprocess on purpose: `main` returning 0 and the process exiting 0 are different
    claims, and `make sizes` depends on the second.
    """
    result = subprocess.run(
        [sys.executable, "-m", "tools.check_file_size"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "none over cap" in result.stdout
    assert f"against {len(CAPS)} rules" in result.stdout, (
        "the success line names how many rules were applied. Matched as a phrase, not as a bare "
        f"digit: {len(CAPS)} can occur inside the file count too, so the loose form would pass "
        "against a synthesized line reading `against 0 rules`."
    )


def test_the_success_line_names_what_was_checked():
    """A gate that prints nothing on success cannot be told from a gate that ran nothing.

    A silent pass and a skipped pass look identical from the outside, so the count is the evidence
    the run happened.
    """
    result = subprocess.run(
        [sys.executable, "-m", "tools.check_file_size"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    scanned = len(tracked_python_files(REPO_ROOT))
    assert f"{scanned} tracked Python files checked" in result.stdout
