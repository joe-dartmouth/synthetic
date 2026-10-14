"""The two `tools/` checkers `make check` runs before anything else, driven rather than described.

These run the checkers against fixture repositories and a repointed lock, so they establish that
each verdict is reachable in both directions.

**There is deliberately no test asserting that *this machine* agrees with the lock.** Inside
`make check` it would be dead weight: `synced` is a prerequisite of `check`, so by the time `test`
runs the same assertion has already passed. Outside the gate a drifted machine would get a pytest
failure, which reads as a repository defect, and that misattribution is the thing `check_lock_drift`
exists to end. `make synced` owns the live verdict; these tests own whether the checker works.

**Every checker here has an anti-vacuity arm, and that is the point of testing them at all.** All
three of the `tools/` checkers share one failure mode: a scan that matches nothing exits clean and
reads exactly like a clean tree. `check_file_size.py` answers it with `DISCOVERY_FLOOR`, driven in
`tests/test_file_size.py`, and `check_lock_drift.py` with `COMPARISON_FLOOR`. `check_rebased.py`
cannot have a floor, because "no trunk ref" is a legitimate pass, so it answers the same concern
structurally: it refuses when git answers about a different repository, and when git could not be
consulted at all. Both of those are driven below.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import tomllib
from importlib.metadata import distributions, version

import pytest

from tests._git_fixtures import (
    advance_remote,
    init_repo,
    rebased_verdict,
    run_git,
    with_remote,
)


def _installed():
    """Every distribution installed in the environment running this suite.

    Returns:
        The distribution objects, for building a fixture lock that names real packages.
    """
    return list(distributions())


def test_the_rebase_check_passes_when_the_trunk_is_an_ancestor(tmp_path, capsys):
    """The ordinary case: a branch that has seen everything on the trunk.

    Both directions matter: a check that passed unconditionally would be indistinguishable from
    this, and the failing case below is what shows it is not.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    (repo / "branch.txt").write_text("two\n", encoding="utf-8")
    run_git(repo, "add", "branch.txt")
    run_git(repo, "commit", "-m", "two")

    # This fixture has no remote, so freshness cannot be established and the pass is reported on
    # stderr with its reason. The ancestry verdict is what this case is about; the stream and the
    # reason are `test_..._freshness_was_not_established_without_a_remote`'s.
    assert rebased_verdict(repo) == 0
    assert trunk[:8] in capsys.readouterr().err


def test_the_rebase_check_refuses_a_branch_the_trunk_has_moved_past(tmp_path, monkeypatch, capsys):
    """The state the check exists for, in miniature.

    The branch commits, then the trunk advances independently, so neither contains the other. Two
    branches that are each green can merge into a red trunk from that state, and a gate run on the
    branch cannot see it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    base = init_repo(repo)
    (repo / "branch.txt").write_text("two\n", encoding="utf-8")
    run_git(repo, "add", "branch.txt")
    run_git(repo, "commit", "-m", "two")

    run_git(repo, "switch", "--detach", base)
    (repo / "trunk.txt").write_text("three\n", encoding="utf-8")
    run_git(repo, "add", "trunk.txt")
    run_git(repo, "commit", "-m", "three")
    advanced = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    run_git(repo, "update-ref", "refs/remotes/origin/main", advanced)
    run_git(repo, "switch", "main")

    assert rebased_verdict(repo) == 1
    assert "Run `git fetch origin && git rebase origin/main`" in capsys.readouterr().err


def test_the_rebase_check_refuses_a_repository_with_no_commits(tmp_path, capsys):
    """An unborn HEAD has no commit to compare, so it is structural rather than a pass.

    The message carries git's status, and says what an unborn branch is, because "HEAD does not
    resolve" alone does not tell the operator that committing is the fix.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "--initial-branch", "main")

    assert rebased_verdict(repo) == 2
    failure = capsys.readouterr().err
    assert "HEAD does not resolve to a commit (git exited with status " in failure
    assert "the branch has no commits yet" in failure


def test_a_git_that_cannot_run_for_head_is_not_reported_as_an_unborn_branch(
    tmp_path, monkeypatch, capsys
):
    """A git that did not run says so; it must not be stated as a fact about HEAD."""
    import tools.check_rebased as subject

    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    real = subject._git

    def unrunnable(*args, root):
        if args == ("rev-parse", "HEAD"):
            return subject.COULD_NOT_RUN, ""
        return real(*args, root=root)

    monkeypatch.setattr(subject, "_git", unrunnable)

    assert subject.main(repo) == 2
    failure = capsys.readouterr().err
    assert "git could not be consulted about HEAD" in failure
    assert "does not resolve to a commit" not in failure


def test_the_rebase_check_does_not_apply_without_a_trunk_ref(tmp_path, capsys):
    """A clone with no remote must still be able to run the gate.

    Refusing here would make `make check` unrunnable in a place where the check has nothing to say,
    such as a fresh clone with no remote or a fork whose default branch has another name, which is
    a worse failure than not checking. Exit 0 with the reason stated, not exit 2.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    run_git(repo, "update-ref", "-d", "refs/remotes/origin/main")

    assert rebased_verdict(repo) == 0
    assert "does not apply" in capsys.readouterr().out


def test_the_rebase_check_reports_a_non_repository_as_structural(tmp_path, monkeypatch, capsys):
    """A directory that is not a checkout cannot answer the question, so it is not a pass.

    The reasoning `tools/check_file_size.py`'s `DISCOVERY_FLOOR` carries: a check that could not run
    must not exit clean, because clean and "nothing to report" read identically.
    """
    bare = tmp_path / "not-a-repo"
    bare.mkdir()

    assert rebased_verdict(bare) == 2
    assert "not a git checkout" in capsys.readouterr().err


def test_the_rebase_check_reports_a_nested_non_repository_as_structural(tmp_path, capsys):
    """git discovers a repository by walking upward, and `cwd` is not a trust boundary.

    A copy of the tree that is not itself a checkout but sits inside one would otherwise get a
    confident verdict about the *outer* repository: exit 0 printing a SHA from a tree that was never
    measured, or exit 1 blaming a branch that does not exist. `cwd=root` does not make git use a
    repository at `root`.

    The sibling test above reaches exit 2 only because `tmp_path` happens to sit outside any
    checkout; on a machine whose `TMPDIR` is inside one it would measure the enclosing repo. This
    is the case that pins the behaviour rather than the ambient layout.
    """
    outer = tmp_path / "outer"
    outer.mkdir()
    init_repo(outer)
    inner = outer / "inner"
    inner.mkdir()

    assert rebased_verdict(inner) == 2
    assert "walking upward" in capsys.readouterr().err


def test_the_rebase_check_does_not_apply_mid_rebase(tmp_path, capsys):
    """A conflicted rebase leaves HEAD at the trunk, so the ancestry test would pass falsely.

    Mid-rebase the working tree is neither the branch's nor the result's, and the message this
    check prints on success is the strongest sentence in the module: "the measured tree is the
    tree that will land". Abstaining is the honest answer, and it goes to stderr: exit 0 with
    nothing measured is a run the operator must read, which `make check >/dev/null` hides on stdout.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    (repo / ".git" / "rebase-merge").mkdir()

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert "rebase is in progress" in captured.err
    assert captured.out == ""


def test_the_rebase_check_does_not_apply_mid_bisect(tmp_path, capsys):
    """Running the suite is what a bisect is *for*, and the rebase advice would destroy it.

    A bisect checks out an older commit, so the ancestry test fails and the failure prints
    `git rebase`, which ends the bisect. The one state where the check's own remedy is destructive
    is the one state it must not have an opinion in.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    (repo / ".git" / "BISECT_LOG").write_text("", encoding="utf-8")

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert "bisect is in progress" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("failure", ["exited", "did-not-run"])
def test_a_git_directory_git_cannot_report_is_structural(tmp_path, monkeypatch, capsys, failure):
    """Without the git directory, whether a rebase or bisect is in progress is unknown.

    Proceeding would run the ancestry test mid-rebase, where a pass describes a tree that is
    neither the branch's nor the result's. The message says whether git did not run or which
    status it exited with.
    """
    import tools.check_rebased as subject

    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    (repo / ".git" / "rebase-merge").mkdir()
    status = 128 if failure == "exited" else subject.COULD_NOT_RUN
    real = subject._git

    def broken(*args, root):
        if args == ("rev-parse", "--absolute-git-dir"):
            return status, ""
        return real(*args, root=root)

    monkeypatch.setattr(subject, "_git", broken)

    assert subject.main(repo) == 2
    expected = "git exited with status 128" if failure == "exited" else "git did not run"
    assert f"could not report this checkout's git directory ({expected})" in capsys.readouterr().err


def test_a_git_that_cannot_run_is_structural_at_every_call_site(tmp_path, monkeypatch, capsys):
    """A git failure must never become a verdict.

    A timeout or an unexecutable git collapsed into git's own exit status 1 would be read by the
    trunk lookup as "the ref does not exist", and that branch returns **0**: a hung git would then
    produce a clean gate and a printed claim about refs that was never measured.

    `COULD_NOT_RUN` is negative precisely so no call site can confuse it with a status git returns.
    This drives the failure at each successive call, which shows the sentinel is honoured
    everywhere rather than at the first site only.

    **The number of call sites is counted rather than written down**, because a bare literal
    silently stops covering whatever a change adds. One instrumented run establishes the bound for
    this fixture; the sites that need a remote are driven by
    `tests/test_rebased_second_remote.py`, which fails if any call site goes undriven.
    """
    import tools.check_rebased as subject

    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    real = subject._git

    reached = {"n": 0}

    def counting(*args, root):
        reached["n"] += 1
        return real(*args, root=root)

    monkeypatch.setattr(subject, "_git", counting)
    subject.main(repo)
    capsys.readouterr()
    sites = reached["n"]
    # No floor is counted from the source text here. `tests/test_rebased_second_remote.py` drives
    # every call site by line, across fixtures that reach each one, and fails when one goes
    # undriven; this sweep only has to cover the sites a checkout with no remote reaches, and must
    # reach some.
    assert sites, "the fixture reached no `_git` call, so the sweep below would prove nothing"

    for fail_after in range(sites):
        calls = {"n": 0}

        def fake(*args, root, _limit=fail_after, _seen=calls):
            _seen["n"] += 1
            if _seen["n"] > _limit:
                return subject.COULD_NOT_RUN, ""
            return real(*args, root=root)

        monkeypatch.setattr(subject, "_git", fake)
        assert subject.main(repo) == 2, (
            f"a git failure at call {fail_after + 1} did not report as structural; a verdict "
            "of 0 here would be a claim about refs that was never measured"
        )
        assert "could not" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure",
    [OSError("git is not on PATH"), subprocess.TimeoutExpired(["git"], 10)],
    ids=["unexecutable", "hung"],
)
def test_a_git_that_cannot_answer_the_toplevel_question_is_structural(
    tmp_path, monkeypatch, capsys, failure
):
    """The call site the test above cannot reach.

    `_preflight` asks the toplevel question through `tools._git.toplevel_of` rather than through
    this module's `_git`, because it needs "git could not run", "not a checkout" and "wrong tree"
    apart while `toplevel_of` reports the first by **raising** rather than by returning the
    sentinel. So the `_git` sweep above does not walk this site.

    Both exception types are driven: `run` raises `OSError` when git cannot be executed at all and
    `TimeoutExpired` when it hangs, and a handler naming one of the two turns the other into a
    traceback out of `make check`.
    """
    import tools.check_rebased as subject

    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)

    def fake(root):
        raise failure

    monkeypatch.setattr(subject, "toplevel_of", fake)

    assert subject.main(repo) == 2
    assert "could not be run at all" in capsys.readouterr().err


def test_the_lock_drift_check_fails_rather_than_only_reporting(tmp_path, monkeypatch, capsys):
    """`make synced` reads only the exit code, so exit 1 is the whole gate.

    Driving `drift()` directly would leave `main`'s report-versus-fail wiring untested, so a
    checker that printed every drifted package and exited clean would be undetectable. This is the
    same arm `tests/test_file_size.py::test_main_returns_one_when_a_file_is_over_cap` covers.

    The fixture lock names installed packages at impossible versions, so every one is compared and
    every one drifts, and it holds enough of them to clear `COMPARISON_FLOOR`, so the verdict is
    drift rather than the structural refusal beneath it.
    """
    import tools.check_lock_drift as subject

    names = sorted({dist.metadata["Name"] for dist in _installed()} - {None})
    assert len(names) > subject.COMPARISON_FLOOR, (
        "the fixture cannot clear the comparison floor, so this would test the structural arm"
    )
    lock = tmp_path / "drifted.lock"
    lock.write_text(
        "".join(f'[[package]]\nname = "{name}"\nversion = "0.0.0.dev0"\n\n' for name in names),
        encoding="utf-8",
    )
    monkeypatch.setattr(subject, "LOCK_PATH", lock)

    assert subject.main() == 1
    captured = capsys.readouterr().err
    assert "have drifted from uv.lock" in captured
    assert subject.REMEDIATION in captured


_ONE = '[[package]]\nname = "a"\nversion = "1"\n'
"""A well-formed entry, so a malformed one after it sits at index 1."""


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, "could not be read (No such file or directory)"),
        (b'[[package]]\nname = "\xff\xfe"\n', "is not valid UTF-8 (invalid start byte"),
        (b"[[package]\n", "is not valid TOML ("),
        (b"version = 1\n", "holds no `package` key rather than a `[[package]]` array"),
        (b'package = "x"\n', "holds a `package` str rather than a `[[package]]` array"),
        (b"package = []\n", "holds an empty `[[package]]` array"),
        (b"package = [1]\n", "has a `[[package]]` entry at index 0 that is not a table"),
        (
            b'[[package]]\nversion = "1"\n',
            "has a `[[package]]` entry at index 0 with no string `name`",
        ),
        (
            (_ONE + '[[package]]\nname = "b"\nversion = 2\n').encode(),
            "has a `[[package]]` entry at index 1 (`b`) with no string `version`",
        ),
        (
            (
                b'[[package]]\nname = "PyTest"\nversion = "9.1.1"\n\n'
                b'[[package]]\nname = "pytest"\nversion = "8.0.0"\n'
            ),
            "pins `pytest` at two versions, 9.1.1 and 8.0.0 (index 1)",
        ),
    ],
    ids=[
        "absent",
        "not-utf8",
        "not-toml",
        "no-package-key",
        "package-not-array",
        "empty-array",
        "entry-not-table",
        "entry-without-name",
        "entry-without-version",
        "two-versions",
    ],
)
def test_a_lock_that_is_not_one_version_per_package_is_structural_and_says_why(
    tmp_path, monkeypatch, capsys, content, expected
):
    """Every way the lock fails to read is exit 2, and the message names which way.

    One generic message for all of them would send the operator to the wrong fix. Two cases carry
    the reasoning: invalid UTF-8 raises `UnicodeDecodeError`, neither `OSError` nor
    `TOMLDecodeError`, which unhandled would exit 1 and read as drift; and a forked lock, two
    `[[package]]` blocks for one name, would otherwise be compared against whichever came last,
    reporting drift that is not drift where `COMPARISON_FLOOR` cannot see it.

    `main` reads `LOCK_PATH` at call time, so repointing it is how `main` is driven here; a
    parameter default would still read the real lock, which is well formed, and exit 0.
    """
    import tools.check_lock_drift as subject

    lock = tmp_path / "uv.lock"
    if content is not None:
        lock.write_bytes(content)

    with pytest.raises(subject.LockReadError, match=re.escape(expected)):
        subject.locked_versions(lock)
    monkeypatch.setattr(subject, "LOCK_PATH", lock)
    assert subject.main() == 2
    captured = capsys.readouterr()
    assert f"{lock} {expected}" in captured.err
    assert captured.out == ""


def test_an_entry_the_lock_cannot_read_refuses_a_lock_that_would_otherwise_pass(
    tmp_path, monkeypatch, capsys
):
    """A malformed entry is not skipped: skipping it is a package whose drift goes unchecked.

    Every other entry names an installed package at its installed version, enough to clear
    `COMPARISON_FLOOR`, so a checker that skipped the malformed one would exit 0 with "agree". The
    control runs the well-formed entries alone and must pass.
    """
    import tools.check_lock_drift as subject

    names = sorted(
        {subject.normalize(d.metadata["Name"]) for d in _installed() if d.metadata["Name"]}
    )
    assert len(names) > subject.COMPARISON_FLOOR, "the fixture cannot clear the comparison floor"
    good = "".join(
        f'[[package]]\nname = "{name}"\nversion = "{version(name)}"\n\n' for name in names
    )
    lock = tmp_path / "uv.lock"
    monkeypatch.setattr(subject, "LOCK_PATH", lock)

    lock.write_text(good, encoding="utf-8")
    assert subject.main() == 0, "the control: the well-formed entries alone agree"
    capsys.readouterr()

    lock.write_text(good + '[[package]]\nname = "unversioned"\n', encoding="utf-8")
    assert subject.main() == 2
    assert "(`unversioned`) with no string `version`" in capsys.readouterr().err


def test_the_lock_drift_check_refuses_a_lock_too_small_to_mean_anything(
    tmp_path, monkeypatch, capsys
):
    """Below `COMPARISON_FLOOR` is structural, because a normalization bug exits clean otherwise.

    A one-package lock is readable and internally consistent, and comparing one package says
    nothing about the environment. The floor is what separates "in agreement" from "did not look".
    """
    import tools.check_lock_drift as subject

    lock = tmp_path / "tiny.lock"
    lock.write_text(
        '[[package]]\nname = "pytest"\nversion = "' + version("pytest") + '"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(subject, "LOCK_PATH", lock)

    assert subject.main() == 2
    assert "below the floor" in capsys.readouterr().err


def test_the_lock_drift_check_normalizes_the_spellings_that_differ(tmp_path):
    """`pyyaml` against `PyYAML` is the shape that makes a naive comparison vacuous.

    The floor exists because that bug exits clean, so this pins the normalization directly rather
    than trusting the floor to notice.
    """
    import tools.check_lock_drift as subject

    assert subject.normalize("PyYAML") == "pyyaml"
    assert subject.normalize("pytest_cov") == "pytest-cov"
    assert subject.normalize("zope.interface") == "zope-interface"
    assert subject.normalize("a__b") == "a-b"


def test_the_lock_drift_check_reports_drift_and_spares_an_absent_package():
    """Both arms of the comparison, driven rather than reasoned about.

    A locked package that is not installed must not count as drift: `make setup` installs the
    `dev` extra rather than the whole lock, so failing on absence would refuse the environment that
    target produces.
    """
    import tools.check_lock_drift as subject

    installed = version("pytest")
    drifted, compared, absent = subject.drift(
        {
            "pytest": "0.0.0-not-a-real-version",
            "definitely-not-installed-anywhere": "1.0",
        }
    )

    assert drifted == [("pytest", "0.0.0-not-a-real-version", installed)]
    assert (compared, absent) == (1, 1)


def test_the_lock_drift_success_line_attributes_the_absent_set_to_no_single_extra(
    tmp_path, monkeypatch, capsys
):
    """The success line may not name which extra is missing, because it cannot know.

    `drift` compares only installed packages, so a missing extra is absence rather than drift and
    never reaches this line's evidence. Naming one would be true on a tree that ran the current
    `make setup` and false on one that did not.

    **The prohibition is broader than that reason, deliberately.** It forbids naming *any* extra,
    including a truthful enumeration of what `make setup` does install, because that enumeration is
    a second copy of a set whose only home is the `Makefile` recipe. A future author whose accurate
    message fails here is failing on the duplication, not on the accuracy.

    **The names are read from `pyproject.toml`, not listed here.** A hand-copied list is another
    copy of the same set, and an extras change would join it unprohibited. Matched bare and
    case-insensitively on a word boundary, so dropping the code spans or capitalising does not
    evade it.
    """
    import tools.check_lock_drift as subject

    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = sorted(declared["project"]["optional-dependencies"])
    assert len(extras) >= 2, f"read {extras} from pyproject.toml; a near-empty set cannot prohibit"

    lock = tmp_path / "many.lock"
    lock.write_text(
        f'[[package]]\nname = "pytest"\nversion = "{version("pytest")}"\n'
        '[[package]]\nname = "definitely-not-installed-anywhere"\nversion = "1.0"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(subject, "LOCK_PATH", lock)
    monkeypatch.setattr(subject, "COMPARISON_FLOOR", 1)

    assert subject.main() == 0
    captured = capsys.readouterr()
    out = captured.out
    # The module docstring promises failure text on stderr and only the success line on stdout.
    # The sibling cases assert presence in one stream and not that the other is empty, so a
    # success path that also wrote to stderr would pass all of them.
    assert captured.err == "", f"success path wrote to stderr: {captured.err!r}"
    # The counts are the payload, so a message satisfying only the prohibition would be useless
    # and would pass.
    assert "1 of 2 locked packages" in out
    assert "1 locked but not installed" in out
    for extra in extras:
        assert not re.search(rf"\b{re.escape(extra)}\b", out, re.IGNORECASE), (
            f"the success line names the {extra!r} extra, but it compares installed packages only "
            f"and cannot tell which extra an absent package came from. Full line: {out!r}"
        )


def test_the_rebase_check_refuses_a_cached_trunk_ref_the_remote_has_moved_past(tmp_path, capsys):
    """The defect the freshness probe exists for: a cache that lags the remote.

    The branch contains everything its *cache* of the trunk holds, so the ancestry test passes, and
    a check that read only the cache would say "the measured tree is the tree that will land" while
    the remote had moved two commits past it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    bare = with_remote(repo, trunk)
    advanced = advance_remote(tmp_path, bare, "two", "three")

    assert rebased_verdict(repo) == 1
    failure = capsys.readouterr().err
    assert "stale" in failure
    assert trunk[:8] in failure, "the failure must name the cache it read"
    assert advanced[:8] in failure, "and the commit the remote actually has"
    assert "does not contain it" in failure, (
        "and must say the containment test ran, since a behind cache alone is not the defect"
    )
    assert "git fetch origin" in failure


def test_a_cached_trunk_ref_behind_a_remote_head_already_contains_is_not_a_refusal(
    tmp_path, capsys
):
    """A behind cache is not automatically a behind branch, and refusing on the mismatch would lie.

    When HEAD already contains what the remote advertises, nothing on the trunk is unmeasured.
    Refusing anyway would assert a stale tree, against this module's own rule that a failure must
    never be stated as a fact about the repository.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)
    (repo / "own.txt").write_text("own\n", encoding="utf-8")
    run_git(repo, "add", "own.txt")
    run_git(repo, "commit", "-m", "own")
    run_git(repo, "push", "origin", "main")
    # Force the cache back: the push updates it opportunistically, which is what makes this case
    # rare rather than unreachable.
    run_git(repo, "update-ref", "refs/remotes/origin/main", trunk)

    assert rebased_verdict(repo) == 0
    passed = capsys.readouterr().out
    assert "already contains that commit" in passed
    assert "stale cache" in passed, "the lag is still reported -- it is just not a refusal"


def test_the_rebase_check_reports_that_freshness_was_not_established_without_a_remote(
    tmp_path, capsys
):
    """Offline is supported, and the weaker claim has to read as weaker, on stderr.

    Three things make this outcome distinguishable from a confirmed pass: a different sentence, the
    absence of any "matches the remote" phrasing, and the stream. On stdout it would be hidden by
    `make check >/dev/null`, which is how triage on stderr works.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert "NOT established" in captured.err
    assert "matches `origin`" not in captured.err
    assert captured.out == "", "a run the operator must read something about writes no stdout"
    assert "git fetch origin" in captured.err, "and it names the next action"
    assert "no `origin` remote is configured" in captured.err, (
        "the reason is a fixed string this check chose"
    )
    assert "fatal:" not in captured.err, (
        "and never git's own stderr, which arrives at whatever length and with whatever control "
        "characters the remote chose"
    )
    assert "yourself" not in captured.err, (
        "the operator is not told to judge the age; the check establishes freshness or says it "
        "did not"
    )


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda: subprocess.TimeoutExpired(cmd=["git"], timeout=1.0), "did not answer"),
        (lambda: OSError("git is not on PATH"), "could not be run to ask the remote"),
    ],
    ids=["timeout", "unexecutable"],
)
def test_a_remote_that_cannot_be_reached_degrades_rather_than_failing(
    tmp_path, monkeypatch, capsys, build, expected
):
    """The offline guarantee, which is the reason the probe degrades instead of gating.

    A plane, a proxy, a locked SSH key: none of them is evidence about the tree, and `make check`
    is the only gate there is, so refusing here would make the gate unrunnable where it has nothing
    to say. Both raising paths are driven because they return different reasons and the message
    names the reason.

    The exceptions are built per-case rather than shared instances, so a re-run cannot mutate a
    previous run's `__traceback__` on the object the parametrization holds.
    """
    import tools._remote as probe

    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)

    real = probe.run

    def refuse(args, **kwargs):
        if args and args[0] == "ls-remote":
            raise build()
        return real(args, **kwargs)

    monkeypatch.setattr(probe, "run", refuse)

    assert rebased_verdict(repo) == 0
    failure = capsys.readouterr().err
    assert "NOT established" in failure
    assert expected in failure, "the message must name why freshness was not established"


def test_a_remote_advertising_no_trunk_branch_degrades_with_that_reason(
    tmp_path, monkeypatch, capsys
):
    """An empty or trunk-less advertisement must not be read as agreement.

    The probe returns a commit or a reason and never a commit it did not read, because the caller
    treats agreement as freshness. A remote that answers without naming the trunk is the case where
    a `None` return and a matching return would otherwise be easy to conflate.
    """
    import tools._remote as probe

    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)

    real = probe.run

    def silent(args, **kwargs):
        if args and args[0] == "ls-remote":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return real(args, **kwargs)

    monkeypatch.setattr(probe, "run", silent)

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert "advertises no" in captured.err
    assert "matches `origin`" not in captured.err


@pytest.mark.parametrize(
    ("advertisement", "why"),
    [
        ("", "an empty advertisement"),
        ("\t" + "refs/heads/main" + "\n", "a line with no commit before the tab"),
        ("0" * 40 + "\trefs/tags/main\n", "a tag rather than a branch"),
        ("0" * 40 + "\trefs/heads/mainline\n", "a branch whose name merely starts the same"),
        ("0" * 40 + "\trefs/heads/a/refs/heads/main\n", "a decoy branch, sorted first"),
    ],
    ids=["empty", "no-commit", "tag", "prefix", "decoy-only"],
)
def test_an_advertisement_without_the_trunk_is_never_read_as_the_trunk(
    tmp_path, monkeypatch, capsys, advertisement, why
):
    """Every shape that must not satisfy the exact ref-name match.

    Dropping `and commit.strip()` would return an empty commit and refuse as stale, printing a
    blank SHA; relaxing `==` to `in` or `endswith` would read the decoy's commit; and dropping
    `--heads` would let the tag answer. One case per shape keeps the parse's discriminating power
    from resting on a single degenerate input.
    """
    import tools._remote as probe

    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)

    real = probe.run

    def advertise(args, **kwargs):
        if args and args[0] == "ls-remote":
            return subprocess.CompletedProcess(args, 0, stdout=advertisement, stderr="")
        return real(args, **kwargs)

    monkeypatch.setattr(probe, "run", advertise)

    assert rebased_verdict(repo) == 0, f"{why} must degrade, not refuse"
    captured = capsys.readouterr()
    assert "advertises no" in captured.err, f"{why} must be reported as no trunk advertised"
    assert "0000000" not in captured.err + captured.out, (
        f"{why} must never have its commit reported as the trunk's"
    )


def test_a_decoy_branch_whose_name_ends_in_the_trunks_is_not_read_as_the_trunk(
    tmp_path, monkeypatch, capsys
):
    """The exact ref-name match, which is the only thing rejecting a decoy.

    `ls-remote`'s pattern is a glob tail-matched on slash boundaries, so `refs/heads/main` also
    advertises a *branch* named `a/refs/heads/main`, which `--heads` cannot exclude, because it is
    a branch, and which sorts **first**. Relaxing the comparison to `endswith` or `in` therefore
    reads the decoy's commit and returns the wrong trunk.

    Driven through a stub advertisement rather than a real decoy branch so the decoy's ordering is
    pinned rather than left to git's sort.
    """
    import tools._remote as probe

    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)
    decoy = "0" * 40

    real = probe.run

    def advertise(args, **kwargs):
        if args and args[0] == "ls-remote":
            lines = f"{decoy}\trefs/heads/a/refs/heads/main\n{trunk}\trefs/heads/main\n"
            return subprocess.CompletedProcess(args, 0, stdout=lines, stderr="")
        return real(args, **kwargs)

    monkeypatch.setattr(probe, "run", advertise)

    assert rebased_verdict(repo) == 0, "the real trunk matches the cache, so this is a pass"
    passed = capsys.readouterr().out
    assert "matches `origin` as of now" in passed
    assert decoy[:8] not in passed, "the decoy's commit must never be reported as the trunk's"


def test_the_rebase_check_says_the_trunk_matches_the_remote_when_it_does(tmp_path, capsys):
    """The positive arm, without which the refusal above could be an unconditional refusal.

    ADR `tooling/file-size-cap`'s anti-vacuity argument is the one that applies: a check that
    refused whenever a remote existed would be indistinguishable from one that measures it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)
    # HEAD ahead of a matching trunk is the everyday state of a working branch. With a single
    # commit `head == trunk` by construction and the freshness comparison would be degenerate: a
    # `remote != head` mutation would block the gate for anyone with unpushed work.
    for name in ("branch-one", "branch-two"):
        (repo / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
        run_git(repo, "add", f"{name}.txt")
        run_git(repo, "commit", "-m", name)
    head = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    assert head != trunk, "the fixture must actually put HEAD ahead of the trunk"

    assert rebased_verdict(repo) == 0
    passed = capsys.readouterr().out
    assert "matches `origin` as of now" in passed
    assert "NOT established" not in passed
    assert trunk[:8] in passed, "the trunk end is named"
    assert head[:8] in passed, "and so is HEAD, so a comparison against the wrong one is visible"


def test_a_tag_named_like_the_trunk_ref_is_excluded_by_the_argv(tmp_path, capsys):
    """`--heads` on the argv, driven against a real advertisement rather than a stub.

    `ls-remote` tail-matches its pattern on slash boundaries, so `refs/heads/main` also matches a
    *tag* literally named `refs/heads/main`. The stubbed cases above cannot see this, because a
    stub never runs the argv: dropping `--heads` would survive them all. This one builds the tag on
    the remote, so the flag is what makes the advertisement single-valued.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    bare = with_remote(repo, trunk)
    run_git(repo, "tag", "refs/heads/main", trunk)
    run_git(repo, "push", "origin", "refs/tags/refs/heads/main")
    advertised = run_git(repo, "ls-remote", str(bare), "refs/heads/main").stdout
    assert advertised.count("\n") == 2, (
        f"the fixture must advertise both refs for the flag to matter, got {advertised!r}"
    )

    assert rebased_verdict(repo) == 0
    passed = capsys.readouterr().out
    assert "matches `origin` as of now" in passed


def test_a_configured_but_unreachable_remote_degrades_with_a_fixed_reason(tmp_path, capsys):
    """The real offline shape: a remote is configured and `ls-remote` fails against it.

    Distinct from having no remote at all, which the config guard catches before `ls-remote` runs
    -- so without this case the non-zero-returncode arm is unreachable. It is also the case that
    checks the reason carries none of git's raw stderr onto a passing line.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    run_git(repo, "remote", "add", "origin", str(tmp_path / "no-such-remote.git"))
    assert trunk

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert "the remote could not be consulted" in captured.err
    assert "fatal:" not in captured.err, "git's own stderr must not reach this message"
    assert captured.out == ""


def test_a_git_that_cannot_answer_about_the_remote_is_not_reported_as_no_remote(
    tmp_path, monkeypatch, capsys
):
    """A broken git must not become a claim about the repository's configuration.

    `COULD_NOT_RUN` on the `remote.origin.url` lookup means git never ran, which is not the same
    fact as "no remote is configured", and stating the second is the defect that sentinel exists
    to prevent. The call-site sweep cannot see the difference because both spellings end at the
    same exit code; only the message differs.
    """
    import tools._remote as probe

    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)
    real = probe.run

    def broken(args, **kwargs):
        if args == ["config", "--get", "remote.origin.url"]:
            raise OSError("git could not be executed")
        return real(args, **kwargs)

    monkeypatch.setattr(probe, "run", broken)

    assert rebased_verdict(repo) == 0
    failure = capsys.readouterr().err
    assert "git could not be consulted about the remote (git did not run)" in failure
    assert "no `origin` remote is configured" not in failure, (
        "a git that never ran must not be reported as a repository with no remote"
    )
