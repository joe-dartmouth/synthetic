"""`make rebased` against a configured `upstream`, the canonical trunk of a fork.

In a fork, `origin` is the fork, so a checkout can match `origin` exactly while the canonical trunk
carries commits it has never measured, which the check must not print as a confirmed pass. These
cases build that layout from bare repositories reached by path
(`tests/_git_fixtures.py`), so the probe runs for real and the suite stays offline (ADR
`tooling/refuse-a-stale-trunk-ref`).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import tools.check_rebased as subject
from tests._git_fixtures import advance_remote, init_repo, rebased_verdict, run_git, with_remote
from tools._rebased_report import CanonicalReport
from tools._remote import NO_TRUNK, UNCONSULTED, advertised_trunk


def _commit(repo, name):
    """Commit one file on HEAD and return the new commit."""
    (repo / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
    run_git(repo, "add", f"{name}.txt")
    run_git(repo, "commit", "-m", name)
    return run_git(repo, "rev-parse", "HEAD").stdout.strip()


def _bare(parent, name, repo, trunk, branch="main"):
    """A bare repository at `parent/name`, configured on `repo` as remote `name`, at `trunk`."""
    parent.mkdir(exist_ok=True)
    bare = parent / f"{name}.git"
    run_git(parent, "init", "--bare", "--initial-branch", branch, str(bare))
    run_git(repo, "remote", "add", name, str(bare))
    run_git(repo, "push", name, f"{trunk}:refs/heads/{branch}")
    return bare


def _advance(tmp_path, bare, message):
    """Land one commit on `bare`'s `main` from a clone outside the checkout, and return it."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return advance_remote(work, bare, message)


def _fork(tmp_path):
    """A checkout current with its `origin`, and an `upstream` at the same trunk.

    Returns:
        The checkout, the `upstream` bare repository, and the shared trunk commit.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)
    upstream = _bare(tmp_path / "canonical", "upstream", repo, trunk)
    return repo, upstream, trunk


def _lagging(repo, trunk):
    """Push a commit to `origin` and force the cache back, so it lags behind a commit HEAD holds."""
    own = _commit(repo, "own")
    run_git(repo, "push", "origin", "main")
    run_git(repo, "update-ref", "refs/remotes/origin/main", trunk)
    return own


def _break(repo, remote):
    """Point `remote` at a path that does not exist, so the probe cannot reach it."""
    run_git(repo, "remote", "set-url", remote, str(repo.parent / f"gone-{remote}.git"))


# --- refusals ---------------------------------------------------------------------------------


def test_a_canonical_remote_ahead_of_head_refuses(tmp_path, capsys):
    """The fork is current with `origin` and the canonical trunk has moved: refuse.

    The canonical commits were never fetched, so HEAD cannot contain them. A check that probed only
    `origin` would return 0 and print that the measured tree is the tree that will land.
    """
    repo, upstream, _ = _fork(tmp_path)
    moved = _advance(tmp_path, upstream, "canonical-one")

    assert rebased_verdict(repo) == 1
    captured = capsys.readouterr()
    assert captured.out == "", "a refusal writes nothing the operator can skim past as a pass"
    assert moved[:8] in captured.err, "the refusal names the canonical commit it read"
    assert "holds no readable copy of that commit" in captured.err
    assert "Run `git fetch upstream && git rebase upstream/main`" in captured.err


def test_a_fetched_canonical_commit_head_does_not_contain_refuses(tmp_path, capsys):
    """Fetched but not rebased onto: the object is present and HEAD still lacks it."""
    repo, upstream, _ = _fork(tmp_path)
    moved = _advance(tmp_path, upstream, "canonical-one")
    run_git(repo, "fetch", "upstream")

    assert rebased_verdict(repo) == 1
    failure = capsys.readouterr().err
    assert moved[:8] in failure
    assert "does not contain it" in failure
    assert "git rebase upstream/main" in failure


def test_a_checkout_behind_both_is_told_about_origin_first(tmp_path, capsys):
    """Behind a fresh `origin` cache and behind `upstream`: `origin` speaks, `upstream` waits.

    The canonical refusal comes after the ancestry test, so `origin` being behind is mentioned
    before `upstream` is.
    """
    repo, upstream, _ = _fork(tmp_path)
    origin_bare = tmp_path / "remote.git"
    _advance(tmp_path, origin_bare, "origin-one")
    run_git(repo, "fetch", "origin")
    _advance(tmp_path, upstream, "canonical-one")

    assert rebased_verdict(repo) == 1
    failure = capsys.readouterr().err
    assert "is not an ancestor of HEAD" in failure
    assert "`upstream`" not in failure


def test_a_stale_origin_cache_head_contains_still_checks_upstream(tmp_path, capsys):
    """The lagging-cache pass must not bypass the canonical probe."""
    repo, upstream, trunk = _fork(tmp_path)
    _lagging(repo, trunk)
    moved = _advance(tmp_path, upstream, "canonical-one")

    assert rebased_verdict(repo) == 1
    assert moved[:8] in capsys.readouterr().err


def test_an_unreachable_origin_does_not_hide_a_canonical_refusal(tmp_path, capsys):
    """`origin` degraded and `upstream` ahead: the refusal still stands."""
    repo, upstream, _ = _fork(tmp_path)
    _break(repo, "origin")
    moved = _advance(tmp_path, upstream, "canonical-one")

    assert rebased_verdict(repo) == 1
    assert moved[:8] in capsys.readouterr().err


def test_a_missing_origin_cache_still_probes_upstream(tmp_path, capsys):
    """With no `origin/main` the check does not apply to `origin`, but `upstream` can still refuse.

    A `git clone -o upstream` checkout has exactly this shape, and "does not apply" on stdout would
    hide a canonical trunk that is ahead.
    """
    repo, upstream, _ = _fork(tmp_path)
    run_git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    moved = _advance(tmp_path, upstream, "canonical-one")

    assert rebased_verdict(repo) == 1
    assert moved[:8] in capsys.readouterr().err


@pytest.mark.parametrize("site", ["canonical", "origin"])
def test_a_corrupt_object_status_is_structural_rather_than_behind(
    tmp_path, monkeypatch, capsys, site
):
    """`merge-base` exiting 128 is a missing or corrupt object, never "HEAD does not contain it".

    Both containment tests are driven: `upstream`'s, and `origin`'s on a lagging cache. Neither may
    state containment as a fact on that status, as the ancestry test in `main` already says
    structural.
    """
    repo, _, trunk = _fork(tmp_path)
    asked = _lagging(repo, trunk) if site == "origin" else trunk
    real = subject._git

    def corrupt(*args, root):
        if args[:2] == ("merge-base", "--is-ancestor") and args[2] == asked:
            return 128, ""
        return real(*args, root=root)

    monkeypatch.setattr(subject, "_git", corrupt)
    assert subject.main(repo) == 2
    failure = capsys.readouterr().err
    assert "missing or corrupt object" in failure
    assert "does not contain it" not in failure


# --- passes -----------------------------------------------------------------------------------


def test_a_contained_canonical_trunk_is_named_in_the_success_line(tmp_path, capsys):
    """HEAD contains `upstream`'s trunk, and the stdout line says both were measured."""
    repo, _, trunk = _fork(tmp_path)
    head = _commit(repo, "own")

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        f"`origin/main` ({trunk[:8]}) is an ancestor of HEAD ({head[:8]}) and matches `origin` "
        "as of now, so the measured tree is the tree that will land there. HEAD also contains the "
        f"trunk of `upstream` ({trunk[:8]}), so the canonical trunk carries nothing this tree has "
        "not seen.\n"
    )


def test_no_canonical_remote_leaves_the_success_line_byte_identical(tmp_path, capsys):
    """Without `upstream` the success line names `origin` alone, byte for byte."""
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        f"`origin/main` ({trunk[:8]}) is an ancestor of HEAD ({trunk[:8]}) and matches `origin` "
        "as of now, so the measured tree is the tree that will land there.\n"
    )


def test_a_contained_canonical_trunk_rides_the_lagging_cache_line(tmp_path, capsys):
    """The lagging-cache pass carries the canonical clause too, on stdout."""
    repo, _, trunk = _fork(tmp_path)
    own = _lagging(repo, trunk)

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert f"the remote has {own[:8]}" in captured.out
    assert captured.out.endswith(
        f"HEAD also contains the trunk of `upstream` ({trunk[:8]}), so the canonical trunk "
        "carries nothing this tree has not seen.\n"
    )


def test_a_contained_canonical_trunk_rides_the_no_cache_line(tmp_path, capsys):
    """No `origin/main`, `upstream` contained: does-not-apply, naming what was measured."""
    repo, _, trunk = _fork(tmp_path)
    run_git(repo, "update-ref", "-d", "refs/remotes/origin/main")

    assert rebased_verdict(repo) == 0
    passed = capsys.readouterr().out
    assert "this check does not apply" in passed
    assert f"`upstream` ({trunk[:8]})" in passed


def test_a_directory_named_upstream_that_is_no_remote_is_ignored(tmp_path, capsys):
    """`git ls-remote upstream` resolves a directory of that name; the configured guard stops it.

    The directory holds a bare repository whose `main` HEAD does not contain, so reading it as the
    canonical trunk would refuse. It is not a configured remote, so it must not be read at all.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)
    decoy = repo / "upstream"
    run_git(repo, "init", "--bare", "--initial-branch", "main", str(decoy))
    _advance(tmp_path, decoy, "decoy-one")

    assert rebased_verdict(repo) == 0
    passed = capsys.readouterr().out
    assert "`upstream`" not in passed
    assert "matches `origin` as of now" in passed


def test_an_empty_upstream_url_reads_as_unconfigured(tmp_path, capsys):
    """`remote.upstream.url` set to nothing is no remote, so the output is the no-`upstream` one."""
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)
    run_git(repo, "config", "remote.upstream.url", "")

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "`upstream`" not in captured.out


# --- degraded passes ---------------------------------------------------------------------------


def test_an_unreachable_canonical_remote_degrades(tmp_path, capsys):
    """A configured `upstream` that cannot answer is a degraded pass, never a confirmed one."""
    repo, _, _ = _fork(tmp_path)
    _break(repo, "upstream")

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert captured.out == "", "a pass the operator must read something about writes no stdout"
    assert "matches `origin` as of now" in captured.err, "origin's confirmed half is still stated"
    assert "NOT established: `upstream`: the remote could not be consulted." in captured.err
    assert "Run `git fetch upstream` once the remote answers" in captured.err


def test_a_canonical_remote_with_no_main_is_told_so_rather_than_to_fetch(tmp_path, capsys):
    """Fetching can never fix a remote whose trunk has another name, so it is not the advice."""
    repo = tmp_path / "repo"
    repo.mkdir()
    trunk = init_repo(repo)
    with_remote(repo, trunk)
    _bare(tmp_path / "canonical", "upstream", repo, trunk, branch="master")

    assert rebased_verdict(repo) == 0
    failure = capsys.readouterr().err
    assert f"`upstream`: {NO_TRUNK}" in failure
    assert "`upstream` has no `main`" in failure
    assert "git fetch upstream" not in failure


def test_a_lagging_cache_with_an_unreachable_canonical_remote_degrades(tmp_path, capsys):
    """The lagging-cache pass moves to stderr when `upstream` could not answer."""
    repo, _, trunk = _fork(tmp_path)
    _lagging(repo, trunk)
    _break(repo, "upstream")

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "is a stale cache" in captured.err
    assert "NOT established: `upstream`" in captured.err


@pytest.mark.parametrize("canonical", ["contained", "unreachable"])
def test_an_unreachable_origin_carries_what_upstream_said(tmp_path, capsys, canonical):
    """`origin` degraded: the canonical clause or paragraph follows its own notice."""
    repo, _, trunk = _fork(tmp_path)
    _break(repo, "origin")
    if canonical == "unreachable":
        _break(repo, "upstream")

    assert rebased_verdict(repo) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Whether that is still the remote's trunk was NOT established" in captured.err
    if canonical == "contained":
        assert f"HEAD also contains the trunk of `upstream` ({trunk[:8]})" in captured.err
    else:
        assert "Whether HEAD contains the canonical trunk was NOT established" in captured.err


def test_a_git_that_cannot_say_whether_upstream_is_configured_degrades(
    tmp_path, monkeypatch, capsys
):
    """`configured` answering None is "could not be consulted", not "not configured"."""
    repo, _, _ = _fork(tmp_path)
    monkeypatch.setattr(subject, "configured", lambda root, remote: None)

    assert rebased_verdict(repo) == 0
    assert f"`upstream`: {UNCONSULTED}" in capsys.readouterr().err


# --- several canonical remotes ------------------------------------------------------------------


def test_every_canonical_remote_is_probed_and_named(tmp_path, monkeypatch, capsys):
    """Two canonical remotes, both contained: both are named, and neither stops the other."""
    repo, _, trunk = _fork(tmp_path)
    _bare(tmp_path / "canonical", "mirror", repo, trunk)
    monkeypatch.setattr(subject, "CANONICAL_REMOTES", ("upstream", "mirror"))

    assert rebased_verdict(repo) == 0
    assert f"`upstream` ({trunk[:8]}), `mirror` ({trunk[:8]})" in capsys.readouterr().out


def test_the_unestablished_paragraph_groups_each_remote_by_its_next_action():
    """Unreachable remotes share one fetch line; a remote with no `main` gets its own sentence."""
    report = CanonicalReport(
        degraded=(("one", "the remote could not be consulted"), ("two", "x"), ("three", NO_TRUNK))
    )
    paragraph = report.unestablished()
    assert "`one`: the remote could not be consulted; `two`: x; `three`: " in paragraph
    assert "Run `git fetch one two` once the remote answers" in paragraph
    assert "`three` has no `main`" in paragraph
    assert report.stream is sys.stderr
    assert CanonicalReport().stream is sys.stdout


def test_a_null_object_id_is_not_read_as_a_trunk(tmp_path, monkeypatch):
    """git normalizes an unparseable advertised value to zeros; that is no commit."""
    import tools._remote as probe

    def advertise(args, *, root, check):
        del root, check
        return subprocess.CompletedProcess(args, 0, stdout="0" * 40 + "\trefs/heads/main\n")

    monkeypatch.setattr(probe, "run", advertise)
    commit, why = advertised_trunk(tmp_path, "upstream")
    assert commit is None
    assert "null object id" in why


# --- every call site ----------------------------------------------------------------------------


def _call_lines() -> set[int]:
    """Every line in `tools/check_rebased.py` that calls `_git(`, read off the syntax tree."""
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    return {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_git"
    }


def _layouts(tmp_path):
    """Checkouts that between them reach every `_git` call site."""
    bare = tmp_path / "bare"
    bare.mkdir()
    plain = bare / "repo"
    plain.mkdir()
    init_repo(plain)

    contained = tmp_path / "contained"
    contained.mkdir()
    contained_repo, _, _ = _fork(contained)

    lagging = tmp_path / "lagging"
    lagging.mkdir()
    lagging_repo, _, trunk = _fork(lagging)
    _lagging(lagging_repo, trunk)
    return [plain, contained_repo, lagging_repo]


def test_every_git_call_site_is_driven_and_honours_the_sentinel(tmp_path, monkeypatch, capsys):
    """Each `_git` call site, reached in some layout, reports structural when git cannot run.

    The sites are read off the syntax tree and the reached ones off the caller's line, so a new
    site fails here until a layout reaches it.
    """
    real = subject._git
    reached: dict[Path, list[int]] = {}
    for repo in _layouts(tmp_path):
        lines: list[int] = []

        def recording(*args, root, _lines=lines):
            _lines.append(sys._getframe(1).f_lineno)
            return real(*args, root=root)

        monkeypatch.setattr(subject, "_git", recording)
        subject.main(repo)
        reached[repo] = lines
    capsys.readouterr()

    undriven = _call_lines() - {line for lines in reached.values() for line in lines}
    assert not undriven, f"no layout reaches the `_git` call(s) on line(s) {sorted(undriven)}"

    for repo, lines in reached.items():
        for fail_at in range(len(lines)):
            calls = {"n": 0}

            def failing(*args, root, _limit=fail_at, _seen=calls):
                _seen["n"] += 1
                if _seen["n"] > _limit:
                    return subject.COULD_NOT_RUN, ""
                return real(*args, root=root)

            monkeypatch.setattr(subject, "_git", failing)
            assert subject.main(repo) == 2, (
                f"a git failure at call {fail_at + 1} (line {lines[fail_at]}) did not report as "
                "structural"
            )
            assert "could not" in capsys.readouterr().err
