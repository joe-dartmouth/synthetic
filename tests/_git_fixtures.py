"""Fixture repositories for the `make rebased` checker's tests.

Shared by `tests/test_repo_checkers.py` and `tests/test_rebased_second_remote.py`, which drive the
same checker against different remote layouts. Every repository is built offline: remotes are bare
repositories reached by path, which `ls-remote` speaks the same protocol to as a network remote.
"""

from __future__ import annotations

import subprocess

from tools._git import child_environment


def init_repo(path):
    """Create a git repository with one commit on `main` and a matching `origin/main`.

    The remote-tracking ref is written directly rather than by cloning, so a fixture needs no
    remote to exercise the ancestry test. A repository built by this helper has no remote at all,
    which drives the checker's degrade path; `with_remote` adds an on-disk remote for the cases
    that need one.

    Args:
        path: An empty directory to initialize.

    Returns:
        The commit the trunk points at.
    """
    run_git(path, "init", "--initial-branch", "main")
    run_git(path, "config", "user.email", "gate@example.invalid")
    run_git(path, "config", "user.name", "Gate")
    # A contributor with `commit.gpgsign = true` globally would otherwise get errored tests from a
    # working checker. `run_git` also blanks the global and system config for the same reason --
    # local `config` cannot override every global setting that breaks a commit.
    run_git(path, "config", "commit.gpgsign", "false")
    (path / "seed.txt").write_text("one\n", encoding="utf-8")
    run_git(path, "add", "seed.txt")
    run_git(path, "commit", "-m", "one")
    head = run_git(path, "rev-parse", "HEAD").stdout.strip()
    run_git(path, "update-ref", "refs/remotes/origin/main", head)
    return head


def run_git(path, *args):
    """Run git in `path`, refusing on failure so a broken fixture is not read as a verdict.

    Args:
        path: The working directory.
        *args: Arguments after `git`.

    Returns:
        The completed process.
    """
    try:
        return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            ["git", *args],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
            # The whole environment comes from `tools._git`, one home with the checkers it
            # exercises: the `GIT_*` strip, and the two config keys blanked on top of it. A
            # contributor's global config reaches a read as well as a write, because
            # `core.fsmonitor` names a program git then runs. The timeout above is this fixture's
            # own and is longer: building a repository costs more than measuring one. `check=True`
            # with `capture_output=True` reports only "returned non-zero exit status N", so the
            # `stderr` is re-raised below rather than lost.
            env=child_environment(),
        )
    except subprocess.CalledProcessError as exc:
        raise AssertionError(f"fixture git {args} failed: {exc.stderr}") from exc


def rebased_verdict(repo_root):
    """Run `check_rebased.main` against a fixture repository.

    `main` takes the root as a parameter, so no module attribute is patched, and these tests
    exercise the same entry point the Makefile does rather than a patched variant of it.

    Args:
        repo_root: The repository the check should read.

    Returns:
        The exit code.
    """
    import tools.check_rebased as subject

    return subject.main(repo_root)


def with_remote(path, trunk):
    """Give `path` a real `origin` on disk, pushed to `trunk`, and cache the ref at it.

    A bare repository reached by path rather than a network remote: `ls-remote` speaks the same
    protocol to both, so the freshness probe is exercised for real while the suite stays offline.

    Args:
        path: A repository built by `init_repo`.
        trunk: The commit its `origin/main` cache points at.

    Returns:
        The bare repository's path.
    """
    bare = path.parent / "remote.git"
    # `--initial-branch` for the same reason `init_repo` passes it: `run_git` blanks the global
    # and system config, so `init.defaultBranch` is not visible here and the bare repository would
    # take git's compiled-in `master`. `main` is then pushed into it below, which leaves HEAD
    # dangling at a ref that does not exist -- so `advance_remote`'s clone lands on an unborn
    # `master` and its `push origin main` fails with "src refspec main does not match any".
    run_git(path.parent, "init", "--bare", "--initial-branch", "main", str(bare))
    run_git(path, "remote", "add", "origin", str(bare))
    run_git(path, "push", "origin", f"{trunk}:refs/heads/main")
    return bare


def advance_remote(tmp_path, bare, *messages):
    """Land `messages` on the remote's `main` from a second clone, and return its new tip.

    A second clone rather than a push from the fixture under test, so the repository being checked
    keeps its stale cache, which is the condition under test.

    Args:
        tmp_path: The test's temporary directory.
        bare: The remote built by `with_remote`.
        *messages: One commit per message.

    Returns:
        The commit `main` now points at on the remote.
    """
    other = tmp_path / f"clone-of-{bare.name}"
    if other.exists():
        run_git(other, "pull", "--ff-only")
    else:
        run_git(tmp_path, "clone", str(bare), str(other))
    run_git(other, "config", "user.email", "gate@example.invalid")
    run_git(other, "config", "user.name", "Gate")
    run_git(other, "config", "commit.gpgsign", "false")
    for message in messages:
        (other / f"{message}.txt").write_text(f"{message}\n", encoding="utf-8")
        run_git(other, "add", f"{message}.txt")
        run_git(other, "commit", "-m", message)
    run_git(other, "push", "origin", "main")
    return run_git(other, "rev-parse", "HEAD").stdout.strip()
