"""One `git` runner for the repository's own checkers, and the guard it must be paired with.

The runner spells the same invocation for every checker: no shell, a pinned `cwd`, a
`GIT_*`-stripped child environment, a bounded timeout. The two halves of the defence live here
together, and :func:`run` names the guard in its own docstring: a checker that reads paths or refs
out of git and does not first establish which repository answered is measuring an unknown tree,
because git discovers a repository by walking upward and a verdict about an enclosing tree would be
printed as a verdict about this one. A checker inherits both halves or neither.

The fixture runner in `tests/_git_fixtures.py` takes :func:`child_environment` and keeps its own
timeout, which is longer because building a repository costs more than measuring one.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from collections.abc import Sequence as AbcSequence
from pathlib import Path
from typing import Final


GIT_TIMEOUT_SECONDS: Final = 10
"""Bound on one `git` call, because a hung git would otherwise block `make check` unbounded."""


GROUP_GRACE_SECONDS: Final = 1.0
"""How long git's group has to exit on `SIGTERM` before :func:`_end_group` sends `SIGKILL`.

Long enough for git to remove a lock file, short enough that a Ctrl-C still feels immediate.
"""


def child_environment() -> dict[str, str]:
    """A child environment for `git` that answers about this tree and skips the user's config.

    **`cwd` alone is not a trust boundary.** `GIT_DIR`, `GIT_WORK_TREE` and `GIT_INDEX_FILE` are
    read from the environment and take precedence over the working directory. With
    `GIT_INDEX_FILE` pointed at a copy of the index with two files removed, `check_file_size.py`
    would scan two fewer files and print its success line: a gate that exits 0 while silently not
    measuring part of the tree, which is worse than one that fails.

    **The global and system config files are blanked too, and that is a code-execution defence
    rather than a determinism one.** Stripping `GIT_*` does not stop git reading `~/.gitconfig`,
    and a config file can name a program git will run: a `.gitconfig` that sets `core.fsmonitor`
    makes git execute that script inside a checker's own `git ls-files` call. It is not a privilege
    crossing, since whoever can set `HOME` for `make check` can also edit the checker, but a gate
    that runs a contributor's hook script is not measuring what it claims to, and closing it costs
    two keys. `tests/test_git_runner.py` reproduces it with `GIT_CONFIG_GLOBAL` unset against a
    fixture `HOME`.

    **What the blanking excludes, and what still reaches git.** Excluded: the global files
    (`~/.gitconfig` and `$XDG_CONFIG_HOME/git/config`, both replaced by `GIT_CONFIG_GLOBAL`), the
    system file under git's own prefix (replaced by `GIT_CONFIG_SYSTEM`), and command-line config
    passed down in `GIT_CONFIG_PARAMETERS` or `GIT_CONFIG_COUNT` (removed by the `GIT_*` strip).
    Still read: the repository's own `.git/config`, its `config.worktree` and whatever they
    include, which belong to the tree being measured; and on Apple's git (Xcode or the Command Line
    Tools) a root-owned file in the developer directory, such as
    `/Applications/Xcode.app/Contents/Developer/usr/share/git-core/gitconfig`, which sets
    `credential.helper=osxkeychain` and which neither key replaces. A contributor cannot edit that
    file without root, so it is outside the defence above; it does mean "no foreign config" holds
    for the files a contributor controls, not for every file git reads.

    Exposed separately from :func:`run` so a caller that reads bytes can build its own
    `subprocess.run` on the same environment.

    Returns:
        A fresh mapping, safe for the caller to add to.
    """
    scrubbed = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    # Prompting off. `check_rebased.py`'s freshness probe runs `ls-remote`, and git's default is to
    # prompt for credentials on `/dev/tty` rather than on stdin, so `stdin=DEVNULL` in :func:`run`
    # does not stop it, and a gate that blocks on an invisible password prompt is worse than one
    # that fails. Set here rather than at the call site because the `GIT_*` strip above removes an
    # operator's own setting, so this restores a value the scrub took away. It does not bound
    # `ssh`, which reads a passphrase from the terminal directly; the session :func:`run` starts
    # git in covers that, since it has no terminal for `ssh` to open.
    scrubbed["GIT_TERMINAL_PROMPT"] = "0"
    # Askpass off too, which the terminal switch above does not reach. git still runs an askpass
    # program (`GIT_ASKPASS`, then `core.askPass`, then `SSH_ASKPASS`) with no terminal, and on a
    # desktop that is a credential dialog popping up during `make check`. An empty `GIT_ASKPASS`
    # ends git's search without naming a program, and `SSH_ASKPASS_REQUIRE=never` stops `ssh` using
    # its own askpass for a key passphrase.
    scrubbed["GIT_ASKPASS"] = ""
    scrubbed["SSH_ASKPASS_REQUIRE"] = "never"
    # No lazy fetch. In a partial clone, asking for an object the checkout lacks fetches it from
    # the promisor remote, so `cat-file -e` on a commit another remote advertised would write
    # objects and report a commit the operator never fetched as held. The gate's steps are
    # read-only, and an absent object must read as absent.
    scrubbed["GIT_NO_LAZY_FETCH"] = "1"
    # `os.devnull` rather than a missing path: git treats an unreadable config as an error on some
    # subcommands, while an empty one is unambiguously "no settings".
    scrubbed["GIT_CONFIG_GLOBAL"] = os.devnull
    scrubbed["GIT_CONFIG_SYSTEM"] = os.devnull
    return scrubbed


def run(
    args: AbcSequence[str],
    *,
    root: Path,
    check: bool,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run one `git` subcommand under `root`.

    **Pair this with :func:`require_toplevel`, or the answer is about an unknown repository.** git
    discovers a repository by walking upward from `cwd`, so running here inside a checkout nested in
    another one answers about whichever it finds first, and `git ls-files` reports paths relative to
    *that* toplevel. Establishing which repository answered is the other half of the defence the
    stripped environment starts.

    `stdin` is closed. A git that decides to prompt, for a credential or a pager it thinks is
    interactive, would otherwise block until `timeout`, and a checker run from `make` has no
    terminal to prompt at.

    **git runs in a session of its own, and a timeout or an interrupt kills its whole process
    group.** `subprocess.run` kills git and not what git forked, so an `ssh` transport helper could
    outlive the freshness probe's timeout holding `/dev/tty` and keep prompting after the gate had
    degraded and moved on. A new session closes both halves: it has no controlling terminal, so a
    helper cannot open `/dev/tty` to prompt at all, and the group kill takes the helper down with
    git. **The cost is that git no longer receives the operator's Ctrl-C**, since the terminal
    signals only its foreground group; the interrupt reaches this process instead, which ends the
    group in its place: `SIGTERM`, then `SIGKILL`, not the `SIGINT` the terminal would have sent.
    :func:`_end_group` states what it cannot reach. `tests/test_git_runner.py` drives both exits.

    Args:
        args: Arguments after `git`. A read-only sequence, so a caller holding a tuple need
            not build a list to pass it. **Put a `--` before any pathspec or operand that is not a
            spelled-out constant.** `check_rebased.py`'s operands are 40-hex object names, which
            cannot begin with `-`, but this is the surface the next checker will use. Nothing here
            adds `--no-optional-locks`: it is irrelevant to `ls-files`, `rev-parse` and
            `merge-base`, and relevant the first time someone calls `run(["status", ...])`.
        root: The directory to run in.
        check: Raise `CalledProcessError` on a non-zero exit. `False` when the caller reads the
            status itself, which `check_rebased.py` does: a non-zero `rev-parse` is one of its
            expected answers rather than a failure.
        timeout: Seconds before git is killed.

    Returns:
        The completed process, with `stdout` and `stderr` decoded as UTF-8 with
        `surrogateescape`. A byte that is not valid UTF-8, in a path or anywhere else git prints,
        becomes a lone surrogate rather than a `UnicodeDecodeError`: a traceback exits 1, which
        the checkers' contract reads as a violation. `os.fsencode` and `Path` turn such a string
        back into the original bytes, so a path decoded here still names its file.

    Raises:
        subprocess.CalledProcessError: If git exits non-zero and `check` is true.
        subprocess.TimeoutExpired: If git hangs past `timeout`, after its group is killed.
        OSError: If git cannot be executed at all.
    """
    argv = ["git", *args]
    # No shell, and no call site reaches user or model input. Not all of it is literals:
    # `check_rebased.py` passes commit ids it read back from a prior `git rev-parse`, so the values
    # are git-produced object names, which is why the `--` note in `Args` matters for the next
    # caller. The `subprocess-without-shell-equals-true` pragma is needed because of the
    # unpacking: with a fully literal argv ruff does not flag the call.
    with subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="surrogateescape",
        cwd=root,
        env=child_environment(),
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            # A timeout, and a Ctrl-C too: the new session keeps the terminal's `SIGINT` away from
            # git, so the interrupt lands here and the whole group is ended in its place.
            _end_group(process)
            raise
    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, argv, stdout, stderr)
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _signal_group(leader: int, signum: signal.Signals) -> bool:
    """Send `signum` to the group `leader` leads, and say whether any member could receive it.

    **Two errors mean "nothing left to signal", and the second is the platform one.** `ESRCH`
    (`ProcessLookupError`) is a group that has gone. `EPERM` (`PermissionError`) is what macOS
    answers for a group whose only member is an unreaped zombie, where Linux signals such a group
    without error. That is git's state when a Ctrl-C or a deadline lands as it exits and before it
    is reaped, and suppressing `ESRCH` alone would let a `PermissionError` replace the caller's
    `KeyboardInterrupt`. Every member is ours and runs as this user, so `EPERM` here cannot mean a
    member we lack the right to signal.

    Args:
        leader: The pid of the process started with `start_new_session=True`, which is its
            process group id as well.
        signum: The signal to send.

    Returns:
        ``True`` when the signal was delivered to at least one member.
    """
    try:
        os.killpg(leader, signum)
    except ProcessLookupError, PermissionError:
        return False
    return True


def _end_group(process: subprocess.Popen[str]) -> None:
    """End every process in git's group: `SIGTERM`, a short grace, then `SIGKILL`.

    **`SIGTERM` first so git can clean up.** git removes a lock file it holds when it is
    terminated, and `SIGKILL` skips that, so an interrupted command that writes the index would
    leave `.git/index.lock` behind. The grace is bounded by :data:`GROUP_GRACE_SECONDS`, and
    `SIGKILL` follows whatever happened in it, for a helper that ignores `SIGTERM`.

    **What this cannot reach**: it runs only when an exception reaches :func:`run`. Python installs
    no handler for `SIGHUP` or `SIGTERM`, so a checker killed by a supervisor, a terminal hangup or
    `SIGKILL` exits without running it, and since git sits in a session of its own, those signals
    no longer reach git through a shared group either. A git whose remote accepts and then stalls
    is orphaned that way, with its helper. A second Ctrl-C landing inside this function also cuts
    it short.

    Args:
        process: git, started with `start_new_session=True`.
    """
    if not _signal_group(process.pid, signal.SIGTERM):
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=GROUP_GRACE_SECONDS)
    _signal_group(process.pid, signal.SIGKILL)


def toplevel_of(root: Path) -> Path | None:
    """The toplevel of the repository that answers about `root`, or ``None`` if none does.

    The *question* both checkers ask, split from the two different ways they answer it:
    :func:`require_toplevel` raises, and `check_rebased.py` prints its own exit codes because it
    needs "git absent", "not a checkout" and "wrong toplevel" apart. Both compare the answer with
    :func:`same_directory`.

    Args:
        root: The directory to ask from.

    Returns:
        The resolved toplevel, or ``None`` when `root` is not inside a checkout.

    Raises:
        subprocess.TimeoutExpired: If git hangs.
        OSError: If git cannot be executed.
    """
    finished = run(["rev-parse", "--show-toplevel"], root=root, check=False)
    if finished.returncode != 0:
        return None
    return Path(finished.stdout.strip()).resolve()


def same_directory(first: Path, second: Path) -> bool:
    """Whether two paths name one directory, by device and inode rather than by spelling.

    **The comparison :func:`toplevel_of`'s callers make, and why it is not a string compare.** git
    reports its toplevel in the on-disk case while `Path.resolve()` keeps the case it was handed, so
    on case-insensitive macOS a checkout addressed as `/users/...` would be refused as not being
    its own toplevel.

    Args:
        first: One path.
        second: The other.

    Returns:
        ``True`` when both exist and are the same file; ``False`` otherwise, including when either
        cannot be read.
    """
    try:
        return first.samefile(second)
    except OSError:
        return False


def require_toplevel(root: Path) -> None:
    """Refuse unless `root` is the answering repository's own toplevel.

    `git ls-files` reports relative to the **git toplevel**, not to `cwd`, so `root / name` is only
    a real path when the two coincide. Clone this repository inside another one and they do not: the
    listing comes back `inner/src/...`, every entry matches no rule, and the gate fails with a
    message blaming the rule table for something it did not do. Checking the assumption directly
    costs one `git` call and makes the diagnostic name the actual cause.

    **Raises rather than returning a code**, which suits a checker whose refusals are exceptions.
    `check_rebased.py` needs the three outcomes apart (git absent, `root` not a checkout, and a
    toplevel that differs), so it asks with :func:`run` and prints its own exit codes instead. That
    is why this is a sibling of the runner rather than folded into it.

    Args:
        root: The directory the caller intends to measure.

    Raises:
        RuntimeError: If the toplevel differs from `root`, or if `root` is not inside a checkout
            at all. Both are the same refusal to this caller, "git will not answer about the
            directory I was told to measure", which is why they share a type and differ only in
            message.
        subprocess.TimeoutExpired: If git hangs.
        OSError: If git cannot be executed.
    """
    toplevel = toplevel_of(root)
    if toplevel is None:
        raise RuntimeError(
            f"{root} is not inside a git checkout, so there is no tracked set to measure and no "
            "toplevel to compare against."
        )
    if not same_directory(toplevel, root):
        raise RuntimeError(
            f"{root} is not this git repository's toplevel, which is {toplevel}. Paths would come "
            "back relative to the toplevel and resolve against the wrong directory."
        )
