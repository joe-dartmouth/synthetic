"""Ask a remote where its trunk is, without fetching and without moving a local ref.

The seam with `tools/check_rebased.py` is between asking and deciding: this module asks the
*question*, where the remote says its trunk is, and the checker decides what the answer means. It
prints nothing, exits nothing and holds no verdict.

It runs its own git calls through `tools._git.run` rather than through `check_rebased`'s
`_git` wrapper, which is what makes the split acyclic. That also means it does not use that
module's `COULD_NOT_RUN` sentinel: every way the trunk lookup fails is a *reason string*, because
a freshness probe that cannot answer is a supported outcome rather than a structural one, and the
caller degrades on it. :func:`configured` answers None with a reason, for the reason its docstring
gives, and the checker decides per remote whether that degrades or refuses.

ADR `tooling/refuse-a-stale-trunk-ref` owns why this exists.
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import Final

from tools._git import GIT_TIMEOUT_SECONDS, run


REMOTE: Final = "origin"
"""The remote this repository's trunk lives on.

Named because the checker derives its trunk ref and every operator-facing mention of it from this
one value, and :func:`_remote_trunk` asks it unless told to ask another.
"""

TRUNK_LABEL: Final = f"{REMOTE}/main"
"""How the trunk is named in operator-facing text, where the qualified form only adds noise."""

UNCONSULTED: Final = "git could not be consulted about the remote"
"""The reason :func:`_remote_trunk` gives when git could not say whether a remote is configured.

:func:`configured`'s own reason follows it in parentheses, naming whether git did not run or which
status it exited with.
"""

CANONICAL_REMOTES: Final = ("upstream",)
"""Remotes that, when configured, hold the canonical trunk of a fork.

In a fork, :data:`REMOTE` is the fork, so a checkout current with it can still lack commits the
canonical trunk carries. `upstream` is the name a fork conventionally gives the repository it was
forked from; it is a convention rather than a guarantee, so a canonical remote under another name is
not probed and the checker's success line names only the remotes it measured. **Configured** means
`remote.<name>.url` is set and non-empty (:func:`configured`), so a remote defined only by a legacy
`.git/remotes/<name>` file, a format git is retiring, is not probed either. Probing every
configured remote was ruled out: a contributor's own mirror is a remote and not a trunk (ADR
`tooling/refuse-a-stale-trunk-ref`).
"""

TRUNK_BRANCH: Final = "refs/heads/main"
"""The trunk as the *remote* names it, which is what `ls-remote` advertises.

Distinct from :data:`TRUNK`, the local cache of it.

**The pattern is a glob, not a ref name, and two separate things narrow it.** `ls-remote`
tail-matches its patterns on slash boundaries, so `refs/heads/main` also advertises anything whose
path *ends* that way. `--heads` on the argv removes one class: a tag literally named
`refs/heads/main`.

**It cannot remove the other, and the exact-equality test in :func:`_remote_trunk` is what does.**
A *branch* named `a/refs/heads/main` is advertised under `--heads` too, because it is a branch, and
it sorts **before** `refs/heads/main`, so a match relaxed to `endswith` or `in` would read the
decoy's commit and return the wrong trunk. The equality test is therefore load-bearing rather than
belt-and-braces.

**`--heads` is redundant given that equality test, and no test can distinguish it.** Dropping the
flag leaves the suite green even against a fixture that advertises a same-named tag, because the
equality test rejects that line anyway. It is kept for narrowing what the parse has to consider
rather than because the parse needs the help. `--heads` rather than `--branches`: the latter is the
spelling git 2.47 prefers, and the former is still accepted and silent on 2.50, so it keeps the
check working on the older git a contributor may have.
"""


NO_TRUNK: Final = f"the remote advertises no `{TRUNK_BRANCH}`"
"""The reason given when a reachable remote has no trunk under that name.

Named so a caller can tell it apart from an unreachable remote: fetching fixes the second and can
never fix this one, so the two need different next actions.
"""


def configured(root: Path, remote: str) -> tuple[bool | None, str]:
    """Whether `remote` is a configured remote of `root`, or None and why git could not say.

    **A configured remote, not a path.** `git ls-remote <name>` falls back to resolving the name as
    a *directory* when no such remote is configured, so a bare repository committed at `./origin`
    would answer for the trunk. That is the same shadowing class as a local branch named
    `origin/main`, which `check_rebased.TRUNK` closes by qualifying the ref.

    **None rather than False unless git answered "unset".** `git config --get` exits 1 for a key
    that is not set and 3 for a config file it cannot parse, among other statuses; only 1 is an
    answer. False off any other status, or off a call that never ran, would state a fact about the
    repository that git did not establish, which is the defect `check_rebased.COULD_NOT_RUN`
    exists to prevent, and for `upstream` it would skip the canonical probe without a word.

    Args:
        root: The repository to read.
        remote: The remote's name.

    Returns:
        True when `remote.<name>.url` is set to a non-empty value, False when it is unset or empty
        (git would refuse an empty URL anyway), each with an empty reason; or None with a reason
        naming why git could not say: it did not run, or the status it exited with.
    """
    try:
        found = run(["config", "--get", f"remote.{remote}.url"], root=root, check=False)
    except subprocess.TimeoutExpired, OSError:
        return None, "git did not run"
    if found.returncode == 1:
        return False, ""
    if found.returncode != 0:
        return None, f"`git config --get remote.{remote}.url` exited with status {found.returncode}"
    return bool(found.stdout.strip()), ""


def _remote_trunk(root: Path, remote: str = REMOTE) -> tuple[str | None, str]:
    """The trunk's commit on `remote`, or None and why it could not be read.

    Reads the ref advertisement only: no objects are transferred and no local ref moves, so this
    answers "is the cached ref still current" without the write a fetch would perform.

    A remote that cannot be reached is not a failure of this check, because running offline is
    supported, so every unreachable case returns a reason for the caller to report rather than an
    exit code. What it must not do is return a commit it did not read, because the caller treats
    agreement as freshness. Note the polarity is the opposite of :func:`_resolve_ends`', where
    `None` means "proceed": here `None` is the problem.

    **The bound is `tools._git`'s shared one rather than a shorter network-specific constant.** The
    bound is not paid on a healthy run, which costs the actual round trip whatever the bound is, so
    it is a per-*failure* cost. What it trades is worst-case latency on a broken link against
    misclassifying a working-but-slow one as unreachable, and a network call bounded *tighter* than
    a local one would have the trade backwards: a contributor on a high-latency link would be
    denied the check, at the full bound, every run.

    Args:
        root: The repository to read.
        remote: The remote to ask; :data:`REMOTE` unless the caller is probing a canonical one.

    Returns:
        The commit `refs/heads/main` points at on the remote and an empty reason, or None and a
        reason naming why the probe could not answer.
    """
    # The configured-remote guard is :func:`configured`'s, and its two failure cases take different
    # reasons for the reason that function's docstring gives.
    known, why = configured(root, remote)
    if known is None:
        return None, f"{UNCONSULTED} ({why})"
    if not known:
        return None, f"no `{remote}` remote is configured"
    return advertised_trunk(root, remote)


def advertised_trunk(root: Path, remote: str) -> tuple[str | None, str]:
    """Read `remote`'s advertised trunk, for a caller that has already established it is configured.

    Split from :func:`_remote_trunk` so a caller that must tell an unconfigured remote apart from an
    unreachable one asks :func:`configured` once rather than twice.

    Args:
        root: The repository to read.
        remote: A configured remote's name. Calling this with a name :func:`configured` has not
            confirmed reopens the directory-shadowing hole that function closes.

    Returns:
        The advertised commit and an empty reason, or None and why it could not be read.
    """
    try:
        # `--heads` restricts the advertisement to branches. Without it the pattern also matches a
        # tag of the same literal name, because `ls-remote` tail-matches on slash boundaries;
        # see :data:`TRUNK_BRANCH`.
        finished = run(["ls-remote", "--heads", remote, TRUNK_BRANCH], root=root, check=False)
    except subprocess.TimeoutExpired:
        return None, f"the remote did not answer within {GIT_TIMEOUT_SECONDS:g}s"
    except OSError:
        return None, "git could not be run to ask the remote"
    if finished.returncode != 0:
        # **A fixed reason, not git's stderr.** git's own message would read as this checker's
        # output, and the reason reaches a *passing* line: a failed probe would splice `fatal:`
        # mid-sentence inside a step that exited 0, at whatever length the remote chose. The
        # operator's next action does not depend on which of git's transport errors it was;
        # running the fetch by hand shows them the message in full.
        return None, "the remote could not be consulted"
    # **Two assumptions worth stating.** For `origin` the caller compares
    # `refs/heads/main` on the remote against `refs/remotes/origin/main` locally, which is the
    # *default* refspec; `child_environment` does not blank a repository's own config, so a
    # customized `remote.origin.fetch` makes the two refs unrelated and the check then refuses every
    # run. And a reachable remote that advertises no trunk at all is a definite answer routed to the
    # could-not-establish wording below -- conservative, and deliberately not its own branch,
    # because the operator's next action is the same either way.
    for line in finished.stdout.splitlines():
        advertised, _, ref = line.partition("\t")
        commit = advertised.strip()
        # **A null object id is not a commit.** git normalizes any advertised value it cannot
        # parse as hex to all zeros, so a remote that sends garbage reads as `0000...`; treating
        # that as a commit would make the caller assert the trunk carries commits that may not
        # exist.
        if ref.strip() != TRUNK_BRANCH or not commit:
            continue
        if not commit.strip("0"):
            return None, "the remote advertised a null object id for its trunk"
        return commit, ""
    return None, NO_TRUNK
