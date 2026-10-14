"""Refuse a gate run whose tree is not the tree that will land.

Two pull requests can each pass `make check` and still merge into a red trunk: one renames
something, the other branched before the rename landed, so its own gate never saw it. Nothing
measures the post-merge tree, so this check at least refuses a tree that is behind the trunk.

**This gates the merge base rather than the merge result, and the difference is the whole scope of
what it can promise.** When the trunk is an ancestor of `HEAD` there is nothing on it the branch has
not seen, so the tree `make check` measures is the tree that will land, for a fast-forward or a
squash, which is how this repository merges. When it is not an ancestor, the trunk carries commits
the gate is blind to. `make check` runs this twice, first and again as `rebased-final`, so a merge
that lands while the tests run is refused at the end when that last probe reaches the remote; a
merge after the gate and before the squash is the part only CI could see.

**A stale trunk ref must not pass.** The ancestry test reads the local remote-tracking ref, a cache
of whatever the last `git fetch` saw, so on its own it proves only that the branch is current with
respect to that cache. So the cache's freshness is measured too, with `ls-remote` rather than a
fetch: a fetch *writes*, and every other step of `make check` is read-only. When the cache and the
remote disagree and HEAD does not already contain what the remote has, this refuses and names the
fetch.

**Offline still works, so the probe degrades rather than gating on itself**, and an unreachable
remote must never read as agreement: the two passing outcomes are different sentences on different
streams, not one sentence with a footnote. **In a fork `origin` is not the canonical trunk**, so a
configured `upstream` is probed too and refuses unless HEAD contains its trunk; the success line
names each remote it measured rather than "the remote".

ADR `tooling/refuse-a-stale-trunk-ref` owns the reasoning. **Deliberately not restated here**: an
argument with several prose homes drifts.

**`git rev-parse HEAD^{tree}` does not substitute for this.** Keying a skipped gate run on a tree
that has already passed removes duplicated work and still never measures the merge result, because
the tree it keys on is the branch's own.

Three properties keep the check from passing while claiming something false, which is the failure
mode a gate has instead of a bug:

1. `_git` distinguishes a timeout or an unexecutable git from git's own exit status 1. The trunk
   lookup reads status 1 as "the ref does not exist", and that branch returns 0, so collapsing the
   two would let a hung git produce a clean gate and a printed claim about refs that was never
   measured. The result is three-state: :data:`COULD_NOT_RUN` is distinguishable from any status
   git itself returns, and every call site maps it to structural.
2. A toplevel guard comes first, and `cwd` is not one: git walks *upward*, so a copy of this tree
   that is not itself a checkout but sits inside one would get a confident verdict about the outer
   repository. The scrubbed environment and the guard live together in `tools/_git.py`, so a
   checker inherits the pair rather than half of it.
3. `origin/main` is an ambiguous refname, because `refs/heads/<name>` resolves ahead of
   `refs/remotes/<name>`: a local branch literally called `origin/main` would shadow the trunk and
   pin the gate green. git's only signal is a warning on stderr, which this module discards by
   design. The lookup is fully qualified.

**It abstains mid-rebase and mid-bisect**, which is the same principle. A conflicted rebase leaves
`HEAD` at the trunk with conflict markers in the working tree, so the ancestry test passes and the
strongest sentence in this file would be printed about a tree that is neither the branch's nor the
result's. A bisect is worse than useless: the ancestry test fails, and the fix the failure prints,
`git rebase`, would end the bisect and destroy the investigation. Running the suite is exactly what
a bisect is for, so those states take the does-not-apply path.

Exit codes, on the contract `tools/check_file_size.py` shares:

- `0`: the branch is current, its cached trunk merely lags behind a commit HEAD already contains,
  or the check does not apply. A pass whose freshness could not be established is `0` too.
- `1`: the branch is behind; its cached trunk ref is stale and HEAD does not contain what the remote
  advertises; or a configured `upstream` advertises a trunk HEAD does not contain.
- `2`: structural. Git could not be consulted, answered about a different repository, or reported
  a missing or corrupt object where an answer was needed, at any call site, `upstream`'s included,
  because stating containment from such a status is a fact git did not establish.

A stale ref is `1` rather than `2` because it is the same defect one step earlier: fetch and the
branch *is* behind, so the fix is the same command and the tree is equally unfit to gate.

**When the trunk cache is absent, `upstream` is still probed.** The check does not apply to
`origin` then, but a configured canonical remote can still refuse, so a clone made with
`git clone -o upstream`, or one whose fork remote has another name, is not waved through while the
canonical trunk is ahead.

**The stream split is by what the operator must read, not by pass and fail.** `stdout` carries only
an outcome that needs nothing from them: a confirmed pass, or a stated does-not-apply. Everything
else goes to `stderr`, including the *passing* run whose freshness could not be established, which
is neither a failure nor a confirmed pass and which `make check >/dev/null` would otherwise hide.
**Exit `0` therefore still conflates a confirmed pass with a degraded one.** A third code would
separate them for a consumer that branches on it, and there is none without CI; `rebased-final`
makes the degraded notice the gate's last line instead.
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path
from typing import Final

from tools._git import run, same_directory, toplevel_of
from tools._rebased_report import (
    CanonicalReport,
    report_current,
    report_lagging,
    report_no_trunk,
)
from tools._remote import (
    CANONICAL_REMOTES,
    REMOTE,
    TRUNK_LABEL,
    UNCONSULTED,
    _remote_trunk,
    advertised_trunk,
    configured,
)


REPO_ROOT: Final = Path(__file__).resolve().parent.parent


COULD_NOT_RUN: Final = -1
"""The status :func:`_git` reports when git did not run at all.

Negative because every status git itself can return is non-negative, so no call site can confuse
"could not be consulted" with a verdict. `1` would not do: git uses it for both "ref absent" and
"not an ancestor", and the ref-absent branch passes.
"""

TRUNK: Final = f"refs/remotes/{REMOTE}/main"
"""The remote-tracking ref a branch must have seen all of.

**Fully qualified, not `origin/main`.** Per gitrevisions, `refs/heads/<name>` resolves before
`refs/remotes/<name>`, so the short form lets a local branch named `origin/main` shadow the trunk
and satisfy the gate. `main` is the default branch and the base for pull requests, so this is the
ref rather than a configurable one; a fork whose trunk is named otherwise has no such ref and takes
the does-not-apply path.
"""


REBASE_STATE_DIRS: Final = ("rebase-merge", "rebase-apply")
"""Marker directories git writes for an interactive and a patch-based rebase respectively."""

BISECT_STATE_FILE: Final = "BISECT_LOG"
"""Marker git writes when a bisect is running."""


def _git(*args: str, root: Path) -> tuple[int, str]:
    """Run a git command in `root` and return its status and stripped stdout.

    Wraps `tools._git.run` with `check=False`, because a non-zero `rev-parse` is one of this
    checker's expected answers rather than a failure, and collapses "git did not run at all" into
    :data:`COULD_NOT_RUN` so a caller can tell that apart from a verdict git actually returned.
    :func:`_preflight` does not use this for the toplevel question, which `tools._git.toplevel_of`
    owns, but the ancestry test and the trunk lookup both do, and each of those has a branch that
    treats a status of 1 as a pass.

    Args:
        *args: The git arguments, without the leading `git`.
        root: The directory to run in.

    Returns:
        The exit status and the stripped stdout, or :data:`COULD_NOT_RUN` and an empty string when
        git did not run. Stderr is discarded: every caller decides from the status, and git's own
        message would read as this checker's output.
    """
    try:
        finished = run([*args], root=root, check=False)
    except subprocess.TimeoutExpired, OSError:
        return COULD_NOT_RUN, ""
    return finished.returncode, finished.stdout.strip()


def _mid_operation(git_dir: Path) -> str | None:
    """Return the name of a git operation in progress, or None.

    Args:
        git_dir: The repository's `.git` directory, as git reports it.

    Returns:
        `"rebase"`, `"bisect"`, or None.
    """
    if any((git_dir / name).exists() for name in REBASE_STATE_DIRS):
        return "rebase"
    if (git_dir / BISECT_STATE_FILE).exists():
        return "bisect"
    return None


def _preflight(root: Path) -> int | None:
    """Establish that git answers about `root`, and that no operation is mid-flight.

    Split from :func:`main` because the two answer different questions, "can this check speak
    about this tree at all" against "is this tree current", and because keeping them together would
    put too many returns in one function for `ruff` and for a reader.

    Args:
        root: The repository to read.

    Returns:
        An exit code to use directly, or None when the ancestry test should proceed.
    """
    # `toplevel_of` rather than `require_toplevel`, and rather than a comparison spelled here: this
    # checker needs "git could not run", "not a checkout" and "wrong tree" apart, and that helper
    # collapses the last two into one raise. What it must not do is re-resolve the reported path
    # itself: that comparison is `tools._git`'s. It is a file comparison, `same_directory`,
    # because a string one would refuse a case-changed spelling.
    try:
        toplevel = toplevel_of(root)
    except subprocess.TimeoutExpired, OSError:
        print(
            "git could not be run at all, so the gate cannot tell what tree it is measuring. That "
            "is a failure rather than a pass. Check that git is on PATH and responding.",
            file=sys.stderr,
        )
        return 2
    if toplevel is None:
        print(
            f"{root} is not a git checkout, so there is no tree to compare. The gate cannot tell "
            "what it is measuring, which is a failure rather than a pass.",
            file=sys.stderr,
        )
        return 2
    if not same_directory(toplevel, root):
        print(
            f"git answers about {toplevel} rather than {root}, because it discovers a repository "
            "by walking upward and this directory is not one. Any verdict would describe the "
            "wrong tree, so this is structural rather than a pass.",
            file=sys.stderr,
        )
        return 2

    status, git_dir = _git("rev-parse", "--absolute-git-dir", root=root)
    operation = _mid_operation(Path(git_dir)) if status == 0 else None
    if operation is not None:
        print(
            f"a {operation} is in progress, so this check does not apply. Mid-{operation} the "
            "working tree is neither the branch's nor the result's, and during a bisect the fix "
            "this check would print -- a rebase -- would end the bisect."
        )
        return 0
    return None


def _resolve_ends(root: Path) -> tuple[int | None, str, str]:
    """Resolve HEAD and the trunk, reporting whichever cannot be resolved.

    Args:
        root: The repository to read.

    Returns:
        An exit code to use directly or None to proceed, then HEAD's commit and the cached trunk's,
        the latter empty when no trunk ref is cached.
    """
    status, head = _git("rev-parse", "HEAD", root=root)
    if status == COULD_NOT_RUN:
        print(
            "git could not be consulted about HEAD, so the gate does not know which commit it is "
            "measuring.",
            file=sys.stderr,
        )
        return 2, "", ""
    if status != 0:
        print(
            "HEAD does not resolve to a commit, so there is nothing to compare. That is an "
            "unborn branch; git could not be consulted is reported separately above, because a "
            "failure must never be stated as a fact about the repository.",
            file=sys.stderr,
        )
        return 2, "", ""

    status, trunk = _git("rev-parse", "--verify", f"{TRUNK}^{{commit}}", root=root)
    if status == COULD_NOT_RUN:
        print(
            f"git could not be consulted about `{TRUNK_LABEL}`, so whether this tree is current "
            "was never established. Reporting that as 'no such ref' would pass, so it is "
            "structural.",
            file=sys.stderr,
        )
        return 2, "", ""
    # No trunk ref is not yet "does not apply": a configured `upstream` is still probed, so the
    # caller decides once it has asked. An empty trunk is how this says so.
    return None, head, trunk if status == 0 else ""


def _probe_canonical(root: Path, head: str) -> CanonicalReport:
    """Ask each configured canonical remote for its trunk and whether HEAD contains it.

    A remote that is not configured is skipped without a word, so a checkout with no `upstream`
    prints what it printed before this probe existed. One that is configured and cannot answer
    degrades the pass, exactly as an unreachable `origin` does.

    Args:
        root: The repository to read.
        head: HEAD's commit.

    Returns:
        What the remotes said; the first refusal stops the probe.
    """
    contained: list[tuple[str, str]] = []
    degraded: list[tuple[str, str]] = []
    for remote in CANONICAL_REMOTES:
        known = configured(root, remote)
        if known is False:
            continue
        commit, why_not = advertised_trunk(root, remote) if known else (None, UNCONSULTED)
        if commit is None:
            degraded.append((remote, why_not))
            continue
        # **Present before contained.** A commit this checkout never fetched cannot be contained,
        # and `merge-base` reports a missing object with the status it uses for corruption, which
        # is structural below; asking `cat-file` first is what lets an unfetched trunk name the
        # fetch instead. `cat-file -e` cannot itself tell absent from unreadable (both are 128),
        # so the refusal says "no readable copy" rather than claiming which.
        present, _ = _git("cat-file", "-e", f"{commit}^{{commit}}", root=root)
        if present == COULD_NOT_RUN:
            return CanonicalReport.unanswered(remote, commit)
        if present != 0:
            return CanonicalReport.behind(remote, commit, head, fetched=False)
        ancestry, _ = _git("merge-base", "--is-ancestor", commit, head, root=root)
        if ancestry == 0:
            contained.append((remote, commit))
        elif ancestry == 1:
            return CanonicalReport.behind(remote, commit, head, fetched=True)
        else:
            return CanonicalReport.unanswered(remote, commit)
    return CanonicalReport(contained=tuple(contained), degraded=tuple(degraded))


def _freshness(root: Path, head: str, trunk: str) -> tuple[int | None, str, str]:
    """Judge the cached trunk against the remote's, refusing a stale one HEAD lacks.

    Split from :func:`main` for the reason :func:`_preflight` was: `ruff` caps the returns in one
    function. The split is along the same seam, "can this tree be judged" against "is it current",
    with freshness a third question ahead of both. It runs before the canonical probe, so a refusal
    here costs no second network round trip.

    Args:
        root: The repository to read.
        head: HEAD's commit.
        trunk: The commit the local cache of the trunk points at.

    Returns:
        An exit code to use directly or None to proceed; the reason the remote could not be
        consulted, **empty exactly when it was**, which is what the caller keys its two success
        lines on; and the remote's commit when the cache merely lags behind a commit HEAD contains,
        else empty. That last pass is reported by the caller, once the canonical remotes have
        answered, because a canonical refusal must not be pre-empted by it.
    """
    remote, why_not = _remote_trunk(root)
    if remote is None or remote == trunk:
        return None, why_not, ""
    # **A behind cache is not automatically a behind branch, and refusing on the mismatch alone
    # would state a falsehood.** If HEAD already contains what the remote advertises then nothing
    # on the trunk is unmeasured and the cache is merely lagging, which is reachable after a
    # fast-forward, a supported merge mode. So only an absent commit or git's own "not an
    # ancestor" refuses as behind (HEAD cannot contain a commit this checkout does not hold),
    # while a status git gives for a corrupt object, or none at all, is structural, as it is at
    # every other call site.
    present, _ = _git("cat-file", "-e", f"{remote}^{{commit}}", root=root)
    if present == 0:
        ancestry, _ = _git("merge-base", "--is-ancestor", remote, head, root=root)
    else:
        ancestry = COULD_NOT_RUN if present == COULD_NOT_RUN else 1
    if ancestry == 0:
        return None, why_not, remote
    if ancestry != 1:
        print(
            f"`{TRUNK_LABEL}` is stale -- this checkout caches {trunk[:8]}, the remote has "
            f"{remote[:8]} -- and git could not say whether HEAD contains the remote's commit: it "
            "did not run, or it reported a missing or corrupt object. Not a pass, and not a "
            "rebase either.\n"
            "Run `git fsck` to check the object store, then run the gate again.",
            file=sys.stderr,
        )
        return 2, why_not, ""
    print(
        f"`{TRUNK_LABEL}` is stale: this checkout caches {trunk[:8]}, the remote has "
        f"{remote[:8]}, and HEAD does not contain it -- so the trunk carries commits this "
        "tree has never been measured against and the ancestry test below would compare "
        "against a superseded trunk. That combination is the pass this check exists to "
        "refuse.\n"
        f"Run `git fetch {REMOTE} && git rebase {TRUNK_LABEL}`, then run the gate again.",
        file=sys.stderr,
    )
    return 1, why_not, ""


def _judge_ancestry(
    root: Path, head: str, trunk: str, why_not: str, canonical: CanonicalReport
) -> int:
    """Run the ancestry test against the cached trunk and report it.

    Args:
        root: The repository to read.
        head: HEAD's commit.
        trunk: The cached trunk's commit.
        why_not: Why `origin` could not be consulted, empty when it was.
        canonical: What the canonical remotes said.

    Returns:
        The exit code.
    """
    status, _ = _git("merge-base", "--is-ancestor", trunk, head, root=root)
    if status == COULD_NOT_RUN:
        print(
            "git could not be consulted about the merge base, so whether this tree is current was "
            "never established.",
            file=sys.stderr,
        )
        return 2
    if status == 0:
        return report_current(head, trunk, why_not, canonical)
    if status != 1:
        print(
            f"git could not decide whether `{TRUNK_LABEL}` is an ancestor of HEAD (status "
            f"{status}), which means a missing or corrupt object rather than a branch that is "
            "behind. Not a pass, and not a rebase either.",
            file=sys.stderr,
        )
        return 2

    print(
        f"`{TRUNK_LABEL}` ({trunk[:8]}) is not an ancestor of HEAD ({head[:8]}), so the trunk "
        "carries commits this tree has never been measured against. Two individually-green "
        "branches merging into a red trunk is the failure this refuses.\n"
        f"Run `git fetch origin main && git rebase {TRUNK_LABEL}`, then run the gate again.",
        file=sys.stderr,
    )
    return 1


def main(root: Path = REPO_ROOT) -> int:
    """Refuse when the trunk carries commits `HEAD` has not seen.

    Args:
        root: The repository to read. Threaded rather than read from the module global so the tests
            can drive it without monkeypatching, against fixture repositories.

    Returns:
        The exit code, on the contract the module docstring's exit-code paragraph states in full;
        it is not repeated here, because a contract stated twice drifts.
    """
    early = _preflight(root)
    if early is not None:
        return early

    early, head, trunk = _resolve_ends(root)
    if early is not None:
        return early

    # **Freshness before ancestry, and the order matters.** An ancestry verdict against a
    # superseded cache is a false pass, so a stale ref must refuse here rather than reach the
    # branch that prints the strongest sentence in this file. The canonical probe runs after it
    # and before any pass is printed, so `origin` speaks first and a refusal from it costs no
    # second round trip.
    early, why_not, lagging = _freshness(root, head, trunk) if trunk else (None, "", "")
    if early is not None:
        return early
    canonical = _probe_canonical(root, head)
    if not trunk:
        return report_no_trunk(canonical)
    if lagging:
        return report_lagging(trunk, lagging, canonical)
    return _judge_ancestry(root, head, trunk, why_not, canonical)


if __name__ == "__main__":
    sys.exit(main())
