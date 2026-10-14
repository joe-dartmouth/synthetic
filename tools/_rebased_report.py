"""What `make rebased` found, and the lines it prints when the branch passes.

The seam with `tools/check_rebased.py` is between asking and saying: the checker runs every git
call, so its `COULD_NOT_RUN` sentinel and the call-site sweep in
`tests/test_rebased_second_remote.py` reach them all, and this module holds what the canonical
remotes said and prints the outcome of a run that got that far: a canonical refusal, or one of the
three passing lines. It runs no git and exits nothing.

ADR `tooling/refuse-a-stale-trunk-ref` owns why a configured `upstream` is probed at all.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import NamedTuple, TextIO

from tools._remote import NO_TRUNK, REMOTE, TRUNK_LABEL


class Refusal(NamedTuple):
    """An exit code and the message that explains it."""

    code: int
    message: str


@dataclass(frozen=True)
class CanonicalReport:
    """What the canonical remotes said, gathered before anything is printed.

    Gathered before the line it joins is printed, because that line has to carry what the canonical
    remotes said whichever passing branch prints it. A plain dataclass rather than a Pydantic model:
    `tools/` imports nothing from `bko` and no checker here defines a model, and the checkers run
    under `uv run --no-sync`, so the lighter standard-library type is the one that cannot drift.

    Attributes:
        contained: Each remote whose trunk HEAD contains, with that commit.
        degraded: Each remote that could not answer, with why.
        refusal: Set when a canonical trunk is not contained or git could not say; else None.
    """

    contained: tuple[tuple[str, str], ...] = ()
    degraded: tuple[tuple[str, str], ...] = ()
    refusal: Refusal | None = None

    @property
    def stream(self) -> TextIO:
        """Where a passing line belongs: stderr when a remote could not answer, else stdout.

        The checker's two-stream contract puts only an outcome that needs nothing from the operator
        on stdout, and a pass a canonical remote could not confirm is not one.
        """
        return sys.stderr if self.degraded else sys.stdout

    def suffix(self) -> str:
        """The clause a passing line gains for the canonical trunks HEAD contains, or nothing."""
        if not self.contained:
            return ""
        named = ", ".join(f"`{remote}` ({commit[:8]})" for remote, commit in self.contained)
        return (
            f" HEAD also contains the trunk of {named}, so the canonical trunk carries nothing "
            "this tree has not seen."
        )

    def unestablished(self) -> str:
        """The paragraph a pass gains when a canonical remote could not answer, or nothing.

        **A remote with no `main` gets a different next action from an unreachable one.** Fetching
        can fix an unreachable remote and can never fix one whose trunk has another name, and both
        degrade every run until the operator acts, so the advice has to name the right act.
        """
        if not self.degraded:
            return ""
        reasons = "; ".join(f"`{remote}`: {why}" for remote, why in self.degraded)
        lines = [f"\nWhether HEAD contains the canonical trunk was NOT established: {reasons}."]
        unreached = [remote for remote, why in self.degraded if why != NO_TRUNK]
        unnamed = [remote for remote, why in self.degraded if why == NO_TRUNK]
        if unreached:
            lines.append(
                f"Run `git fetch {' '.join(unreached)}` once the remote answers, checking its URL "
                "with `git remote -v` if it never does, and re-run the gate to establish it."
            )
        if unnamed:
            lines.append(
                f"{', '.join(f'`{remote}`' for remote in unnamed)} has no `main`, so this check "
                "cannot measure its trunk. If it is not the canonical trunk, rename or remove the "
                "remote so the notice stops."
            )
        return "\n".join(lines)

    @classmethod
    def unanswered(cls, remote: str, commit: str) -> CanonicalReport:
        """A structural refusal: git could not say whether HEAD contains `remote`'s trunk.

        Args:
            remote: The canonical remote.
            commit: The trunk commit it advertised.

        Returns:
            A result whose refusal exits 2.
        """
        message = (
            f"git could not say whether HEAD contains `{remote}`'s trunk ({commit[:8]}): it did "
            "not run, or it reported a missing or corrupt object rather than an answer. Not a "
            "pass, and not a rebase either.\n"
            "Run `git fsck` to check the object store, then run the gate again."
        )
        return cls(refusal=Refusal(2, message))

    @classmethod
    def behind(cls, remote: str, commit: str, head: str, *, fetched: bool) -> CanonicalReport:
        """A refusal: HEAD does not contain `remote`'s trunk.

        Args:
            remote: The canonical remote.
            commit: The trunk commit it advertised.
            head: HEAD's commit.
            fetched: Whether this checkout holds a readable copy of the commit; it cannot contain
                one it never fetched, and saying which case it is names the right first step.

        Returns:
            A result whose refusal exits 1.
        """
        held = (
            f"HEAD ({head[:8]}) does not contain it"
            if fetched
            else "this checkout holds no readable copy of that commit"
        )
        message = (
            f"`{remote}` advertises {commit[:8]} as its trunk and {held}, so the canonical trunk "
            "carries commits this tree has never been measured against. A fork current with "
            f"`{REMOTE}` is still behind its upstream, which is the pass this check exists to "
            "refuse.\n"
            f"Run `git fetch {remote} && git rebase {remote}/main`, then run the gate again."
        )
        return cls(refusal=Refusal(1, message))


def refused(canonical: CanonicalReport) -> int | None:
    """Print a canonical refusal and return its code, or None when there is none.

    Every passing branch asks this first, after `origin` has had its say, so a checkout behind both
    is told about `origin` first and about `upstream` once that is fixed.

    Args:
        canonical: What the canonical remotes said.

    Returns:
        The refusal's exit code, or None to go on and print the pass.
    """
    if canonical.refusal is None:
        return None
    print(canonical.refusal.message, file=sys.stderr)
    return canonical.refusal.code


def report_no_trunk(canonical: CanonicalReport) -> int:
    """Report a checkout with no cached trunk: does not apply, unless a canonical remote refused.

    Args:
        canonical: What the canonical remotes said.

    Returns:
        A canonical refusal's code, else 0.
    """
    code = refused(canonical)
    if code is not None:
        return code
    line = (
        f"no `{TRUNK_LABEL}` ref in this checkout, so there is no trunk to compare against "
        "and this check does not apply. That is the case for a clone with no remote and for a "
        f"fork whose default branch has another name.{canonical.suffix()}"
    )
    print(line + canonical.unestablished(), file=canonical.stream)
    return 0


def report_lagging(trunk: str, remote: str, canonical: CanonicalReport) -> int:
    """Report a cache that lags behind a commit HEAD already contains, unless a canonical refused.

    Args:
        trunk: The cached trunk's commit.
        remote: The commit the remote advertises, which HEAD contains.
        canonical: What the canonical remotes said.

    Returns:
        A canonical refusal's code, else 0.
    """
    code = refused(canonical)
    if code is not None:
        return code
    line = (
        f"`{TRUNK_LABEL}` ({trunk[:8]}) is a stale cache -- the remote has {remote[:8]} "
        "-- but HEAD already contains that commit, so nothing on the trunk is unmeasured "
        f"and the tree is the tree that will land.{canonical.suffix()}"
    )
    print(line + canonical.unestablished(), file=canonical.stream)
    return 0


def report_current(head: str, trunk: str, why_not: str, canonical: CanonicalReport) -> int:
    """Print the passing line for a branch the trunk is an ancestor of, on the stream it belongs on.

    Args:
        head: HEAD's commit.
        trunk: The cached trunk's commit.
        why_not: Why `origin` could not be consulted, empty when it was.
        canonical: What the canonical remotes said.

    Returns:
        A canonical refusal's code, else 0: a degraded pass is still a pass, because offline must
        run.
    """
    code = refused(canonical)
    if code is not None:
        return code
    if not why_not:
        line = (
            f"`{TRUNK_LABEL}` ({trunk[:8]}) is an ancestor of HEAD ({head[:8]}) and matches "
            f"`{REMOTE}` as of now, so the measured tree is the tree that will land there."
            f"{canonical.suffix()}"
        )
        print(line + canonical.unestablished(), file=canonical.stream)
        return 0
    # **To stderr, with no age judgement, and with a next action.** On stdout it would be hidden
    # by `make check >/dev/null`, which is how triage on stderr works, and the two-stream
    # contract has a slot for an outcome that is neither a failure nor a confirmed pass. No age
    # can reveal staleness, which `tests/test_repo_checkers.py` holds as a control. It stays
    # exit 0 because offline must run.
    print(
        f"`{TRUNK_LABEL}` ({trunk[:8]}) is an ancestor of HEAD ({head[:8]}), so the tree is "
        "current with respect to the last fetch. Whether that is still the remote's trunk "
        f"was NOT established: {why_not}. The age of the cached commit cannot tell you -- "
        "staleness is the remote having moved, not the cache being old.\n"
        f"Run `git fetch {REMOTE}` once you have a network and re-run the gate to establish "
        f"it.{canonical.suffix()}{canonical.unestablished()}",
        file=sys.stderr,
    )
    return 0
