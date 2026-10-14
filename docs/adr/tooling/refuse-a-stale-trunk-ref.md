---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-10-14
---
# Refuse a gate run on a stale trunk ref

## Context

Four people branch off `main` at once. Two pull requests can each pass `make check` and still merge
into a red trunk: one renames something, the other branched before the rename landed, so its gate
never saw it. There is no CI to gate the merge result, so the local gate must at least refuse to
measure a tree that is behind the trunk.

`origin/main` is a remote-tracking ref: a local cache of whatever the last `git fetch` saw. An
ancestry test against that cache proves only that the branch is current with respect to the cache.
When other pull requests merge while a branch is in progress, the cache can point behind them and
the ancestry test still passes.

## Decision Drivers

* **Must refuse a branch that does not contain the current trunk**, measured against the remote and
  not only the local cache.
* **Must keep every step of `make check` read-only**, so the gate never moves refs.
* **Must not read an unreachable remote as agreement.**

## Decision

`make rebased` runs `tools/check_rebased.py`, which:

1. Probes the remote with `git ls-remote origin refs/heads/main` and refuses with exit 1 when the
   remote's trunk differs from the cached `origin/main` and `HEAD` does not already contain it.
2. Requires `origin/main` to be an ancestor of `HEAD`, refusing with exit 1 otherwise.
3. Probes a configured `upstream` remote as well and refuses unless `HEAD` contains its trunk.
4. When a probe cannot answer (no remote, no network, an unreadable credential, a timeout), proceeds
   on the ancestry test alone and says so on stderr in a different sentence from a confirmed pass.
   A local git failure (git did not run, or could not say whether `upstream` is configured) is a
   structural exit 2 that names the cause.

`make check` asks the question twice: `rebased` first, and `rebased-final` last, so a merge that
lands while the tests run is refused at the end.

## Rationale

`ls-remote` reads the remote without writing anything locally, which a `git fetch` would; the gate
stays read-only. A stale ref is exit 1 rather than exit 2 because it is the same defect one step
earlier: after fetching, the branch is behind, and the remedy is the same rebase. Asking again at
the end closes the window in which a teammate's merge lands during a test run.

This gates the merge base, not the merge result. With squash merges, a trunk that is an ancestor of
`HEAD` carries nothing the branch has not seen, so the measured tree is the tree that lands.

## Consequences

**Gained**: two green branches can no longer merge into a red trunk unnoticed, as long as each
rebases before its gate.
**Accepted**: the gate needs network access to give a confirmed verdict; offline, it passes on the
ancestry test with a degraded notice. The checker runs `git` with global config blanked, so an
HTTPS remote that relies on a global credential helper is not probed and the run degrades (Apple
git still reads its Xcode gitconfig, whose `osxkeychain` helper applies);
`docs/DEVELOPMENT.md` asks for an SSH remote. A merge after the gate and before the squash is
outside what a local check can see.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Ancestry against the cached ref only | Passes on a cache that is behind merged work | 2026-10-13 |
| `git fetch` inside the gate | Writes refs; every other gate step is read-only | 2026-10-13 |
| A CI merge queue | There is no CI | 2026-10-13 |

## Enforcement

- Owner: `tools/check_rebased.py`
- Pinned by: `tests/test_repo_checkers.py`
- Also `tests/test_rebased_second_remote.py` (the `upstream` probe) and `tests/test_git_runner.py`
  (the git runner and its toplevel guard).

## Reversal Conditions

**A CI workflow with a required up-to-date-branch rule makes the local check redundant.** The day
`main` gains a branch-protection rule requiring branches to be current before merging,
`gh api repos/joe-dartmouth/synthetic/branches/main/protection` shows it and this check can be
retired from `make check`.

## Related

- [Per-tree line caps](file-size-cap.md)
- [uv, ruff and ty](uv-ruff-ty.md)
