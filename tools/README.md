# `tools/` -- the repository's own checkers

Run by `make check` and imported by nothing under `bko`. `tests/` imports them freely, including
`_git`. The Make targets that run them are described in
[`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#make-targets).

| Module | Role |
|--------|------|
| `check_file_size.py` | `make sizes`: refuses a tracked Python file over its per-tree line cap. `CAPS` is the only home for the numbers, and the module is under its own cap (ADR `tooling/file-size-cap`). |
| `check_lock_drift.py` | `make synced`: compares the installed environment against `uv.lock`, for the packages that are installed. Refuses a lock it cannot read as one version per package. |
| `check_rebased.py` | `make rebased` / `rebased-final`: refuses a tree the trunk has moved past (ADR `tooling/refuse-a-stale-trunk-ref`). |
| `_git.py` | Not a checker. The one `git` runner, its timeout, and the toplevel guard that must be paired with it. |
| `_remote.py` | The trunk-freshness probe: asks a remote where its trunk is and returns a commit or a reason, never a verdict. |
| `_rebased_report.py` | Holds what a configured `upstream` said and prints the lines a passing `make rebased` ends on. It runs no git, so the checker's sentinel still covers every call. |

## Contracts

- **Exit codes**: `0` clean, `1` violated, `2` structural. Make reads only zero against non-zero;
  the `1`/`2` split is for the operator and for `tests/`.
- **Streams**: stdout carries only an outcome that needs nothing from the operator, such as a
  success line. Every failure goes to stderr, and so does an exit-`0` run the operator must read:
  `check_rebased.py`'s pass whose freshness was not established, and its abstention mid-rebase or
  mid-bisect.
- **Structural refusals name their cause.** Exit `2` is never a generic "could not check": each
  message says what could not be read or which git call failed, and for git whether it did not run
  or which status it exited with. `check_lock_drift.py` refuses a lock it cannot read as one version
  per package rather than skipping an entry it cannot read; `check_file_size.py` refuses a tracked
  file it cannot read; `check_rebased.py` refuses when git cannot say whether `upstream` is
  configured. A traceback would exit `1` and read as a violation, so git output is decoded with
  `surrogateescape` in `_git.py` and a path that is not valid UTF-8 is measured or refused, never
  raised.
- **Anti-vacuity floors**: a checker that finds nothing exits clean and reads like a clean tree, so
  `check_file_size.py` carries `DISCOVERY_FLOOR` and `check_lock_drift.py` carries
  `COMPARISON_FLOOR`. `check_rebased.py` cannot have one, since "no trunk ref" is a legitimate pass;
  it answers the concern structurally instead, refusing when git answers about a different
  repository or could not be consulted at all.
- **The git runner and the toplevel guard go together.** Environment scrubbing without the guard
  gives a false pass about an enclosing repository. `tests/test_git_runner.py` scans every module
  that has a `main` for the pairing, selecting on `main` rather than on a name list so a helper
  cannot be swept in by name; `_remote.py` and `_rebased_report.py` have none.
- **Invoke the two git-running checkers, `check_file_size` and `check_rebased`, as modules**
  (`python -m tools.check_file_size`), not as scripts: a script puts `tools/` on `sys.path` rather
  than the repository root, so the package import fails.
