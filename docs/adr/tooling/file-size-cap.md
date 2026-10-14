---
status: accepted
topic: tooling
created: 2026-10-14
updated: 2026-10-14
---
# Per-tree line caps on Python files

## Context

Long modules are where review stops reading carefully. ruff lints per line and per construct; it
has no rule that limits a file's length. The reference repository caps file length with a small
checker run by its gate, and this repository adopts the same checker from its first commit, so
there is no backlog of over-cap files to split.

## Decision Drivers

* **Must refuse an over-cap file before it reaches `main`**, as part of `make check`.
* **Should allow tests more room than library code**, since a test file grows with its cases.
* **Must hold its own numbers in one place.**

## Decision

We will use **per-tree caps in a `make check` target**: `tools/check_file_size.py`, run by
`make sizes`, caps every tracked `.py` file under `src/`, `tools/` and `tests/`. `CAPS` in that
module is the single home for the numbers, with `tests/` allowed more lines than `src/` and
`tools/`. A tracked `.py` file outside those trees is a structural error.

## Rationale

A checker run by the gate is the only enforcement available while ruff has no per-file-length rule.
Applying it from the first commit means no grandfather list is ever needed. Counting with
`data.count(b"\n")` matches `wc -l`.

## Consequences

**Gained**: no module grows past what a reviewer reads in one sitting; the cap is visible in one
tuple, `CAPS`.
**Accepted**: a cohesive module just over the cap is split anyway. An untracked file is measured
only once it is staged, because discovery reads `git ls-files`.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A ruff rule | ruff has no per-file-length rule | 2026-10-13 |
| No cap | Long modules are where review attention fails | 2026-10-13 |
| Including untracked files | Puts every scratch file an author has not decided about under the gate | 2026-10-13 |

## Enforcement

- Owner: `tools/check_file_size.py`
- Pinned by: `tests/test_file_size.py`
- Exit codes are the contract: 0 clean, 1 over cap, 2 structural (`git ls-files` failed, a tracked
  `.py` matched no cap, or fewer files were found than `DISCOVERY_FLOOR`).
- `tools/` is in the coverage `source`, so the checker is held to the same branch floor as the
  library.

## Reversal Conditions

**A per-file-length rule in ruff retires `tools/check_file_size.py`.** The only reason for custom
code is that ruff lacks one; `ruff rule --all` answers whether it exists.

**Discovery widens to untracked files the first time an over-cap file reaches `main` through the
untracked window**, using `git ls-files --others --exclude-standard`.

## Related

- [uv, ruff and ty](uv-ruff-ty.md)
- [Refuse a stale trunk ref](refuse-a-stale-trunk-ref.md)
