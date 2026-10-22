---
status: accepted
topic: tooling
created: 2026-10-21
updated: 2026-10-21
---
# Run folders: one gitignored folder per run of record, artifacts under fixed role names

## Context

Runs of record keep their artifacts as `scratchpad/<run-id>.<ext>`, with each entry's own
extensions: E1 writes `.jsonl`, `.csv` and `.err`, C1 writes `.out` and `.closure.err`. R1 and E2
wrote fixed names that later runs overwrote. The Streamlit app must list the runs on the machine
and open each one's artifacts without knowing the command that wrote them.

## Decision Drivers

* **Must keep every EVIDENCE path, hash and recorded command true**: the register cites
  `scratchpad/` paths in its Runs and Result fields.
* **Must let one reader open any run by role**, whatever entry it belongs to.
* **Should stay local**: paid call logs reach hundreds of MB.

## Decision

We will keep each run of record's artifacts in **`runs/<run-id>/`**, gitignored. Each artifact
takes a role name from `bko.app.runs.ROLES`, which `docs/DEVELOPMENT.md` (Runs) mirrors in a
table. The guarded commands in `docs/DEVELOPMENT.md` write there. A run folder's name is the run's
EVIDENCE id; a run that checks or extends another takes that run's id and a suffix, such as
`e1-3-replay` or `c1-2-closure`. The latest run of each entry (`e1-3`, `c1-2`, `c1-2-closure`,
`g1-13`, `e2-5`, `r1-2`) is hard-linked into its folder; earlier runs stay where their rows say.

## Rationale

Hard links give the standard layout with no second copy and no rewritten record: the linked file
is the register's artifact under a second name, so its path, sha256 and command stay true. Role
names let one reader serve every entry, and the register stays the one home of each run's
identity.

## Consequences

**Gained**: one layout for every run and for the app.
**Accepted**: a run shows only on a machine that holds its folder; copying a folder to another
machine is manual; rows of earlier runs keep their `scratchpad/` paths.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Committing small results | The team keeps run artifacts local, and paid call logs are large | 2026-10-21 |
| Indexing `scratchpad/` names in place | Each entry uses its own extensions, and fixed-name runs keep only their latest | 2026-10-21 |
| Moving the files into `runs/` | Falsifies the Runs and Result fields that cite `scratchpad/` paths and the commands as run | 2026-10-21 |
| Copying the files into `runs/` | Duplicates call logs of hundreds of MB, and two copies can drift | 2026-10-21 |
| A manifest per run | EVIDENCE already holds each run's identity; one home per kind of statement | 2026-10-21 |

## Enforcement
- Owner: `docs/DEVELOPMENT.md` (Runs)
- Pinned by: `tests/test_run_folders.py`

## Reversal Conditions
Earlier runs are needed on several machines (then commit derived results or sync the folders), or
a run writes an artifact that no role names.

## Related
- [E1 evaluation protocol](../research/e1-evaluation-protocol.md)
- [Call-log replay](../research/call-log-replay.md)
- [One home per kind of statement](one-home-per-kind-of-statement.md)
