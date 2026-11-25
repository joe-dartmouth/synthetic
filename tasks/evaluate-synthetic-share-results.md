---
id: evaluate-synthetic-share-results
title: Run the synthetic-share ablation and record each share and arm
status: ready
depends_on: --
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the synthetic-share ablation and record each share and arm

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

[E18](../docs/EVIDENCE.md#e18-synthetic-share-sweep-on-the-diabetes-130-frame) holds the runs
`e18-two` to `e18-all` and their Configuration. This unit runs them and records what each share
shows. Every arm is free.

## Scope

**In.**

- Run the registered runs from a detached worktree with `score`, `rows`, `panel` and `privacy`,
  as [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) gives them. No API call.
- The EVIDENCE entry's Runs, Result and Read against, recorded per share and arm.

**Out.**

- Any change to a share, an arm or a reading.

## Acceptance

- The entry's status is `complete`.
- Each run's sha256 is recorded.
- `make check` passes.
