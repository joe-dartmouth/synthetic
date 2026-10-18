---
id: proposal-fusion-repair
title: Fuse validated records and measure gap closure
status: blocked
depends_on: proposal-llm-generation
blocks: proposal-streamlit-app, proposal-final-evaluation
parallel_safe: true
owner: Shirley
milestone: status-update-2
paid: false
adrs: --
---

# Fuse validated records and measure gap closure

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The completeness criterion asks that at least half the detected gaps close after fusion
([RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria)). Fusion is in Shirley's workstream while `fuse`
lives in `bko.bkb`; this unit settles the interface between the two. `fuse` takes fragments and one
reliability each, and refuses a fused BKB as a fragment (ADR `research/bkb-layer-source`).

## Scope

**In.**

- Fuse validated records back into the BKO and re-run gap detection under each definition of ADR
  `research/bko-formal-gap-definition`, counting each one's closed (cell, statement) pairs; the
  completeness criterion reads the per-subgroup count.
- Run completeness on its own request, as ADR `research/bko-formal-gap-definition` (Decision,
  Completeness) states: 20 rows for each cell holding a per-subgroup gap in the default training
  split, the validated records joining their cell's records before detection runs again.

**Out.** The downstream prediction comparison.

## Acceptance

- Gap closure is computed from one command and recorded in `docs/EVIDENCE.md`.
- `make check`.
