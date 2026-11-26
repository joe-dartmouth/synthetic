---
id: evaluate-quota-sweep-results
title: Run the quota sweep and read representativeness and recall per count
status: blocked
depends_on: evaluate-diabetes-control-results, gaps-share-based-quota
blocks: evaluate-composite-arm, paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the quota sweep and read representativeness and recall per count

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Whether the recommended count makes a cell representative, and what it costs or buys in recall,
is an empirical question. Unit [`gaps-share-based-quota`](gaps-share-based-quota.md) registers the
sweep; this unit runs it.

## Scope

**In.**

- The sweep's runs and readings for every arm at every count. The arms are those of the E6
  family of [E12](../docs/EVIDENCE.md#e12-control-arms-on-the-diabetes-130-removal), all
  free; no paid Messages arm runs.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Runs, Result and Read against.

**Out.**

- Counts not registered.

**Findings from planning.**

- The unit's first step recomputes E6's counts with `bko.gaps.quota`, which unit
  [`gaps-share-based-quota`](gaps-share-based-quota.md) adds; how far each count sits from the
  fixed fill is that unit's ADR's to state.

## Acceptance

- The entry is `complete`, with the representativeness measure and recall at each count per arm.
- `make check` passes.
