---
id: evaluate-quota-sweep-results
title: Run the quota sweep and read representativeness and recall per count
status: blocked
depends_on: evaluate-diabetes-control-results, gaps-share-based-quota
blocks: evaluate-composite-arm
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

- On E6's cell, computed on holdout 0 (the unit's first step recomputes the counts with
  `bko.gaps.quota`, which unit `gaps-share-based-quota` adds), only `coverage` and its multiples
  ask more than 30 rows away from the fixed fill; `expected-size` and `recommended` ask within 30 rows of it.

## Acceptance

- The entry is `complete`, with the representativeness measure and recall at each count per arm.
- `make check` passes.
