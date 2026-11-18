---
id: evaluate-quota-sweep-results
title: Run the quota sweep and read representativeness and recall per count
status: blocked
depends_on: gaps-share-based-quota
blocks: --
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

- The sweep's runs and readings for every arm at every count.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Runs, Result and Read against.

**Out.**

- Counts not registered.

## Acceptance

- The entry is `complete`, with the representativeness measure and recall at each count per arm.
- `make check` passes.
