---
id: evaluate-quota-sweep-results
title: Run the quota sweep and read representativeness and recall per count
status: blocked
depends_on: evaluate-diabetes-control-results
blocks: evaluate-composite-arm, paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/share-based-quota
---

# Run the quota sweep and read representativeness and recall per count

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Whether the recommended count makes a cell representative, and what it costs or buys in recall,
is an empirical question. `RUNS` holds the sweep, E21
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e21-share-based-quota-sweep-on-the-diabetes-130-removal);
ADR [`research/share-based-quota`](../docs/adr/research/share-based-quota.md)); this unit runs it.

## Scope

**In.**

- The sweep's runs and readings for every arm at every count. The arms are those of the E6
  family of [E12](../docs/EVIDENCE.md#e12-control-arms-on-the-diabetes-130-removal), all
  free; no paid Messages arm runs.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Runs, Result and Read against.

**Out.**

- Counts not registered.

**Findings from planning.**

- The runs, their slices, the fixed fill they read against and the privacy reference arm:
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e21-share-based-quota-sweep-on-the-diabetes-130-removal),
  E21's Configuration. E12's runs are this unit's prerequisite, since `compare --beside` reads
  the fixed fill from them.
- The `quota-rule` line, the `representativeness` command and its columns:
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation). How far each count sits from the
  fixed fill at holdout 0: ADR
  [`research/share-based-quota`](../docs/adr/research/share-based-quota.md), Rationale.

## Acceptance

- The entry is `complete`, with the representativeness measure and recall at each count per arm.
- `make check` passes.
