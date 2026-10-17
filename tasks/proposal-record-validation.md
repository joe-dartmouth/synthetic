---
id: proposal-record-validation
title: Record consistency checks
status: ready
depends_on: --
blocks: proposal-fusion-repair
parallel_safe: true
owner: Shirley
milestone: status-update-2
paid: false
adrs: --
---

# Record consistency checks

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

Each candidate record is validated before augmentation, and the validity criterion asks for at least
95% passing ([RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria)).

## Scope

**In.**

- Consistency checks in `src/bko/validate/` built from the mapping table's range constraints and
  ontology relationships, with a per-check rejection count.

**Out.** Generating the records.

## Acceptance

- Planted invalid records are each rejected by the check that should catch them.
- `make check`.
