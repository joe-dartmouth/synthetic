---
id: evaluate-tableshift-results
title: Run the TableShift readmission runs and read them beside its Table 7
status: ready
depends_on: --
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/tableshift-comparison
---

# Run the TableShift readmission runs and read them beside its Table 7

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

ADR [`research/tableshift-comparison`](../docs/adr/research/tableshift-comparison.md) records the
split, and the [P1 entry](../docs/EVIDENCE.md#p1-tableshift-readmission-split) of
`docs/EVIDENCE.md` names the runs `p1-id` and `p1-ood` and their Configuration. This unit runs them
and records what they show beside TableShift's published baselines.

## Scope

**In.**

- Run the registered runs from a detached worktree, as
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) gives the steps for the arms the
  runs name.
- The EVIDENCE entry's Runs, Result and Read against, beside TableShift's Table 7.

**Out.**

- Any change to the split, an arm or a reading.

## Acceptance

- The entry's status is `complete`.
- The entry's Read against compares with TableShift's Table 7.
- Each run's sha256 is recorded.
- `make check` passes.
