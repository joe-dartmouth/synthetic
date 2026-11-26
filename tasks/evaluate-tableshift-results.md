---
id: evaluate-tableshift-results
title: Run the TableShift readmission runs and read them beside its Table 7
status: blocked
depends_on: evaluate-p1-partial-replicas
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
`docs/EVIDENCE.md` holds the Configuration. This unit runs `p1-id-partial` and `p1-ood-partial`
and records what they show beside TableShift's published baselines.

It waits on unit [`evaluate-p1-partial-replicas`](evaluate-p1-partial-replicas.md), which adds
those runs. The registered runs `p1-id` and `p1-ood` cannot complete: runs `p1-1` and `p1-2` each
exit 2 on holdout 0, when TVAE's replica fill falls short of TableShift's rare cells on all 5
attempts and is refused (P1 Runs).

## Scope

**In.**

- Run `p1-id-partial` and `p1-ood-partial` from a detached worktree, as
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) gives the steps for the arms the
  runs name.
- The EVIDENCE entry's Runs, Result and Read against, beside TableShift's Table 7. The Result
  records, per holdout and arm, each partial fill's short cells and their missing rows, read
  from the `shortfall` lines, and each partial replica's row count beside the rows it asked.

**Out.**

- Any change to the split, an arm or a reading.

## Acceptance

- The entry's status is `complete`.
- The entry's Read against compares with TableShift's Table 7.
- Each run's sha256 is recorded.
- The Result names every partial replica, so none reads as a full copy of the training cells.
- `make check` passes.
