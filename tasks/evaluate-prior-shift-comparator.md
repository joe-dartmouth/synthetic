---
id: evaluate-prior-shift-comparator
title: Add the prior-shift arm, a scoring-time logit offset that adds no rows
status: blocked
depends_on: plan-direct-method-research, refactor-evaluate-size-splits
blocks: evaluate-control-runs-registered
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/prior-shift-comparator
---

# Add the prior-shift arm, a scoring-time logit offset that adds no rows

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

E6's gain for `bko-direct` is at the default cutoff with ranking unchanged: flagged AUC-PR reads
level with `none` ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal),
E6 Result). That is the pattern a shift in the subgroup's risk level produces. An arm that makes
only that shift, adding no rows, measures how much of the gain it explains. The control reading
rule in [`docs/RESEARCH.md`](../docs/RESEARCH.md) reads it, and the IDEAS entry
[A threshold-shift comparator for E6](../docs/IDEAS.md#a-threshold-shift-comparator-for-e6)
proposes it.

## Scope

**In.**

- Arm `prior-shift` in a new `src/bko/evaluate/shift.py`. For each requested cell's (age band,
  sex) subgroup: the target prevalence after the gap request's fill; a logit offset solved by
  bisection on the training subgroup's predictions; the offset applied to that subgroup's test
  logits; the cutoff unchanged.
- Wired in `run._arm_scores` and `panel.REGENERATED` (`bko.evaluate`).
- A new ADR under `docs/adr/research/` that records the arm.
- Deletes the IDEAS entry A threshold-shift comparator for E6.

**Out.**

- A weighting method: it is undefined when the cell is empty.
- Estimating the shift from test records.

## Acceptance

- The offset matches a hand-computed literal on a fixture.
- The offset is finite when the cell holds no record.
- Every other arm's `scores.csv` bytes are equal with and without the arm on a fixture run.
- `make check` passes.
