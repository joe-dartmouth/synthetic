---
id: proposal-final-evaluation
title: Final evaluation
status: blocked
depends_on: proposal-fusion-repair
blocks: proposal-guidance-ablation
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# Final evaluation

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The Final Report milestone reads every criterion against the baselines and controls
([RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria), [ROADMAP](../docs/ROADMAP.md#final-report)).

## Scope

**In.**

- Register the three BKO arms of ADR `research/e1-evaluation-protocol` in `bko.evaluate`: `bko`,
  `bko-population` and `bko-frequency`, each running gap detection under its definition (ADR
  `research/bko-formal-gap-definition`), generation, validation and fusion on a holdout's training
  rows only, and holding to that record's rule for added rows. Add all three to `E1.arms` and none
  to `E1.comparators`, and give `E1` the record's budget per attempt. Expose the per-row rules of
  `bko.baselines.records.check` so the runners drop rows by them rather than by a copy. With the BKO arms in
  `E1.arms`, make `Experiment` refuse a candidate that is not an arm, so `score` cannot run a
  configuration `compare` will refuse.
- Decide whether the BKO arms join `E1.replica_arms`, all three or none; `docs/RESEARCH.md`
  section 5 reports fidelity for every generator, so leaving them out is recorded there.
- Compute each BKO arm's validity rate as ADR `research/e1-evaluation-protocol` sums it, and record
  the rates, with gap closure, in `docs/EVIDENCE.md`.
- Run E1 (UCI Heart Disease augmentation) and E2 (Synthea ablation) as configured in
  `docs/EVIDENCE.md`, including the null-result case.

**Out.** Slides and the demo itself.

## Acceptance

- Every number in the report is regenerable from a committed command and recorded in
  `docs/EVIDENCE.md`, with each run's spend in its Runs field.
- Before the judged run, E1's Configuration in `docs/EVIDENCE.md` names the budget per attempt.
- `make check`.
