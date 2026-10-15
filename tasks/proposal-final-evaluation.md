---
id: proposal-final-evaluation
title: Final evaluation
status: blocked
depends_on: proposal-eval-harness, proposal-fusion-repair
blocks: --
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

- Run E1 (UCI Heart Disease augmentation) and E2 (Synthea ablation) as configured in
  `docs/EVIDENCE.md`, including the null-result case.

**Out.** Slides and the demo itself.

## Acceptance

- Every number in the report is regenerable from a committed command and recorded in
  `docs/EVIDENCE.md`, with each run's spend in its Runs field.
- `make check`.
