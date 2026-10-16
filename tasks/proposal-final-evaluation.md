---
id: proposal-final-evaluation
title: Final evaluation
status: blocked
depends_on: proposal-fusion-repair
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

- Register the BKO arm in `bko.evaluate`: a runner named `bko`, added to `E1.arms`, that runs gap
  detection, generation, validation and fusion on a holdout's training rows only and returns
  exactly the rows per cell the gap request asks for (ADR `research/e1-evaluation-protocol`).
  With `bko` in `E1.arms`, make `Experiment` refuse a candidate that is not an arm, so `score`
  cannot run a configuration `compare` will refuse.
- Decide whether the BKO arm joins `E1.replica_arms`; `docs/RESEARCH.md` section 5 reports
  fidelity for every generator, so leaving it out is recorded there.
- Compute the validity rate (records passing every check over records generated, from
  `bko.validate`'s counts) and record it, with gap closure, in `docs/EVIDENCE.md`.
- Run E1 (UCI Heart Disease augmentation) and E2 (Synthea ablation) as configured in
  `docs/EVIDENCE.md`, including the null-result case.

**Out.** Slides and the demo itself.

## Acceptance

- Every number in the report is regenerable from a committed command and recorded in
  `docs/EVIDENCE.md`, with each run's spend in its Runs field.
- `make check`.
