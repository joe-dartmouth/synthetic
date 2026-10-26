---
id: evaluate-data-scarcity-curve
title: Augmentation against training-set size
status: blocked
depends_on: evaluate-e1-second-configuration, evaluate-named-runs
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# Augmentation against training-set size

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

Whether augmentation beats resampling is mixed in the literature
([`docs/RESEARCH.md`](../docs/RESEARCH.md#7-augmentation-versus-resampling)), and the answer may
depend on how scarce the real data is. UCI's 238 training rows are one point. Training each arm on
a quarter, a half and three quarters of them, with the test split fixed, draws the curve and says
whether knowledge guidance pays when data is scarcer than UCI.

## Scope

**In.**

- A training fraction in a named run's `Experiment`, 1 by default: each holdout's training rows
  are subsampled per cell by a seeded draw to the fraction, rounded half up and never below 2
  records, since SMOTE and ADASYN refuse a stratum under 2 (`bko.baselines.oversample`); the test
  split is unchanged, and every stage that reads the training rows, the gaps, the guidance, the
  baselines and the classifier, reads the subsample.
- Three named runs (`evaluate-named-runs`) on the judged configuration, fractions 0.25, 0.5 and
  0.75, each with the baselines and the `bko` arm, its gap and replica fills live. Spend estimate
  per run: one arm's share of the judged run's spend, its replica part scaled by the fraction and
  its gap-fill part by the rows the subsample's shortfall asks over the rows the full split's
  asks, since the shortfall grows as rows leave, stated in the Runs row before the run, under the
  run's ceiling and the protocol of `evaluate-run-protocol`.
- Per fraction: each arm's recall and AUC-PR over the whole test split and the flagged subgroups
  with each contrast's corrected interval, and which cells fall under the threshold, in a new
  `docs/EVIDENCE.md` entry that reads the judged run as the fraction-1 point.

**Out.** Changing the test split; a verdict.

## Acceptance

- Each fraction's measures regenerate from `score <run>` and `compare <run>`, its run is
  recorded with its spend and replays byte for byte, and a test pins the subsample draw and its
  floor on a fixture split.
- `make check`.
