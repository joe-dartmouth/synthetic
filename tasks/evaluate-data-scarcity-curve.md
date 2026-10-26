---
id: evaluate-data-scarcity-curve
title: Augmentation against training-set size
status: blocked
depends_on: evaluate-e1-second-configuration
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

- A training fraction in `E1`, 1 by default: each holdout's training rows are subsampled per cell
  by a seeded draw to the fraction, the test split unchanged; every stage that reads the training
  rows, the gaps, the guidance, the baselines and the classifier, reads the subsample.
- Runs at fractions 0.25, 0.5 and 0.75 on the judged configuration: the baselines free; the BKO
  arms paid at each fraction, since their guidance and replica read the subsample, at an estimate
  of the fraction times the judged run's replica spend plus its gap-fill spend, stated before each
  run; each run detached, kept awake, on mains power and watched as `e1-3` was.
- Per fraction: each arm's recall and AUC-PR over the whole test split and the flagged subgroups
  with each contrast's corrected interval, and which cells fall under the threshold, in a new
  `docs/EVIDENCE.md` entry that reads the judged run as the fraction-1 point.

**Out.** Changing the test split; a verdict.

## Acceptance

- Each fraction's measures regenerate from a committed command and its run is recorded with its
  spend.
- `make check`.
