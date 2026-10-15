---
id: proposal-baselines
title: Baseline augmentation methods
status: ready
depends_on: --
blocks: proposal-final-evaluation
parallel_safe: true
owner: Joe
milestone: status-update-1
paid: false
adrs: --
---

# Baseline augmentation methods

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The primary criterion compares against CTGAN, TVAE and SMOTE at equal row counts, with
no-augmentation and class-weighting controls; ADASYN is listed too
([RESEARCH section 4](../docs/RESEARCH.md#4-success-criteria)). Status Update 1 asks for baselines locked.

## Scope

**In.**

- No augmentation, class weighting, SMOTE and ADASYN (imbalanced-learn), CTGAN and TVAE (SDV) in
  `src/bko/baselines/`, each producing a dataset of a requested row count.

**Out.** Scoring the results, which is `proposal-eval-harness`.

## Acceptance

- Each method runs from one command with a fixed seed and writes its row count.
- The `synth` extra joins `make setup`.
- `make check`.
