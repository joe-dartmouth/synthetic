---
id: paper-scarcity-results
title: Measure scarcity curves and knowledge failure regimes
status: blocked
depends_on: paper-scarcity, evaluate-confirmatory-results
blocks: paper-manuscript
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Measure scarcity curves and knowledge failure regimes

The set's graph is [direct-index.md](direct-index.md).

## Why this unit exists

A cross-dataset curve can show where outside knowledge helps and where its errors outweigh its benefit.

## Scope

Score the named development baseline runs on prediction and correlation fidelity. Run the registered scarcity matrix after the tagged confirmatory results have been recorded. Preserve split and mask identities, per-cell counts, compute, skipped comparisons and uncertainty. Compare reconstruction against removed records only after fitting. Separate knowledge uncertainty from variation across splits.

## Acceptance

Produce a reproducible figure of performance versus remaining cell count with paired intervals and a full-data reference. Show dataset and subgroup variation, calibration and false-positive costs. Report crossover regions or the absence of one and compare corruption and deletion at matched severity.
