---
id: paper-baselines
title: Prepare modern synthesis and prediction comparators
status: blocked
depends_on: paper-control-results
blocks: paper-scarcity
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Prepare modern synthesis and prediction comparators

The set's graph is [direct-index.md](direct-index.md).

## Why this unit exists

Predictive utility can survive imperfect joint fidelity. A strong predictor on scarce real data can also close the observed gap.

## Scope

Reuse TabDDPM. Pin TabSyn and ReTabSyn implementations and weights. Add TabPFN trained or conditioned on the same retained real examples without augmentation. Document limits, tuning budget, training time and compute. Keep test outcomes out of reward design, model selection and preprocessing.

## Acceptance

A small common-split fixture exercises each comparator. Register named development runs, score prediction and correlation fidelity separately, and document unsupported frames and short fills.
