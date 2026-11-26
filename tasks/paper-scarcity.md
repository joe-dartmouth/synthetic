---
id: paper-scarcity
title: Register a paired support-removal and knowledge-quality sweep
status: blocked
depends_on: paper-control-results, paper-baselines
blocks: paper-scarcity-results
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Register a paired support-removal and knowledge-quality sweep

The set's graph is [direct-index.md](direct-index.md).

## Why this unit exists

E5 varies training size, E18 varies synthetic share and P3 supplies removal fractions. The paper needs performance indexed by surviving target-cell observations.

## Scope

Reuse matching P3 masks and existing splits. Nest removals within several prespecified cells per dataset from full support through zero. Keep the test split fixed and train every method only on retained rows. Cross intact K with seeded levels of incorrect probabilities or relations and separately with partial deletion. Include zero external information and measure the amount retained.

## Acceptance

Register the full-data reference, no augmentation, oversampling, matched controls, neural generators, TabPFN and BKO. Fix subgroup and overall recall, AUC-PR, false-positive rate at a training-selected cutoff, calibration metric, denominators and paired intervals. Set undefined metric and multiplicity rules. Target four to six real datasets including nonmedical ACS. Coordinate untouched datasets with confirmatory registration before scoring and keep ACS exploratory.
