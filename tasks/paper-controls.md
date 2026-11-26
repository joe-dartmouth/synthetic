---
id: paper-controls
title: Isolate the contribution of external knowledge
status: ready
depends_on: --
blocks: paper-control-results
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Isolate the contribution of external knowledge

The set's graph is [direct-index.md](direct-index.md).

## Why this unit exists

E9, E12, E13 and E15 supply related controls. A change in donor selection or field dependence can also change predictive utility.

## Scope

Audit their actual draws against a matched design. Keep split, quota, seed, field grouping, donor eligibility and classifier fixed wherever compatible. Pair BKO-direct with the identical sampler using empirical weights alone. Include conditional row resampling, independent empirical marginals, empirical Bayes shrinkage, and a hierarchical Bayesian partial-pooling baseline fitted on retained training cells. Add subgroup reassignment of positive donors with other fields fixed as a diagnostic, without a causal interpretation.

## Acceptance

Register a development run with an information-source table. Include an ontology-only draw using no empirical donors or fitted marginals. Mark fields K cannot generate and compare on common supported fields. Test that each arm reads only its declared inputs and that swapping K changes only the intended draw component.
