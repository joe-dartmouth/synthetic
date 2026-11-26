---
id: paper-scarcity
title: Register a paired support-removal and knowledge-quality sweep
status: blocked
depends_on: evaluate-confirmatory-registration, ontology-auto-mapping, paper-baselines
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

Reuse matching P3 masks and existing splits. Nest removals within several prespecified cells per dataset from full support through zero. Keep the test split fixed and train every method only on retained rows. For BKO-direct, perturb the field-to-concept mapping used to select donors. At each severity, use the same seeded subset of mappings for plausible incorrect assignments and for deletion. Require a fixture where the perturbation changes the eligible donor pool and the generated records with the seed fixed. Treat a closure-only edge deletion as a trace control. Test probability or relation corruption only on a named variant whose draw consumes that input. Include zero external information and measure the amount retained.

**Findings from planning.**

- The mapping perturbation reuses the permutation unit `ontology-auto-mapping` adds
  (`bko.ontology.automatic.permuted`).
- "P3 masks" means P3's registered removals (`bko.evaluate.runs_published.P3`), not its results.

## Acceptance

Register the full-data reference, no augmentation, oversampling, matched controls, neural generators, TabPFN and BKO. Fix subgroup and overall recall, AUC-PR, false-positive rate at a training-selected cutoff, calibration metric, denominators and paired intervals. Set undefined metric and multiplicity rules. Target four to six real datasets including nonmedical ACS. Register distinct sweep entries after the signed confirmatory registration releases the protected frames. Keep ACS exploratory.
