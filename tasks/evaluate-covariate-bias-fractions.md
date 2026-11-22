---
id: evaluate-covariate-bias-fractions
title: Register Juwara's removal fractions on E6's frame beside the controls
status: blocked
depends_on: generate-no-mapping-ablation, baselines-label-bootstrap, evaluate-prior-shift-comparator
blocks: evaluate-covariate-bias-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/covariate-bias-comparison
---

# Register Juwara's removal fractions on E6's frame beside the controls

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

[`docs/RESEARCH.md`](../docs/RESEARCH.md#published-comparisons) runs four protocols beside
published numbers. [Juwara et al.](https://doi.org/10.1016/j.patter.2024.100946) remove a growing
share of a minority group, fill it with synthetic records and read the group's AUC. The study
reads the same removal fractions on its own frame, apart from the control reading rule's
fractions. The released scripts are on OSF ([code](https://doi.org/10.17605/OSF.IO/RKF9T), MIT).
They call the non-public `sdgs` package, so they do not run as released, and the paper describes
the generator as a sequential tree-based synthesizer. Our sequential CART, `cart`, stands in.

## Scope

**In.**

- New named runs on E6's frame at removal fractions 0.15, 0.30, 0.50, 0.80 and 0.95. The names are
  new and no existing `RUNS` entry is edited. Juwara et al. use 15%, 30%, 50% and 80% on real data
  and 15% to 95% in simulation.
- Arms `bko-direct`, `cart`, `smote`, `ctgan`, `none` and `real`, and the controls `bko-nomap`,
  `label-bootstrap` and `prior-shift`.
- Each fill sized as Juwara's code sizes it: it restores the cell to about its count before
  removal, adding the total size less the minority count that remains. The paper states the rule
  only qualitatively; the size comes from the released code.
- A reading of AUC on the removed cell's test rows, beside the set's measures.
- The EVIDENCE Configuration, `not yet run`, quoting Table 2's rows at 15%, 50% and 80% (AUC,
  ORZ (SD), I Z) from the paper, with no value copied into this unit.
- A new ADR under `docs/adr/research/` that says what differs from Juwara et al.: a cell against
  one covariate level, one seeded fill against the paper's 100 averaged copies, and our CART port
  because their OSF script calls the non-public `sdgs` package.

**Out.**

- Their request-only datasets.
- Their breast-cancer data, which has no sex column.

## Acceptance

- A test pins the fractions and the arms.
- `make check` passes.
