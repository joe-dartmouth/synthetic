---
id: paper-control-results
title: Read the matched development contrasts
status: blocked
depends_on: evaluate-removed-truth-reading, paper-controls
blocks: evaluate-confirmatory-registration, paper-baselines, paper-novelty
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Read the matched development contrasts

The set's graph is [direct-index.md](direct-index.md).

## Why this unit exists

The matched contrast determines which claim merits a larger study.

## Scope

Run the registered controls on development holdouts. Report paired differences and uncertainty
against each empirical and statistical control, plus sample sizes, failed fills and provenance.
Relate findings to E9 and the existing
[RESEARCH section 11](../docs/RESEARCH.md#control-reading-rule), the E12/E13 reading rule.

**Findings from planning.**

- The runs are E19's families, which unit [`paper-controls`](paper-controls.md) registers:
  `e19-thin`, `e19-empty`, `e19-supported-thin` and `e19-supported-empty`, each sliced in ten.
- It reads the primary and secondary contrasts as unit [`paper-controls`](paper-controls.md)
  fixes them (and E19's Configuration once written), and no K contrast at full removal (ADR
  [`research/no-mapping-ablation`](../docs/adr/research/no-mapping-ablation.md),
  Consequences).
- It first reads `real` minus `none` on the supported runs; where that contrast is below the
  margin, the supported runs' contrasts are descriptive.
- Once E12 has run (unit
  [`evaluate-diabetes-control-results`](evaluate-diabetes-control-results.md)), it checks that
  E19's `bko-direct`, `none` and `real` scores equal E12's per holdout.
- It reads E6's removed-truth reading, over the E6 `rows` stores unit
  [`evaluate-removed-truth-reading`](evaluate-removed-truth-reading.md) reads, beside the
  downstream contrasts.
- The decision is recorded in E19's Read against in [`docs/EVIDENCE.md`](../docs/EVIDENCE.md).

## Acceptance

Record evidence for benefit, equivalence, harm or unresolved uncertainty. If empirical controls
explain the gains, carry conditional sampling and its limits forward as the paper question.
Record the decision before expanding runs.
