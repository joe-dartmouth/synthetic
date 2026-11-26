---
id: slm-lora-results
title: Run the LoRA comparison of the arms on E6's frame
status: blocked
depends_on: evaluate-composite-dev-results, slm-lora-day-budget
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the LoRA comparison of the arms on E6's frame

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Whether the differences between generators survive in a fine-tuned language model is unknown.

## Scope

**In.**

- The run on E6's frame
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)), under the design
  of ADR [`research/slm-comparison-design`](../docs/adr/research/slm-comparison-design.md), in
  the environment ADR [`tooling/slm-extra`](../docs/adr/tooling/slm-extra.md) sets, with these
  arms:
  - the confirmatory candidate that unit
    [`evaluate-composite-dev-results`](evaluate-composite-dev-results.md) names;
  - `bko-direct`, `label-bootstrap`, `ctgan`, `tabddpm` and `smote`;
  - the controls E12 registers.
- Each arm's fills drawn on E6's frame by this unit, including `label-bootstrap`'s (ADR
  [`research/baseline-augmentation-design`](../docs/adr/research/baseline-augmentation-design.md))
  and `tabddpm`'s (ADR [`research/diffusion-baseline`](../docs/adr/research/diffusion-baseline.md)).
- Each test record's probability, in a sha-verified file.
- The reading beside the tabular classifiers' ranking.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Runs, Result and Read against.

**Out.**

- A claim that the language model beats the tabular classifiers.

## Acceptance

- The entry is `complete`, with the controls and the power limit stated.
- `make check` passes.
