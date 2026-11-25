---
id: slm-lora-day-budget
title: Fit the LoRA comparison of the arms into one day of machine time
status: ready
depends_on: --
blocks: slm-lora-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Fit the LoRA comparison of the arms into one day of machine time

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

ADR [`tooling/slm-extra`](../docs/adr/tooling/slm-extra.md) measured a full-split fine-tune at 17
to 28 hours on this machine, so the design of ADR
[`research/slm-comparison-design`](../docs/adr/research/slm-comparison-design.md) costs days per
holdout. The comparison must run in under one day.

## Scope

**In.**

- Measure the levers on this machine: a smaller base model, a subsampled training set, fewer
  steps, fewer arms and shorter prompts.
- Choose a design whose every fine-tune and scoring pass fits one day of machine time in total.
- Revise ADR [`research/slm-comparison-design`](../docs/adr/research/slm-comparison-design.md)
  and ADR [`tooling/slm-extra`](../docs/adr/tooling/slm-extra.md) in place with the holdouts, the
  seeds, the power limit and the share mixes of
  [RESEARCH section 11](../docs/RESEARCH.md#synthetic-share-ablation) the measured time allows.

**Out.**

- The run: unit [`slm-lora-results`](slm-lora-results.md).

## Acceptance

- The ADR states the design and the measured time behind it.
- `make check` passes.
