---
id: slm-lora-spike
title: Resolve a LoRA stack in a scratch lock and time one fine-tune on this machine
status: ready
depends_on: --
blocks: slm-lora-trainer
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: tooling/slm-extra
---

# Resolve a LoRA stack in a scratch lock and time one fine-tune on this machine

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The lock holds no transformers or peft. Their package metadata shows no bound that blocks this
interpreter, but no resolver has run and no fine-tune has been timed on this machine.

## Scope

**In.**

- A scratch lock that resolves transformers, peft, tokenizers, safetensors and accelerate,
  through the try-an-extra recipe of
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#environment).
- One measured LoRA fine-tune, run from `scratchpad/`, of one small openly licensed model with
  documented training data, on the training set of one E6 holdout
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)).
- A new ADR under `docs/adr/tooling/` that records the resolved versions, the base model and the
  measured time.

**Out.**

- Any change to `pyproject.toml`, `uv.lock` or `src/`.

## Acceptance

- The ADR records the resolved versions and the measured wall time.
- `make check` passes.
