---
id: slm-lora-trainer
title: Add the slm extra, a record serializer and a LoRA trainer for a fair arm comparison
status: blocked
depends_on: slm-lora-spike
blocks: slm-lora-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/slm-comparison-design
---

# Add the slm extra, a record serializer and a LoRA trainer for a fair arm comparison

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

A language-model comparison of the arms is fair only if every arm shares one serializer, one base
model and one training budget.

## Scope

**In.**

- The `slm` extra in the lock, at the versions unit [`slm-lora-spike`](slm-lora-spike.md)
  resolved. Under ADR [`tooling/uv-ruff-ty`](../docs/adr/tooling/uv-ruff-ty.md), adding it changes
  the extras list and the `make setup` line in [`AGENTS.md`](../AGENTS.md) and the environment
  section of [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#environment).
- The tooling ADR that unit [`slm-lora-spike`](slm-lora-spike.md) writes, revised in place.
- A serializer: fixed-order field descriptions from the `FrameSpec` (`bko.datasets.frames`), and
  a yes or no answer scored as a probability that feeds `utility()` (`bko.evaluate.utility`). One
  prompt template serves training and scoring; the training target is the answer token alone,
  over records of both labels; a test checks that a training prompt minus its answer equals the
  scoring prompt.
- A trainer, tested in the gate on a tiny randomly initialised model.
- Pinned base weights under `data/models/`, loaded only as safetensors and only after their
  sha256 verifies.
- The LoRA adapter, fitted on patient-derived records, kept under the gitignored `data/models/` or
  `runs/`, at a path the new ADR names.
- A new ADR under `docs/adr/research/` that fixes:
  - the training set per arm: the full training split, plus the arm's fill;
  - the controls: no fine-tune, `none` and `real`;
  - the LoRA settings;
  - the holdouts and seeds, chosen from unit [`slm-lora-spike`](slm-lora-spike.md)'s measured
    time;
  - the stated power limit;
  - which synthetic-share mixes of unit
    [`evaluate-synthetic-share-sweep`](evaluate-synthetic-share-sweep.md) the measured time allows;
  - whether two fine-tunes at one seed give equal adapters on this machine, measured, and what the
    design does if they do not.
- The ADR cites the one prior study of this design through
  [`docs/literature.md`](../docs/literature.md). No registry entry fits it today, so this unit
  adds the entry.
- The Configuration of a new [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry, with status
  `not yet run`.
- The [`docs/IDEAS.md`](../docs/IDEAS.md#slm-distillation-stretch-goal) entry "SLM distillation
  stretch goal" narrowed to what this design leaves out.

**Out.**

- Network use in the gate.
- The run.

## Acceptance

- Serialization is byte-stable against a golden string.
- The trainer test uses a randomly initialised model built in the test, so it needs no network
  fixture; the session guard in `tests/conftest.py` already refuses API calls.
- `make check` passes.
