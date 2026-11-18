---
id: baselines-label-bootstrap
title: Add the label-bootstrap baseline, whole label-group records drawn into the cell
status: blocked
depends_on: plan-direct-method-research
blocks: evaluate-control-runs-registered, slm-lora-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Add the label-bootstrap baseline, whole label-group records drawn into the cell

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Copying whole label-group records keeps every within-record correlation. It is the null a
reviewer names first against `bko-direct`, which pools field values from the same label group.
The control reading rule in [`docs/RESEARCH.md`](../docs/RESEARCH.md) reads it.

## Scope

**In.**

- `Method.LABEL_BOOTSTRAP` (`label-bootstrap`) in `bko.baselines.request`. Each requested row is
  one training record of the cell's label whose age is in the cell's band, drawn uniformly with
  replacement, with its sex set to the cell's.
- A stratum with no such record raises `BaselineError` (`bko.baselines.errors`).
- ADR
  [`research/baseline-augmentation-design`](../docs/adr/research/baseline-augmentation-design.md)
  revised in place to hold the method.

**Out.**

- Joining `METHOD_ARMS` (`bko.evaluate.experiment`): E1 and E3 keep their arms.

## Acceptance

- A fixture fill returns only rows equal to a label-group record except for sex.
- The same seed gives equal rows twice.
- `make check` passes.
