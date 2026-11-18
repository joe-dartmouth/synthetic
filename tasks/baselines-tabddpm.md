---
id: baselines-tabddpm
title: Add an in-repo TabDDPM baseline on the locked torch
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration, slm-lora-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/diffusion-baseline
---

# Add an in-repo TabDDPM baseline on the locked torch

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Reviewers expect a diffusion synthesizer among the comparators, and synthcity does not resolve
here (ADR
[`tooling/python-314-without-synthcity`](../docs/adr/tooling/python-314-without-synthcity.md)).
[`docs/IDEAS.md`](../docs/IDEAS.md#diffusion-generators-tabddpm-and-tabsyn) holds the idea and its
sources.

## Scope

**In.**

- `Method.TABDDPM` (`bko.baselines.request`), implemented in a new `bko.baselines.diffusion` on
  the locked torch:
  - Gaussian diffusion on quantile-transformed numerics;
  - multinomial diffusion per coded field;
  - an MLP denoiser;
  - label conditioning;
  - rows drawn and kept by their own values through `fill()`, as CTGAN's are
    (`bko.baselines.synthesizers`).
- A new ADR under `docs/adr/research/` that pins the authors' published configuration and fixes
  the training steps once, by validation loss on holdout 0's `none` training split, as ADR
  [`research/classifier-panel`](../docs/adr/research/classifier-panel.md) fixes the perceptron's
  epochs.
- The IDEAS entry "Diffusion generators: TabDDPM and TabSyn" narrowed to TabSyn.

**Out.**

- Per-dataset search.
- A cell-conditional variant.
- TabSyn.
- New dependencies.

## Acceptance

- The same seed gives byte-equal rows twice at one thread.
- A tiny fixture fit fills a two-stratum request.
- A test pins the ADR's constants.
- `make check` passes.
