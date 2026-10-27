---
id: evaluate-frame-generic-pipeline
title: The stages read a frame, not the UCI constants
status: ready
depends_on: --
blocks: evaluate-synthea-downstream, generate-prompt-training-bounds, generate-effort-sweep
parallel_safe: false
owner: Joe
milestone: final-report
paid: false
adrs: --
---

# The stages read a frame, not the UCI constants

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

`bko.baselines`, `bko.evaluate`, `bko.generate` and `bko.validate` bind `bko.datasets.uci`'s
columns, codebook, ranges and feature list at import, and the baselines' output contract names
`FRAME_COLUMNS`. A second evaluation frame, the Synthea one of
[`evaluate-synthea-downstream`](evaluate-synthea-downstream.md), cannot run through them until
each stage reads those values from a frame it is given.

## Scope

**In.**

- A frame value in `bko.datasets` holding a dataset's columns, categorical codes, numeric ranges
  and precision, cell columns and label, with UCI as its first instance and its values unchanged.
- Every stage that binds a UCI constant takes the frame instead: the baselines' request and output
  contract, the classifier's features, fidelity, privacy and test copies, validation's checks, and
  the prompt's field list, codes, units and ranges.
- UCI's behaviour is unchanged: E1's judged scores regenerate byte for byte through
  `score --replay` at the landing commit, and every prompt test's rendered text is unchanged.
- `parallel_safe: false`: it touches a module of every stage.

**Out.** The Synthea frame itself; any change to what a UCI call sends.

## Acceptance

- The judged run's scores regenerate byte for byte through `score --replay` at the landing
  commit, and the gate's prompt and validation pins are unchanged.
- `make check`.
