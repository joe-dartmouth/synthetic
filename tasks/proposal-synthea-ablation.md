---
id: proposal-synthea-ablation
title: Synthea population and gap-detection ablation
status: ready
depends_on: --
blocks: proposal-gap-detection
parallel_safe: true
owner: Joe
milestone: status-update-1
paid: false
adrs: research/synthea-ablation-design
---

# Synthea population and gap-detection ablation

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

E2 tests gap detection on a Synthea population with selected cases removed, so the removed cases
are a known answer ([RESEARCH section 2](../docs/RESEARCH.md#2-intended-users-and-evaluation-setting)).
Only gap detection reads Synthea, so it is a unit apart from the UCI loading the baselines need.

## Scope

**In.**

- Generate a Synthea population with the pinned v4.0.0 jar under `data/`, every parameter that
  changes the output pinned: seeds, reference and end dates, population size, state, history
  window, thread count and time zone.
- Load its patients and conditions into `src/bko/datasets/`, shaped so `bko.datasets.cells` counts
  them.
- An ablation that removes every patient with a chosen condition code, in one age band and sex or
  across the whole population, and returns the removed patients as ground truth with the cell
  counts before and after.

**Out.** E2's choice of targets, which E2's configuration fixes; Synthea tables other than
patients and conditions.

## Acceptance

- Generation, loading and the ablation run from one command; a second run reuses the population.
- A new ADR under `docs/adr/research/` records the generation parameters and the ablation design.
- `make check`.
