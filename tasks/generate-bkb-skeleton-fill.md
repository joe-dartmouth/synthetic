---
id: generate-bkb-skeleton-fill
title: BKB-drawn concepts, LLM-filled fields
status: blocked
depends_on: generate-bkb-sampler
blocks: --
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: --
---

# BKB-drawn concepts, LLM-filled fields

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

At level `ontology` the prompt lists the findings a gap rests on and leaves the model to decide
which records show them, and E1's Result reads the drift that follows: trained on its replica
alone, `bko` scores the lowest fidelity quality of every arm
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation)). Conditional
generation puts the BKB in charge of each record's concepts and the model in charge of the fields
consistent with them: the sampler of [`generate-bkb-sampler`](generate-bkb-sampler.md) draws each
record's present concepts, and one call fills the records.

## Scope

**In.**

- A request form carrying, per asked record, the concepts the sampler drew present, rendered in
  the user turn as one line per record after the components of the chosen level; ADR
  [`research/record-generation-design`](../docs/adr/research/record-generation-design.md) gains
  the component and its reason.
- A drop reason beside off-cell in `bko.baselines.records.drops`: a decoded record whose mapped
  concepts differ from its prescribed ones is dropped and topped up, counted like an off-cell
  drop; `bko.validate` is unchanged, since it reads no ontology relationship (ADR
  [`research/record-validation-design`](../docs/adr/research/record-validation-design.md)).
- A named run (`bko.evaluate.experiment.RUNS`) with the baselines and an arm `bko-skeleton` at `SETTINGS`
  on E1's holdouts with its gap and replica fills, under the named run's ceiling and the paid-run protocol
  (`docs/DEVELOPMENT.md`, Evaluation). Spend estimate: one arm's share of the judged run's spend plus the
  longer user turn's input tokens, stated in the Runs row before the run.
- A new `docs/EVIDENCE.md` entry with validity, drops, replica fidelity and utility, test copies,
  and each criterion's measures with no verdict, read beside the judged run's `bko` and the
  sampler's `bko-direct`.

**Out.** Replacing `bko`'s prompt; a verdict on the arm.

## Acceptance

- The rendered user turn is pinned by a test on a fixture draw, and a test pins the new drop
  reason on a record whose concepts differ from its line.
- The run is recorded in `docs/EVIDENCE.md` with its spend and replays byte for byte.
- `make check`.
