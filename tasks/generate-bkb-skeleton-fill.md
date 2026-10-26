---
id: generate-bkb-skeleton-fill
title: BKB-sampled states, LLM-filled fields
status: blocked
depends_on: generate-bkb-sampler
blocks: --
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: --
---

# BKB-sampled states, LLM-filled fields

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

At level `ontology` the prompt lists the findings a gap rests on and leaves the model to decide
which records show them; `cleanup:generate/replica-marginals-drift` measures how far the result
drifts from the cell's field distributions. Conditional generation puts the BKB in charge of each
record's concept states and the model in charge of the fields consistent with them: the sampler of
[`generate-bkb-sampler`](generate-bkb-sampler.md) draws the states, and one call fills the records.

## Scope

**In.**

- A request form carrying, per asked record, the states the sampler drew, rendered in the user turn
  as one line per record after the components of the chosen level; ADR
  [`research/record-generation-design`](../docs/adr/research/record-generation-design.md) gains
  the component and its reason.
- `bko.validate` gains a `states` check: a decoded record whose mapped states differ from its
  prescribed ones fails it, counted in validity like any other check.
- An arm `bko-skeleton` in `E1`, no criterion reading it, run at `SETTINGS` on every holdout with
  its gap and replica fills. Spend estimate: one arm's share of `e1-3`'s spend plus the longer user
  turn's input tokens, stated before the run; the run is detached, kept awake and on mains power,
  and watched as `e1-3` was.
- A new E1 Runs row in `docs/EVIDENCE.md` with validity, drops, replica fidelity and utility, test
  copies, and each criterion's measures with no verdict, beside `bko` and `bko-direct`.

**Out.** Replacing `bko`'s prompt; a verdict on the arm.

## Acceptance

- The rendered user turn is pinned by a test on a fixture sample.
- The run is recorded in `docs/EVIDENCE.md` with its spend and replays byte for byte.
- `make check`.
