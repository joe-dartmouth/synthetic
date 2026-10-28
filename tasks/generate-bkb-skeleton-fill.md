---
id: generate-bkb-skeleton-fill
title: BKB-drawn concepts, LLM-filled fields
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: research/record-generation-design, research/e1-evaluation-protocol, research/bkb-direct-generation
---

# BKB-drawn concepts, LLM-filled fields

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

At every guidance level the prompt leaves the model to decide which records show the findings it
names. At `e1-3` (level `ontology`), trained on its replica alone, `bko` scored the lowest fidelity
quality of every arm, 0.7766 against `tvae`'s 0.8039
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation), Read against).
At `e1-4` (`summary`, with the training split's bounds in the prompt) that premise no longer holds:
`bko`'s replica quality 0.8194, column shapes 0.8918 and column pair trends 0.7470 are above
`tvae`'s 0.8039, 0.8654 and 0.7423, though its replica quality stays below `smote`'s, `adasyn`'s and
`ctgan`'s and its replica recall, 0.6884, below every baseline's but `ctgan`'s. The unit's question
is whether concepts drawn from the BKB beat the in-run `bko` on replica fidelity and utility.
Conditional generation puts the BKB in charge of each record's concepts and the model in charge of
the fields consistent with them: the sampler `bko.generate.direct` (ADR `research/bkb-direct-generation`)
draws each record's present concepts, and one call fills the records.

## Scope

**In.**

- A request form carrying, per asked record, the concepts the sampler drew present, rendered in
  the user turn as one line per record after the components of the chosen level; ADR
  [`research/record-generation-design`](../docs/adr/research/record-generation-design.md) gains
  the component and its reason.
- A drop reason beside off-cell in `bko.baselines.records.drops`: a decoded record whose mapped
  concepts match no line of its call left unconsumed is dropped and topped up, counted like an
  off-cell drop; matching is by concept set within the call's own cell (ADR
  [`research/e1-evaluation-protocol`](../docs/adr/research/e1-evaluation-protocol.md));
  `bko.validate` is unchanged, since it reads no ontology relationship (ADR
  [`research/record-validation-design`](../docs/adr/research/record-validation-design.md)).
- A named run (`bko.evaluate.experiment.RUNS`) with the baselines, an arm `bko-skeleton` and an
  in-run `bko` control, both at `SETTINGS` (level `summary`), on E1's holdouts with their gap and
  replica fills, under the named run's ceiling and the paid-run protocol (`docs/DEVELOPMENT.md`,
  Evaluation). Spend estimate: one arm's share of the judged run's spend per Messages arm plus the
  longer user turn's input tokens and a top-up allowance, stated in the Runs row before the run.
- A new `docs/EVIDENCE.md` entry with validity, drops, replica fidelity and utility, test copies,
  and each criterion's measures with no verdict, read beside the judged run's `bko` and the
  sampler's `bko-direct`.

**Out.** Replacing `bko`'s prompt; a verdict on the arm.

## Acceptance

- The rendered user turn is pinned by a test on a fixture draw, and a test pins the new drop
  reason on a record whose concepts match no line of its call left unconsumed.
- The run is recorded in `docs/EVIDENCE.md` with its spend and replays byte for byte.
- `make check`.
