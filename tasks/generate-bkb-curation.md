---
id: generate-bkb-curation
title: Curate LLM candidates by their BKB joint mass
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: research/bkb-candidate-curation
---

# Curate LLM candidates by their BKB joint mass

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

The BKO guides generation only through the prompt. CLLM
([`docs/IDEAS.md`](../docs/IDEAS.md#cllm-and-great-as-alternative-generators)) curates generated
rows by a classifier's confidence; the fused BKB offers a knowledge likelihood instead, the joint
mass of the S-nodes a record's mapped states ground. Whether that mass separates records that
match the real cell's distribution from records that do not can be measured for free over
`e1-3`'s call log, before any paid arm is built on it.

## Scope

**In.**

- `bko.validate` or `bko.fusion` gains a score per candidate: the joint mass under the fused BKB
  (ontology fragments plus the cell's and population's dataset fragments) of the S-nodes that
  `bko.bkb.inference.instantiate` reaches from the record's mapped states, through
  `bko.bkb.inference.probability`. A new ADR under `docs/adr/research/` fixes how a state no S-node
  supports and a component the record does not map are read, and whether the mass is normalized
  across the cell's candidates.
- Phase 1, free: one committed command scores every candidate `e1-3`'s call log decodes, and
  reports per cell the score's spread, the share of records at the minimum, and the replica
  fidelity (quality, column shapes, column pair trends against the training rows) of the top half
  by score against the bottom half, with the candidates read from the log as
  `generate-bkb-sampler`'s record reads them.
- Phase 2, paid, only if phase 1 shows the top half closer to the training rows than the bottom
  half on every fidelity measure: an arm `bko-curated` that asks each cell for a multiple of its
  quota at `SETTINGS` and keeps the quota's highest-scoring records, scored on E1's holdouts beside
  the other arms. Spend estimate: the multiple times one arm's share of `e1-3`'s spend, stated in
  the Runs row before the run. If phase 1 shows no separation, the record says so and phase 2 does
  not run.
- A new `docs/EVIDENCE.md` entry holding phase 1's measures and, when it runs, phase 2's run with
  its spend.

**Out.** A verdict on the arm; changing what `bko` sends.

## Acceptance

- Phase 1's measures come from one committed command over `e1-3`'s call log, no API call.
- Phase 2 is recorded with its spend, or the entry states the phase 1 measure that stopped it.
- `make check`.
