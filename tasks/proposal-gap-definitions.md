---
id: proposal-gap-definitions
title: Per-subgroup and frequency-aware gap detection
status: ready
depends_on: --
blocks: proposal-fusion-repair
parallel_safe: true
owner: Partiban
milestone: status-update-2
paid: false
adrs: research/bko-formal-gap-definition, research/frequency-aware-gap-score, research/dataset-fragment-encoding
---

# Per-subgroup and frequency-aware gap detection

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

ADR `research/bko-formal-gap-definition` reads gaps per cell three ways, per-subgroup grounding
primary, and `bko.gaps` builds only the population-level one. Without the other two, UCI Heart
Disease has no BKO gap for generation to target or for the completeness criterion to count
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md), E2, Result).

## Scope

**In.**

- The per-subgroup definition, one dataset fragment per cell (ADR
  `research/dataset-fragment-encoding`), and the frequency-aware one (ADR
  `research/frequency-aware-gap-score`), in `bko.gaps` beside the population-level one, each giving
  (cell, statement) gaps.
- `python -m bko.gaps uci` and `python -m bko.gaps synthea` report each definition beside the
  count-based list.
- E2 run again, each definition scored as ADR `research/bko-formal-gap-definition` states.
- The tests that pin the two new definitions named in the Enforcement of those three records, in
  place of the bullets that name this unit.

**Out.** Generation, fusion and E1.

## Acceptance

- `docs/EVIDENCE.md` (E2) states its reading over (cell, statement) pairs and records, for each
  definition under both ablations, its recovered gaps and its gaps outside the truth, from one
  committed command.
- `bko.gaps` returns a dataset's (cell, statement) gaps under a named definition, the input
  `proposal-llm-generation` takes.
- `make check`.
