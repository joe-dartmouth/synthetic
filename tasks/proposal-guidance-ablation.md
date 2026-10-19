---
id: proposal-guidance-ablation
title: Guidance-level ablation of record generation
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: --
---

# Guidance-level ablation of record generation

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

The team asks how much guidance and ontology structure record generation needs. `bko.generate`
builds six nested guidance levels, and this unit measures what each one adds.

## Scope

**In.**

- Run the candidate arm `bko` at every level of `bko.generate.messages.Level` on E1's holdouts.
- Report each level's validity, dropped candidates, spend, and E1's measures beside E1's judged
  run.
- Record the runs in a new `docs/EVIDENCE.md` entry.

**Out.** Changing E1's judged configuration.

## Acceptance

- Each run's spend and log location are in its `docs/EVIDENCE.md` Runs field.
- The levels' measures come from one committed command.
- `make check`.
