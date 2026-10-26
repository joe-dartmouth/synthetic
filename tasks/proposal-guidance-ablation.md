---
id: proposal-guidance-ablation
title: Guidance-level ablation of record generation
status: ready
depends_on: --
blocks: evaluate-e1-second-configuration, evaluate-generator-comparison
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
- Run each level in E1's judged configuration, Haiku 5.5 settings and BKO replicas included, so
  the level is the only difference from E1's judged run; level `ontology` is that run itself.
  Report each level's validity, dropped candidates, spend, and E1's measures beside it.
- Read `cell` and `coherent` as the no-ontology controls: `cell` sends the cell, label, age band,
  numeric ranges and record count and nothing else, `coherent` adds the coherence rules, and
  neither sends a finding, so `cell` is the LLM with no BKO guidance.
- Report each level's replica fidelity (quality, column shapes, column pair trends) and replica
  utility beside its validity, since `cleanup:generate/replica-marginals-drift` is discharged by
  the level whose replica sits closest to the training rows, and
  [`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md) sets `SETTINGS` to it.
- Spend estimate before the run: one arm's share of `e1-3`'s spend and live minutes per holdout,
  per level ([`experiments-index.md`](experiments-index.md#spend)).
- Record the runs in a new `docs/EVIDENCE.md` entry.

**Out.** Changing E1's configuration.

## Acceptance

- Each run's spend and log location are in its `docs/EVIDENCE.md` Runs field.
- The levels' measures come from one committed command.
- `make check`.
