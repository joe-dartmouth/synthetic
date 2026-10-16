---
id: proposal-evaluation-data
title: Grow the evaluation data
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: research/evaluation-data
---

# Grow the evaluation data

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

E1's holdouts test 59 records with 27 positives, and its underrepresented cells hold 5 positive
test records, so most criteria resolve only large effects (ADR `research/e1-evaluation-protocol`).
More records tighten every interval.

## Scope

**In.**

- Choose the source of more records, recorded in a new ADR under `docs/adr/research/`. One option
  is the other three sites of the UCI archive, which reverses ADR `research/uci-evaluation-frame`
  (Cleveland only; pooling the four sites is ruled out there): measured 2026-10-15, the four sites
  hold 740 complete records once `slope`, `ca` and `thal` are dropped, and 661 once records with a
  cholesterol of 0 (the archive's code for a missing value) are dropped too. The other is another
  open dataset.
- Load it in `bko.datasets` and carry any schema change through `bko.baselines` and
  `bko.evaluate`.
- Re-check the underrepresented cells on the new training split: if no cell falls under the
  threshold, the age bands are re-chosen as ADR `research/uci-evaluation-frame`'s reversal
  conditions say, since E1's gap request needs a cell to fill.
- Re-fix `E1`'s holdout count and margins and ADR `research/e1-evaluation-protocol` together,
  before E1's first run.

**Out.** Running E1 on it, which is `proposal-final-evaluation`.

## Acceptance

- `python -m bko.datasets cells` and `python -m bko.evaluate score` run on the new frame.
- `make check`.
