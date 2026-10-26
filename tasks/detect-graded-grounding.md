---
id: detect-graded-grounding
title: Graded grounding and the frequency-aware level
status: blocked
depends_on: detect-partial-removal
blocks: --
parallel_safe: true
owner: Partiban
milestone: final-report
paid: false
adrs: --
---

# Graded grounding and the frequency-aware level

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

A statement is grounded by one record, so the two grounding definitions flag only complete absence,
and on UCI most of what they flag sits in the 2-record cell that counts already flag
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e2-synthea-gap-detection-ablation), the `uci` reading).
The frequency-aware rule flags at posterior mass 0.95, and two UCI flags sat within 0.003 of that
level. Neither the grounding count nor the level has been varied, and
[`detect-partial-removal`](detect-partial-removal.md) supplies the truth that makes varying them
measurable.

## Scope

**In.**

- A grounding count `k` for the per-subgroup and population-level definitions: a statement counts
  as grounded only when at least `k` records of the fragment show its concept, `k` of 1 being
  today's rule; and the frequency-aware level as a parameter, 0.95 being today's.
- A sweep over `k` in 1, 2, 3 and 5 and over the level in 0.90, 0.95 and 0.99, read against the
  partial-removal truth at every fraction, and listed on the UCI default training split; per
  setting, each detector's sensitivity and specificity.
- The rule that picks a default, stated before the sweep runs: the setting with the highest
  specificity among those whose sensitivity at fraction 0.5 is at least the count-based
  detector's, or today's values when none qualifies.
- ADRs [`research/bko-formal-gap-definition`](../docs/adr/research/bko-formal-gap-definition.md)
  and [`research/frequency-aware-gap-score`](../docs/adr/research/frequency-aware-gap-score.md)
  edited: a `k` above 1 departs from the ungrounded-statement definition of Santos et al.'s
  Proposition 5.1.1, so it runs as an ablation beside the formal definition and E1's judged arm
  keeps `k` of 1 unless [`docs/RESEARCH.md`](../docs/RESEARCH.md#detect) changes first.
- E2's entry gaining the sweep.

**Out.** Rerunning E1 under a new default; a Dirichlet score in place of the coverage-style one.

## Acceptance

- The sweep prints from one committed command, identical on a second run, with a test pinning the
  `k` of 1 and level 0.95 setting to today's gap lists.
- The chosen default is recorded with the measure that chose it.
- `make check`.
