---
id: detect-graded-grounding
title: Graded grounding and the frequency-aware level
status: ready
depends_on: --
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
E2's fraction ablations (`e2-7`) supply the truth that makes varying them measurable.

## Scope

**In.**

- A grounding count `k` for the per-subgroup and population-level definitions, a statement
  counting as grounded only when at least `k` records of the fragment show its concept, through
  `bko.gaps.detect`'s held set; and the frequency-aware level as a parameter of `bko.gaps.gaps`;
  `k` of 1 and level 0.95 are the rules E2 ran.
- A sweep over `k` in 1, 2, 3 and 5 and over the level in 0.90, 0.95 and 0.99, read against the
  partial-removal truth at every fraction of both ablations, and listed on the UCI default
  training split; per setting and detector, sensitivity and specificity as
  E2 defines them (`bko.gaps.experiment.detection`).
- The rule that picks the reported default, stated here before the sweep runs: for the
  frequency-aware definition on the `cell` ablation, the level with the highest specificity among
  those whose sensitivity at fraction 0.5 is at least the count-based detector's, the lower level
  on a tie; for the grounding definitions, `k` is reported at every value and E1's judged arm
  keeps `k` of 1, since a count above 1 departs from the ungrounded-statement definition of
  Santos et al.'s Proposition 5.1.1 (ADR
  [`research/bko-formal-gap-definition`](../docs/adr/research/bko-formal-gap-definition.md)).
- A new `docs/EVIDENCE.md` entry, since ADR
  [`research/frequency-aware-gap-score`](../docs/adr/research/frequency-aware-gap-score.md) makes
  a changed rule after E2's run a new experiment entry; that ADR and
  `research/bko-formal-gap-definition` record the parameters and that `bko-frequency` reads the
  chosen level only once [`docs/RESEARCH.md`](../docs/RESEARCH.md#detect) says so.

**Out.** Rerunning E1 under a new rule; a Dirichlet score in place of the coverage-style one.

## Acceptance

- The sweep prints from `uv run --no-sync python -m bko.gaps synthea` and `bko.gaps uci`,
  identical on a second run, and a test on a `tmp_path` population pins that `k` of 1 and level
  0.95 list the same gaps as before the parameters existed.
- The chosen level is recorded with the measure that chose it.
- `make check`.
