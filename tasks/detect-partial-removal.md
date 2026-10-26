---
id: detect-partial-removal
title: Partial removal in the Synthea ablation
status: ready
depends_on: --
blocks: detect-graded-grounding
parallel_safe: true
owner: Partiban
milestone: final-report
paid: false
adrs: --
---

# Partial removal in the Synthea ablation

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

E2 removes every case of a cell or of the population, so it measures recovery of a whole absence
and no sensitivity or specificity for any detector
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e2-synthea-gap-detection-ablation), Limits). Per-subgroup
and population-level grounding are blind to partial removal by construction, since one remaining
record grounds a statement (ADR
[`research/synthea-ablation-design`](../docs/adr/research/synthea-ablation-design.md)); the
frequency-aware score and the count-based detector can see it, and how much of a cell must go
before each flags it is the number the status update's next step asks for.

## Scope

**In.**

- `AblationSpec` gains a removed fraction: the cases the spec names are ordered by a seeded draw and
  the first fraction of them removed whole, as today's removal does, with the fraction 1 equal to
  today's ablation.
- `E2_ABLATIONS` gains the `cell` and `condition` ablations at fractions 0.25, 0.5, 0.75 and 1.
- The reading, per fraction and detector: the truth is the cells the removal touched (for the
  count-based detector, those whose count fell below the threshold; for the three definitions, the
  (cell, concept) pairs of the touched cells); sensitivity is the share of the truth flagged after
  the removal and not before, and specificity the share of the untouched cells not newly flagged.
  Both grounding definitions are expected at zero sensitivity below fraction 1, and the record
  states that as the definitions' consequence, not a defect.
- ADR `research/synthea-ablation-design` edited as its reversal condition says, and the E2 entry in
  `docs/EVIDENCE.md` gaining the configuration, a Runs row and the Result at each fraction.

**Out.** Changing a definition's rule (that is
[`detect-graded-grounding`](detect-graded-grounding.md)); a new mapped concept (that is
[`detect-synthea-mappings`](detect-synthea-mappings.md)).

## Acceptance

- `uv run --no-sync python -m bko.gaps synthea` prints every fraction's reading, identical on a
  second run, and a test pins the fraction-1 reading to today's.
- E2's entry records the sweep.
- `make check`.
