---
id: detect-partial-removal
title: Partial removal in the Synthea ablation
status: ready
depends_on: --
blocks: detect-graded-grounding, detect-synthea-mappings
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
before each flags it is unmeasured.

## Scope

**In.**

- `AblationSpec` gains a removed fraction: the cases the spec names are ordered by a seeded draw and
  the first fraction of them removed whole, as today's removal does; fraction 1 is the existing
  `cell` and `condition` ablations.
- `E2_ABLATIONS` gains `cell` and `condition` at fractions 0.25, 0.5 and 0.75.
- The reading keeps E2's truth, the (cell, statement) pairs the removed patients ground and the
  cells whose count fell, and its recovered, outside, new-outside and vanished counts, and adds
  per fraction and detector: sensitivity, the share of eligible truth pairs (or cells) flagged
  after the removal and not before, and specificity, the share of pairs (or cells) outside the
  truth not newly flagged. Both grounding definitions read zero sensitivity below fraction 1, and
  the record states that as the definitions' consequence.
- ADR `research/synthea-ablation-design` edited as its reversal condition says, and the E2 entry in
  `docs/EVIDENCE.md` gaining the fractions in its Configuration, a Runs row and the Result at each
  fraction.

**Out.** Changing a definition's rule (that is
[`detect-graded-grounding`](detect-graded-grounding.md)); a new mapped concept (that is
[`detect-synthea-mappings`](detect-synthea-mappings.md)).

## Acceptance

- `uv run --no-sync python -m bko.gaps synthea` prints every fraction's reading, identical on a
  second run over the pinned population, and the fraction-1 rows equal `e2-6`'s `recovery` lines
  byte for byte.
- A test on a `tmp_path` population pins the fraction draw and the sensitivity and specificity
  arithmetic.
- `make check`.
