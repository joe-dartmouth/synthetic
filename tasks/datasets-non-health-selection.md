---
id: datasets-non-health-selection
title: Rank the non-health candidate datasets and choose the first frames
status: blocked
depends_on: datasets-selection-rule
blocks: datasets-generalised-cell-axes
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/non-health-dataset-selection
---

# Rank the non-health candidate datasets and choose the first frames

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The study's only non-health family is ACS PUMS. The
[IDEAS entry](../docs/IDEAS.md#candidate-datasets-and-ontologies-outside-healthcare) lists
candidates outside healthcare and marks what it could not verify. A dataset whose unit is not a
person needs the cell axes generalised, which unit
[`datasets-selection-rule`](datasets-selection-rule.md) excludes. This unit is the later
evaluation of those candidates.

## Scope

**In.**

- Re-read from each primary page what the entry marks UNVERIFIED: FARS and CRSS terms and code
  hierarchy, SBA status codes, NIBRS's public fields, USAspending and TED row counts, and the
  VERIS disclosure field.
- The rule a non-health frame must meet: a binary outcome, two named cell axes, an open
  machine-readable hierarchy, and terms that permit the use.
- Every candidate ranked, with its exclusion reason.
- The first frames chosen, each as an exploratory family outside the health multiplicity family.
- [`docs/RESEARCH.md`](../docs/RESEARCH.md) changed to list them.
- A frame unit added to the board for each chosen dataset.
- The IDEAS entry narrowed to what is not chosen.
- A new ADR under `docs/adr/research/` that records the rule and the ranking.

**Out.**

- Loading any candidate's records.

## Acceptance

- The ADR ranks every candidate with terms quoted from the primary page.
- `make check` passes.
