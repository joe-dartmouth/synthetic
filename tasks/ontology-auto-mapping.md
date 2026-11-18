---
id: ontology-auto-mapping
title: Build mapping rows automatically from a codebook and a pinned release, with no paid model
status: blocked
depends_on: generate-ancestor-shrunk-shares
blocks: ontology-auto-mapping-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/automatic-mapping
---

# Build mapping rows automatically from a codebook and a pinned release, with no paid model

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The mapping table is hand-written. NVIDIA auto-ontology does not replace it. Read at its source
on 2026-11-18, it makes one self-named term per table, its term prompt rules out `is_a`
relations, and it links to no standard ontology.

## Scope

**In.**

- Two builders of `FieldMapping` rows (`bko.ontology.schema`) from a frame's codebook and a
  pinned release. Neither calls a paid API.
  - `auto-lexical` uses no model. A coded value maps through the release's cross-references to
    its code system. A flag or level maps to the term whose label or synonym best matches its
    codebook description above a fixed score. Numeric columns stay unmapped.
  - `auto-local` judges the same candidates with a pinned on-device model through
    `bko.generate.local`. It may also propose a cutoff row for a numeric column, with its cited
    source.
- Each builder emits a pinned artifact with a sha256, and its rows pass the structural check
  `bko.ontology.check.check` against the pinned releases.
- `RUNS` entries on E6's frame and on `diabetes-codes` (unit
  [`ontology-diabetes-diagnosis-codes`](ontology-diabetes-diagnosis-codes.md)) for `bko-direct`
  and `bko-tree` (unit [`generate-ancestor-shrunk-shares`](generate-ancestor-shrunk-shares.md))
  over the hand mapping, over each automatic mapping and over a permuted-hierarchy control,
  beside `bko-nomap`.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Configuration with status `not yet run`,
  with the margin for automatic minus hand.
- A new ADR under `docs/adr/research/` that records the route, the NVIDIA finding and the leakage
  rule: a mapping is built from the codebook, never from records.

**Out.**

- Paid models.
- Running NVIDIA auto-ontology.
- Confirmatory datasets.

## Acceptance

- On UCI Heart Disease, `auto-lexical`'s rows are byte-equal across two runs.
- A test computes each builder's precision and recall against the hand table on a fixture
  codebook.
- `make check` passes.
