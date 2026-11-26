---
id: ontology-auto-mapping-results
title: Run the automatic mappings against the hand mapping on the development frames
status: blocked
depends_on: ontology-auto-mapping
blocks: paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the automatic mappings against the hand mapping on the development frames

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

An automatic BKO matters only if its rows are as useful as the hand-built one's.

## Scope

**In.**

- The runs registered by unit [`ontology-auto-mapping`](ontology-auto-mapping.md).
- Each mapping's precision and recall against the hand tables.
- Automatic minus `bko-nomap`, and automatic minus hand under the registered margin.
- The permuted-hierarchy control.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Runs, Result and Read against.

**Out.**

- Claims beyond the development frames.

## Acceptance

- The entry is `complete`.
- `make check` passes.
