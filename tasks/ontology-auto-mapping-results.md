---
id: ontology-auto-mapping-results
title: Run the automatic mappings against the hand mapping on the development frames
status: ready
depends_on: --
blocks: paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/automatic-mapping
---

# Run the automatic mappings against the hand mapping on the development frames

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

An automatic BKO matters only if its rows are as useful as the hand-built one's.

ADR [`research/automatic-mapping`](../docs/adr/research/automatic-mapping.md) records the
builders, the measures and the permuted control, and the
[E22 entry](../docs/EVIDENCE.md#e22-automatic-mappings-on-the-diabetes-130-removal) of
`docs/EVIDENCE.md` holds the Configuration. `RUNS` holds E22's sixteen runs
(`bko.evaluate.runs_direct.E22` and `E22_FAMILIES`), each `<family>-thin` or `<family>-empty`
with slices `<run>a` to `<run>j`:

| Family | Dataset | Arms |
|---|---|---|
| `e22-hand` | `diabetes` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-hier-w` |
| `e22-auto-lexical` | `diabetes-auto-lexical` | `none`, `real`, `bko-direct`, `bko-nomap` |
| `e22-auto-local` | `diabetes-auto-local` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-hier-w` |
| `e22-permuted` | `diabetes-permuted` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-hier-w` |
| `e22-codes-hand` | `diabetes-codes` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-tree` |
| `e22-codes-auto-lexical` | `diabetes-codes-auto-lexical` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-tree` |
| `e22-codes-auto-local` | `diabetes-codes-auto-local` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-tree` |
| `e22-codes-permuted` | `diabetes-codes-permuted` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-tree` |

Every run is `bko-direct` against `bko-nomap` and `none`, with no replica, no paid arm and
`REFERENCE` `bko-direct`. `diabetes-auto-lexical` holds no row, so its family has no
`bko-hier-w` (ADR, Consequences). Each variant's load logs
`mapping dataset=%s rows=%d sha256=%s`.

Precision and recall come from `bko.ontology.automatic.precision_recall(built, hand)`, over
(field, code, concept) atoms with an exact curie, each `None` on an empty denominator. Measured
on the committed tables against the hand rows (`bko.ontology.auto_tables` against
`DIABETES_FIELD_ROWS` and `DIABETES_CODES_FIELD_ROWS`):

| Table | Rows | Precision | Recall |
|---|---|---|---|
| `diabetes-auto-lexical` | 0 | `None` | 0 |
| `diabetes-auto-local` | 10 | 1 | 5/7 |
| `diabetes-permuted` | 12 | 1/2 | 1/2 |
| `diabetes-codes-auto-lexical` | 96 | 1 | 24/25 |
| `diabetes-codes-auto-local` | 154 | 48/77 | 24/25 |
| `diabetes-codes-permuted` | 98 | 7/100 | 7/100 |

## Scope

**In.**

- The sixteen E22 runs and their slices, as above.
- Each mapping's precision and recall against the hand tables, as above. An exact match is
  reported apart from a hit on an ancestor or a descendant of the hand concept, which
  `precision_recall` counts as a miss.
- Automatic minus `bko-nomap`, and automatic minus hand under the registered margin.
- The permuted-hierarchy control.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Runs, Result and Read against.

**Out.**

- Claims beyond the development frames.

## Acceptance

- The entry is `complete`.
- `make check` passes.
