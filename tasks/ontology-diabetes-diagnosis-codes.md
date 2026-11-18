---
id: ontology-diabetes-diagnosis-codes
title: Map Diabetes 130's ICD-9 diagnosis codes as a new dataset name
status: blocked
depends_on: refactor-evaluate-size-splits
blocks: generate-ancestor-shrunk-shares
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/diabetes-diagnosis-codes
---

# Map Diabetes 130's ICD-9 diagnosis codes as a new dataset name

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Diabetes 130's frame holds one indicator concept per ICD-9 category row and lab row of
`bko.ontology.icd9`, twelve in all, which leaves the hierarchy almost nothing to do. Its three
diagnosis fields (`bko.datasets.diabetes.DIAGNOSES`) hold several hundred ICD-9 codes. Unit
[`generate-ancestor-shrunk-shares`](generate-ancestor-shrunk-shares.md) needs a coded field that
fine.

## Scope

**In.**

- A new dataset name, `diabetes-codes`, over the same pinned archive.
- A frame that keeps each diagnosis field's ICD-9 code.
- One field row per code that has a MONDO cross-reference in the pinned release, generated from
  the release's cross-references and checked by `python -m bko.ontology check`.
- Codes with no cross-reference stay unmapped and are counted in a new ADR under
  `docs/adr/research/`, which records the mapping beside ADR
  [`research/ontology-mapping-design`](../docs/adr/research/ontology-mapping-design.md).
- `Dataset.DIABETES` (`bko.evaluate.data`) and its field rows are unchanged.

**Out.**

- Any sampler change.
- Any run.

## Acceptance

- A test shows every generated row's concept is a live MONDO term and every mapped code maps to
  one concept.
- The existing Diabetes 130 run-folder checks still pass: `rows` reproduces one E6 slice.
- `make check` passes.
