---
id: evaluate-variant-dev-results
title: Run the direct variants on the development frames and apply the inclusion rule
status: ready
depends_on: --
blocks: evaluate-composite-arm, paper-novelty
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the direct variants on the development frames and apply the inclusion rule

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Each variant claims one improvement over `bko-direct`. The runs say which hold without costing
removed-cell recall, and the variant inclusion rule in [`docs/RESEARCH.md`](../docs/RESEARCH.md)
says which enter `bko-direct-2`.

## Scope

**In.**

- `bko-micro`'s runs `e20-thin` and `e20-empty`, each sliced in ten, `e20-thina` to `e20-thinj`
  and `e20-emptya` to `e20-emptyj`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e20-microaggregated-count-fields-on-the-diabetes-130-removal)),
  `bko-capped`'s runs
  `e14-thin` and `e14-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e14-donor-floor-and-reuse-cap-on-the-diabetes-130-removal)),
  `bko-tree`'s runs `e15-thin` and `e15-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e15-ancestor-shrunk-shares-on-the-diabetes-codes-frame)),
  the `bko-hier` arms' runs `e16-thin` and `e16-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e16-conditional-is_a-draws-on-the-diabetes-130-removal)),
  and `bko-coherent`'s runs `e17-thin` and `e17-empty`
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e17-coherent-free-fields-on-the-diabetes-130-removal)).
- For each arm, `contradictions --rows` (ADR
  [`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md),
  Decision 2), `attribution` (ADR
  [`research/record-explainability`](../docs/adr/research/record-explainability.md), Decision 7),
  and `privacy`'s match attack and
  donor reuse (ADR [`research/membership-inference-reading`](../docs/adr/research/membership-inference-reading.md)).
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entries' Runs, Result and Read against.
- For each variant, whether it meets the variant inclusion rule.

**Out.**

- Choosing parameters not registered.

**Findings from planning.**

- E20's primary contrast is `bko-micro` minus `bko-capped`, which `privacy` reads per holdout
  with `bko-capped` the reference arm; `bko-direct` minus `bko-capped` is read beside it.
- `bko-micro`'s count fields carry the share of their donors' variance ADR
  [`research/microaggregated-donors`](../docs/adr/research/microaggregated-donors.md) states
  (Rationale); its fidelity reading is read with that.
- `bko-micro`'s `score` log writes the `micro-field` lines whose format and keys
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) holds; on E6's frame they name the
  eight count fields.
- A free slice of `e20-thina` (PR 4's evidence, not a run of record) read, summed over its ten
  holdouts' `micro-field` lines: `classless` on 98.5% of each count field's values, since the
  thin cell's 19 records fill few means under the cap, so nearly every mean is drawn from the
  label group; `under` 0; `1 - sse/sst` from 0.324 to 0.336; and `equal` from 0.075
  (`num_lab_procedures`) to 0.923 (`number_emergency`), so on the zero-heavy fields most means
  equal one of their donors. The reading states these beside the privacy contrast.

## Acceptance

- Each entry is `complete` and states pass or fail under the rule.
- `make check` passes.
