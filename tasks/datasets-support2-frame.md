---
id: datasets-support2-frame
title: Add SUPPORT2 as a confirmatory frame with in-hospital death as the label
status: blocked
depends_on: datasets-selection-rule
blocks: evaluate-confirmatory-registration
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/support2-frame
---

# Add SUPPORT2 as a confirmatory frame with in-hospital death as the label

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

One positive dataset is the study's weakest point. SUPPORT2 is intensive-care data, with
diagnosis groups that map to MONDO and physiology that maps to HPO through cited cutoffs. Its
terms and its in-hospital death rate are what unit
[`datasets-selection-rule`](datasets-selection-rule.md) re-reads from the primary page.

## Scope

**In.**

- A `Source` pin (`bko.datasets.sources`) of the Vanderbilt archive. Its `agreement` carries the
  value unit [`datasets-selection-rule`](datasets-selection-rule.md) records, and is `None` only
  where that unit's ADR says the terms permit sending records to a third-party API.
- A loader.
- A `FrameSpec` (`bko.datasets.frames`) with in-hospital death as the label, without the columns
  its publisher warns against using as predictors.
- Mapping rows with cited cutoffs, committed from the codebook and the ontology before any gap
  listing is read.
- How missing laboratory values are read: a missing value shows no concept.
- A `Dataset` member (`bko.evaluate.data`).
- A `python -m bko.gaps support2` subcommand over the training split.
- The contradictions census (`python -m bko.evaluate contradictions`).
- A new ADR under `docs/adr/research/` that records the frame and the mapping.

**Out.**

- The test split.
- Any classifier fit or arm score.
- Picking the removed cell (unit
  [`evaluate-confirmatory-registration`](evaluate-confirmatory-registration.md)).

## Acceptance

- `python -m bko.ontology check` passes with the new rows.
- The subcommand lists the training split's gaps.
- The `contradictions` reading's output over the training split is recorded in the unit's ADR.
- `make check` passes.
