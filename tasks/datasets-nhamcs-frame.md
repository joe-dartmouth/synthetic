---
id: datasets-nhamcs-frame
title: Add NHAMCS 2022 ED as a confirmatory frame with hospital admission as the label
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/nhamcs-frame
---

# Add NHAMCS 2022 ED as a confirmatory frame with hospital admission as the label

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

A survey of emergency-department visits gives the confirmatory family a frame whose concepts come
from coded diagnoses. Its record count, its public-use terms and the NCHS clause on assessing
disclosure methods are quoted in ADR
[`research/confirmatory-dataset-selection`](../docs/adr/research/confirmatory-dataset-selection.md).
This unit copies [`datasets-support2-frame`](datasets-support2-frame.md) in scope and acceptance.

## Scope

**In.**

- A `Source` pin (`bko.datasets.sources`) of the NCHS file. Its `agreement` carries the value
  ADR `research/confirmatory-dataset-selection` records, and the unit's ADR cites that record's
  reading of the clause on assessing disclosure methods.
- A loader.
- A `FrameSpec` (`bko.datasets.frames`) with `ADMITHOS` as the label.
- Concepts from the ICD-10-CM diagnosis fields through MONDO's cross-references, and from the
  chronic-condition flags, committed before any gap listing is read.
- A `Dataset` member (`bko.evaluate.data`).
- A `python -m bko.gaps nhamcs` subcommand over the training split.
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
