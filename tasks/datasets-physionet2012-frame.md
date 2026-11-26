---
id: datasets-physionet2012-frame
title: Add PhysioNet/CinC Challenge 2012 as a confirmatory frame with in-hospital death as the label
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/physionet2012-frame
---

# Add PhysioNet/CinC Challenge 2012 as a confirmatory frame with in-hospital death as the label

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

A second intensive-care frame beside SUPPORT2's (ADR
[`research/support2-frame`](../docs/adr/research/support2-frame.md)) widens the confirmatory
family. Its record count, licence and in-hospital death rate are quoted in ADR
[`research/confirmatory-dataset-selection`](../docs/adr/research/confirmatory-dataset-selection.md).
This unit copies the SUPPORT2 frame's scope and acceptance, as ADR `research/support2-frame`
records.

## Scope

**In.**

- A `Source` pin (`bko.datasets.sources`) of the PhysioNet archive. Its `agreement` carries the
  value ADR `research/confirmatory-dataset-selection` records.
- A loader that reduces each stay's time series to one record by a rule fixed before any gap
  listing is read.
- A `FrameSpec` (`bko.datasets.frames`) with in-hospital death as the label.
- Mapping rows with cited cutoffs, committed from the codebook and the ontology before any gap
  listing is read.
- How missing values are read: a missing value shows no concept.
- A `Dataset` member (`bko.evaluate.data`).
- A `python -m bko.gaps physionet2012` subcommand over the training split.
- The contradictions census (`python -m bko.evaluate contradictions`).
- A new ADR under `docs/adr/research/` that records the frame, the reduction rule and the
  mapping.

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
