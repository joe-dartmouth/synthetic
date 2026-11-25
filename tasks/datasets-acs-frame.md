---
id: datasets-acs-frame
title: Add one ACS PUMS state-year as an exploratory non-health frame with occupation codes on the Occupation Ontology
status: ready
depends_on: --
blocks: evaluate-confirmatory-registration, evaluate-crc-acs-benchmark
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/acs-frame
---

# Add one ACS PUMS state-year as an exploratory non-health frame with occupation codes on the Occupation Ontology

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

A second domain tests whether the method depends on clinical ontologies. Occupation codes give
the hierarchy a five-level tree to work on. No page states one state-year's row count; ADR
[`research/confirmatory-dataset-selection`](../docs/adr/research/confirmatory-dataset-selection.md)
records what the PUMS user guide states, and this unit records the count from its file before any
gap is read.

## Scope

**In.**

- First, a survey, recorded in the new ADR under `docs/adr/research/`, of the ACS person file's
  coded columns (industry, field of degree, insurance, military service, place of birth, class of
  worker, education) for an openly licensed ontology that converts to OBO.
- Then, as unit [`datasets-support2-frame`](datasets-support2-frame.md), for one state-year of ACS
  PUMS, with income above the Folktables threshold as the label:
  - the detailed occupation code mapped to the Occupation Ontology through its SOC annotations,
    aggregate codes to their broad or minor class;
  - the six disability items mapped to HPO;
  - every column the survey found an ontology for, its release held per dataset (ADR
    [`research/ontology-mapping-design`](../docs/adr/research/ontology-mapping-design.md));
  - a `python -m bko.gaps acs` subcommand.
- The `Source` pin carries the `agreement` value ADR `research/confirmatory-dataset-selection`
  records for the Census terms, and the frame follows that ADR's reading of their
  re-identification clause.
- A registry entry in [`docs/literature.md`](../docs/literature.md) for the Folktables work.

**Out.**

- As unit [`datasets-support2-frame`](datasets-support2-frame.md): the test split, any classifier
  fit or arm score, and picking the removed cell.
- Columns with no ontology found, listed in the ADR.

## Acceptance

- The unit's ADR records the state-year's row count, counted from the training split before any
  gap is read. Below 5,000 records the dataset is excluded and ADR
  `research/confirmatory-dataset-selection` revised before registration.
- `python -m bko.ontology check` passes with the new rows.
- `python -m bko.gaps acs` lists the training split's gaps.
- The `contradictions` reading's output over the training split is recorded in the unit's ADR.
- `make check` passes.
