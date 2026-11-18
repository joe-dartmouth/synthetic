---
id: datasets-selection-rule
title: Record the confirmatory-dataset selection rule and re-read each choice from its primary page
status: blocked
depends_on: plan-direct-method-research
blocks: datasets-support2-frame, datasets-nhanes-frame, datasets-acs-frame
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/confirmatory-dataset-selection
---

# Record the confirmatory-dataset selection rule and re-read each choice from its primary page

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

A confirmatory dataset chosen after seeing results confirms nothing. The choice this set makes
rests on web reads not yet checked against every primary page. This unit fixes the rule before
any candidate's records are loaded and checks each read against its source.

## Scope

**In.**

- A new ADR under `docs/adr/research/` stating the selection rule:
  - open access, or an agreement the team holds;
  - a binary outcome;
  - numeric age and binary sex;
  - mapped concepts from a code crosswalk or a cited cutoff, with at least one hierarchical coded
    field for the non-health family;
  - a minimum record count taken from the dataset's own documentation;
  - the mechanical rule that later picks the removed cell from the training split.
- In the same ADR, the ranking of every candidate considered, with its exclusion reason:
  SUPPORT2, NHANES, BRFSS, MEPS, CDC Diabetes Health Indicators, eICU demo, MIMIC-IV demo,
  Framingham teaching set, ACS PUMS, UCI Adult and HMDA. The three chosen are SUPPORT2, NHANES
  and ACS PUMS.
- A re-read, from each primary page, of what this set took unverified: SUPPORT2's terms at ICPSR
  and its in-hospital death rate, the size of the NHANES composite outcome, and the row count of
  one ACS state-year.
- For each of the three chosen datasets, the ADR records:
  - whether its terms permit sending records to a third-party API, as the `Agreement` value its
    `Source` pin will carry (`bko.datasets.sources`; `agreement=None` means open data, which
    `permits_sending` lets go to the API);
  - any clause on re-identification and the team's reading of it. For NHANES that reading is that
    scoring the privacy of generated rows is outside the clause; SUPPORT2 and ACS PUMS get the
    same treatment.
- This unit changes [`docs/RESEARCH.md`](../docs/RESEARCH.md#6-datasets) section 6 to list the
  three. ACS PUMS is its own exploratory family, outside the health datasets' multiplicity
  family.

**Out.**

- Loading any candidate's records.
- Generalising the cell axes.

## Acceptance

- The ADR ranks every candidate, with its license and access quoted from the primary page.
- The re-read contradicts no choice, or the ADR revises the choice before units
  [`datasets-support2-frame`](datasets-support2-frame.md),
  [`datasets-nhanes-frame`](datasets-nhanes-frame.md) and
  [`datasets-acs-frame`](datasets-acs-frame.md) start.
- The ADR records, for each chosen dataset, its `Agreement` value and its re-identification
  clause with the team's reading, or that its terms hold none.
- `make check` passes.
