---
id: datasets-selection-rule
title: Record the confirmatory-dataset selection rule and re-read each choice from its primary page
status: ready
depends_on: --
blocks: datasets-support2-frame, datasets-nhanes-frame, datasets-acs-frame, datasets-physionet2012-frame, datasets-nhamcs-frame, datasets-non-health-selection
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
  Framingham teaching set, ACS PUMS, UCI Adult, HMDA, the PhysioNet/CinC Challenge 2012 set,
  NHAMCS 2022 ED, SPARCS 2022, the PhysioNet/CinC Challenge 2019 set and UCI Census-Income KDD.
  The five chosen are SUPPORT2, NHANES, the PhysioNet/CinC Challenge 2012 set, NHAMCS 2022 ED and
  ACS PUMS.
- A re-read, from each primary page, of what this set took unverified: SUPPORT2's terms at ICPSR
  and its in-hospital death rate, the size of the NHANES composite outcome, the row count of one
  ACS state-year, the PhysioNet/CinC Challenge 2012 set's licence and in-hospital death rate, and
  NHAMCS 2022 ED's terms and record count, including whether its file is public-use or held under
  an agreement; a dataset under an agreement the team does not hold is excluded.
- For each of the five chosen datasets, the ADR records:
  - whether its terms permit sending records to a third-party API, as the `Agreement` value its
    `Source` pin will carry (`bko.datasets.sources`; `agreement=None` means open data, which
    `permits_sending` lets go to the API);
  - any clause on re-identification and the team's reading of it. For NHANES that reading is that
    scoring the privacy of generated rows is outside the clause; SUPPORT2, the PhysioNet/CinC
    Challenge 2012 set, NHAMCS 2022 ED and ACS PUMS get the same treatment, each reading recorded
    only after its terms are re-read.
- This unit changes [`docs/RESEARCH.md`](../docs/RESEARCH.md#6-datasets) section 6 to list the
  five. ACS PUMS is its own exploratory family, outside the health datasets' multiplicity
  family.

**Out.**

- Loading any candidate's records.
- Generalising the cell axes.

## Acceptance

- The ADR ranks every candidate, with its license and access quoted from the primary page.
- The re-read contradicts no choice, or the ADR revises the choice before units
  [`datasets-support2-frame`](datasets-support2-frame.md),
  [`datasets-nhanes-frame`](datasets-nhanes-frame.md),
  [`datasets-physionet2012-frame`](datasets-physionet2012-frame.md),
  [`datasets-nhamcs-frame`](datasets-nhamcs-frame.md) and
  [`datasets-acs-frame`](datasets-acs-frame.md) start.
- The ADR records, for each chosen dataset, its `Agreement` value and its re-identification
  clause with the team's reading, or that its terms hold none.
- `make check` passes.
