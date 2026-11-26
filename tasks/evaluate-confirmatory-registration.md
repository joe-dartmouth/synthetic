---
id: evaluate-confirmatory-registration
title: Register the confirmatory runs, endpoint and contrasts on the new datasets and tag the commit
status: blocked
depends_on: evaluate-diabetes-control-results, evaluate-synthea-direct-results, datasets-physionet2012-frame, datasets-nhamcs-frame, evaluate-composite-dev-results
blocks: evaluate-confirmatory-results
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: --
---

# Register the confirmatory runs, endpoint and contrasts on the new datasets and tag the commit

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The confirmatory run is evidence only if the method, the comparators, the criteria and the
readings are fixed before any score exists on the new datasets.

## Scope

**In.**

- Confirmatory `RUNS` entries per new dataset, the four health datasets of
  [`docs/RESEARCH.md`](../docs/RESEARCH.md) section 11 (SUPPORT2, NHANES, the PhysioNet/CinC
  Challenge 2012 set and NHAMCS 2022 ED, framed by ADRs
  [`research/support2-frame`](../docs/adr/research/support2-frame.md) and
  [`research/nhanes-frame`](../docs/adr/research/nhanes-frame.md) and units
  [`datasets-physionet2012-frame`](datasets-physionet2012-frame.md) and
  [`datasets-nhamcs-frame`](datasets-nhamcs-frame.md)) and ACS PUMS, framed by ADR
  [`research/acs-frame`](../docs/adr/research/acs-frame.md), a separate exploratory family:
  - candidate from unit [`evaluate-composite-dev-results`](evaluate-composite-dev-results.md).
    When that unit reports the composite failing the non-inferiority margin, `bko-direct` is
    candidate and `bko-direct-2` an ablation arm. Otherwise `bko-direct-2` is candidate and
    `bko-direct` an ablation arm;
  - comparators `METHOD_ARMS` (`bko.evaluate.experiment`), `bayes-net`, `tabddpm`,
    `label-bootstrap` and `prior-shift`, plus every control whose contrast units
    [`evaluate-diabetes-control-results`](evaluate-diabetes-control-results.md) or
    [`evaluate-synthea-direct-results`](evaluate-synthea-direct-results.md) read in the `below`
    branch. This is how the control reading rule's `below` branch is applied, `bko-nomap`
    included;
  - ablations `bko-nomap` (unless it joins the comparators), `bko-logic`, and `bko` where the
    dataset's agreement permits sending records.
- The removed cells, each picked from its dataset's training split by the count filter and grid
  order of ADR `research/confirmatory-dataset-selection`, with the registered cutoff at the training
  split's prevalence, and recorded with that split's sha256. A dataset on which no cell passes is
  excluded and that ADR revised before the tag.
- In [`docs/RESEARCH.md`](../docs/RESEARCH.md) section 11:
  - the primary endpoint, taken from the branch units
    [`evaluate-diabetes-control-results`](evaluate-diabetes-control-results.md) and
    [`evaluate-synthea-direct-results`](evaluate-synthea-direct-results.md) recorded;
  - one primary contrast per dataset when the control reading rule registers a superiority
    hypothesis. When it registers none, the run reports equivalence, privacy and attribution
    readings, as the rule says;
  - Holm's correction across the comparators and the four health datasets, with its registry
    entry in [`docs/literature.md`](../docs/literature.md); ACS PUMS is a separate exploratory
    family outside it;
  - the equivalence margin;
  - the holdout count, derived from E6's interval widths
    ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)).
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) Configurations.
- A signed tag.
- ADR [`research/e1-evaluation-protocol`](../docs/adr/research/e1-evaluation-protocol.md) revised
  in place.

**Out.**

- Any score, rehearsal or fit on a confirmatory dataset.
- Any change after the tag.

## Acceptance

- A test constructs every confirmatory `NamedRun` (`bko.evaluate.named_run`) in
  `bko.evaluate.named_runs.RUNS`.
- `git tag -v <tag>` verifies the signed tag, and each Configuration names the tagged commit.
- `make check` passes.
