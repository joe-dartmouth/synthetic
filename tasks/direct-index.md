# Direct method: work-unit index

**What this set is.** The units that study `bko-direct` as the candidate of the direct-method study,
each one pull request. The plan is [`docs/RESEARCH.md`](../docs/RESEARCH.md); a unit that changes a
criterion, a definition or a decision an ADR records edits that document in its own PR. Each
unit's `milestone` is `direct-method-research`, the
[ROADMAP milestone](../docs/ROADMAP.md#direct-method-research) this set serves.

## Graph and status

### Controls

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`evaluate-control-runs-registered`](evaluate-control-runs-registered.md) | ready | -- | direct-method-research |
| [`evaluate-diabetes-control-results`](evaluate-diabetes-control-results.md) | blocked | evaluate-control-runs-registered | direct-method-research |
| [`evaluate-synthea-direct-results`](evaluate-synthea-direct-results.md) | blocked | evaluate-control-runs-registered | direct-method-research |

### Method and measurement

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`generate-is-a-conditional-draws`](generate-is-a-conditional-draws.md) | ready | -- | direct-method-research |
| [`generate-ancestor-shrunk-shares`](generate-ancestor-shrunk-shares.md) | ready | -- | direct-method-research |
| [`privacy-donor-limits`](privacy-donor-limits.md) | ready | -- | direct-method-research |
| [`privacy-value-microaggregation`](privacy-value-microaggregation.md) | blocked | privacy-donor-limits | direct-method-research |
| [`generate-coherent-free-fields`](generate-coherent-free-fields.md) | ready | -- | direct-method-research |
| [`evaluate-variant-dev-results`](evaluate-variant-dev-results.md) | blocked | generate-is-a-conditional-draws, generate-ancestor-shrunk-shares, privacy-donor-limits, privacy-value-microaggregation, generate-coherent-free-fields | direct-method-research |
| [`gaps-share-based-quota`](gaps-share-based-quota.md) | blocked | evaluate-control-runs-registered | direct-method-research |
| [`evaluate-quota-sweep-results`](evaluate-quota-sweep-results.md) | blocked | gaps-share-based-quota | direct-method-research |

### Datasets

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`datasets-support2-frame`](datasets-support2-frame.md) | ready | -- | direct-method-research |
| [`datasets-nhanes-frame`](datasets-nhanes-frame.md) | ready | -- | direct-method-research |
| [`datasets-acs-frame`](datasets-acs-frame.md) | ready | -- | direct-method-research |
| [`datasets-physionet2012-frame`](datasets-physionet2012-frame.md) | ready | -- | direct-method-research |
| [`datasets-nhamcs-frame`](datasets-nhamcs-frame.md) | ready | -- | direct-method-research |
| [`datasets-non-health-selection`](datasets-non-health-selection.md) | ready | -- | direct-method-research |
| [`datasets-generalised-cell-axes`](datasets-generalised-cell-axes.md) | blocked | datasets-non-health-selection | direct-method-research |

### Published comparisons

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`evaluate-covariate-bias-fractions`](evaluate-covariate-bias-fractions.md) | ready | -- | direct-method-research |
| [`evaluate-covariate-bias-results`](evaluate-covariate-bias-results.md) | blocked | evaluate-covariate-bias-fractions | direct-method-research |
| [`evaluate-tableshift-results`](evaluate-tableshift-results.md) | ready | -- | direct-method-research |
| [`evaluate-crc-acs-benchmark`](evaluate-crc-acs-benchmark.md) | blocked | datasets-acs-frame | direct-method-research |
| [`evaluate-crc-acs-results`](evaluate-crc-acs-results.md) | blocked | evaluate-crc-acs-benchmark, evaluate-confirmatory-results | direct-method-research |
| [`evaluate-syntheval-results`](evaluate-syntheval-results.md) | ready | -- | direct-method-research |
| [`evaluate-synthetic-share-sweep`](evaluate-synthetic-share-sweep.md) | ready | -- | direct-method-research |
| [`evaluate-synthetic-share-results`](evaluate-synthetic-share-results.md) | blocked | evaluate-synthetic-share-sweep | direct-method-research |

### Composite and confirmation

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`evaluate-composite-arm`](evaluate-composite-arm.md) | blocked | evaluate-variant-dev-results, evaluate-quota-sweep-results | direct-method-research |
| [`evaluate-composite-dev-results`](evaluate-composite-dev-results.md) | blocked | evaluate-composite-arm | direct-method-research |
| [`evaluate-confirmatory-registration`](evaluate-confirmatory-registration.md) | blocked | evaluate-diabetes-control-results, evaluate-synthea-direct-results, datasets-support2-frame, datasets-nhanes-frame, datasets-acs-frame, datasets-physionet2012-frame, datasets-nhamcs-frame, evaluate-composite-dev-results | direct-method-research |
| [`evaluate-confirmatory-results`](evaluate-confirmatory-results.md) | blocked | evaluate-confirmatory-registration | direct-method-research |

### Follow-ups

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`ontology-auto-mapping`](ontology-auto-mapping.md) | blocked | generate-ancestor-shrunk-shares | direct-method-research |
| [`ontology-auto-mapping-results`](ontology-auto-mapping-results.md) | blocked | ontology-auto-mapping | direct-method-research |
| [`slm-lora-day-budget`](slm-lora-day-budget.md) | ready | -- | direct-method-research |
| [`slm-lora-results`](slm-lora-results.md) | blocked | evaluate-composite-dev-results, slm-lora-day-budget | direct-method-research |

```mermaid
flowchart LR
    u37[evaluate-covariate-bias-fractions] --> u38[evaluate-covariate-bias-results]
    u6[evaluate-control-runs-registered] --> u7[evaluate-diabetes-control-results]
    u6 --> u8[evaluate-synthea-direct-results]
    u15[privacy-donor-limits] --> u16[privacy-value-microaggregation]
    u11[generate-is-a-conditional-draws] --> u19[evaluate-variant-dev-results]
    u13[generate-ancestor-shrunk-shares] --> u19
    u15 --> u19
    u16 --> u19
    u17[generate-coherent-free-fields] --> u19
    u6 --> u20[gaps-share-based-quota]
    u20 --> u21[evaluate-quota-sweep-results]
    u27[datasets-acs-frame] --> u41[evaluate-crc-acs-benchmark]
    u49[datasets-non-health-selection] --> u50[datasets-generalised-cell-axes]
    u19 --> u28[evaluate-composite-arm]
    u21 --> u28
    u28 --> u29[evaluate-composite-dev-results]
    u7 --> u30[evaluate-confirmatory-registration]
    u8 --> u30
    u45[evaluate-synthetic-share-sweep] --> u46[evaluate-synthetic-share-results]
    u25[datasets-support2-frame] --> u30
    u26[datasets-nhanes-frame] --> u30
    u27 --> u30
    u47[datasets-physionet2012-frame] --> u30
    u48[datasets-nhamcs-frame] --> u30
    u29 --> u30
    u30 --> u31[evaluate-confirmatory-results]
    u41 --> u42[evaluate-crc-acs-results]
    u31 --> u42
    u13 --> u32[ontology-auto-mapping]
    u32 --> u33[ontology-auto-mapping-results]
    u29 --> u36[slm-lora-results]
    u51[slm-lora-day-budget] --> u36
```

## What every unit relies on

- Everything the experiments set relies on
  ([`experiments-index.md`](experiments-index.md#what-every-unit-relies-on)).
- **No unit edits an existing `RUNS` entry, an existing dataset's field rows or `bko-direct`'s
  draw.** Old run folders are re-read under identity checks, and the draw is pinned by digests,
  so a sampler change is a new arm name and a mapping change is a new dataset name.
- **A code unit lands its runs' names and Configuration; a results unit runs them.** A code unit
  adds the arm or reading, its named `RUNS` entries and the
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Configuration with status `not yet run`. A
  results unit runs it and records Runs, Result and Read against. A free reading over stores that
  already exist lands with its code unit, recorded as a Runs row in the code unit's PR; a
  reading that runs for hours is recorded by a results unit instead.
- **Which data is development and which is confirmatory is the plan's to say**
  ([`docs/RESEARCH.md`](../docs/RESEARCH.md)).

## Where results go

[`docs/EVIDENCE.md`](../docs/EVIDENCE.md): a new entry per question.

## Spend

As the experiments set's ([`experiments-index.md`](experiments-index.md#spend)).

## Conventions

As the experiments set's ([`experiments-index.md`](experiments-index.md#conventions)).
