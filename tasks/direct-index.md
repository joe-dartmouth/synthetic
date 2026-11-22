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
| [`generate-no-mapping-ablation`](generate-no-mapping-ablation.md) | ready | -- | direct-method-research |
| [`baselines-label-bootstrap`](baselines-label-bootstrap.md) | ready | -- | direct-method-research |
| [`evaluate-prior-shift-comparator`](evaluate-prior-shift-comparator.md) | ready | -- | direct-method-research |
| [`evaluate-control-runs-registered`](evaluate-control-runs-registered.md) | blocked | generate-no-mapping-ablation, baselines-label-bootstrap, evaluate-prior-shift-comparator | direct-method-research |
| [`evaluate-diabetes-control-results`](evaluate-diabetes-control-results.md) | blocked | evaluate-control-runs-registered | direct-method-research |
| [`evaluate-synthea-direct-results`](evaluate-synthea-direct-results.md) | blocked | evaluate-control-runs-registered | direct-method-research |

### Method and measurement

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`evaluate-added-row-contradictions`](evaluate-added-row-contradictions.md) | ready | -- | direct-method-research |
| [`gaps-is-a-conditional-fragments`](gaps-is-a-conditional-fragments.md) | ready | -- | direct-method-research |
| [`generate-is-a-conditional-draws`](generate-is-a-conditional-draws.md) | blocked | generate-no-mapping-ablation, gaps-is-a-conditional-fragments, evaluate-added-row-contradictions | direct-method-research |
| [`ontology-diabetes-diagnosis-codes`](ontology-diabetes-diagnosis-codes.md) | ready | -- | direct-method-research |
| [`generate-ancestor-shrunk-shares`](generate-ancestor-shrunk-shares.md) | blocked | generate-no-mapping-ablation, ontology-diabetes-diagnosis-codes | direct-method-research |
| [`privacy-donor-attacks`](privacy-donor-attacks.md) | ready | -- | direct-method-research |
| [`privacy-donor-limits`](privacy-donor-limits.md) | blocked | generate-no-mapping-ablation, privacy-donor-attacks | direct-method-research |
| [`privacy-value-microaggregation`](privacy-value-microaggregation.md) | blocked | privacy-donor-limits | direct-method-research |
| [`generate-coherent-free-fields`](generate-coherent-free-fields.md) | blocked | generate-no-mapping-ablation, evaluate-added-row-contradictions, privacy-donor-attacks | direct-method-research |
| [`evaluate-trace-attribution`](evaluate-trace-attribution.md) | ready | -- | direct-method-research |
| [`evaluate-variant-dev-results`](evaluate-variant-dev-results.md) | blocked | generate-is-a-conditional-draws, generate-ancestor-shrunk-shares, privacy-donor-limits, privacy-value-microaggregation, generate-coherent-free-fields, evaluate-trace-attribution | direct-method-research |
| [`gaps-share-based-quota`](gaps-share-based-quota.md) | blocked | evaluate-control-runs-registered | direct-method-research |
| [`evaluate-quota-sweep-results`](evaluate-quota-sweep-results.md) | blocked | gaps-share-based-quota | direct-method-research |
| [`baselines-tabddpm`](baselines-tabddpm.md) | ready | -- | direct-method-research |

### Datasets

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`datasets-selection-rule`](datasets-selection-rule.md) | ready | -- | direct-method-research |
| [`ontology-per-dataset-releases`](ontology-per-dataset-releases.md) | ready | -- | direct-method-research |
| [`datasets-support2-frame`](datasets-support2-frame.md) | blocked | datasets-selection-rule | direct-method-research |
| [`datasets-nhanes-frame`](datasets-nhanes-frame.md) | blocked | datasets-selection-rule | direct-method-research |
| [`datasets-acs-frame`](datasets-acs-frame.md) | blocked | datasets-selection-rule, ontology-per-dataset-releases | direct-method-research |
| [`datasets-physionet2012-frame`](datasets-physionet2012-frame.md) | blocked | datasets-selection-rule | direct-method-research |
| [`datasets-nhamcs-frame`](datasets-nhamcs-frame.md) | blocked | datasets-selection-rule | direct-method-research |
| [`datasets-non-health-selection`](datasets-non-health-selection.md) | blocked | datasets-selection-rule | direct-method-research |
| [`datasets-generalised-cell-axes`](datasets-generalised-cell-axes.md) | blocked | datasets-non-health-selection | direct-method-research |

### Published comparisons

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`evaluate-covariate-bias-fractions`](evaluate-covariate-bias-fractions.md) | blocked | generate-no-mapping-ablation, baselines-label-bootstrap, evaluate-prior-shift-comparator | direct-method-research |
| [`evaluate-covariate-bias-results`](evaluate-covariate-bias-results.md) | blocked | evaluate-covariate-bias-fractions | direct-method-research |
| [`evaluate-tableshift-readmission`](evaluate-tableshift-readmission.md) | ready | -- | direct-method-research |
| [`evaluate-tableshift-results`](evaluate-tableshift-results.md) | blocked | evaluate-tableshift-readmission | direct-method-research |
| [`evaluate-crc-acs-benchmark`](evaluate-crc-acs-benchmark.md) | blocked | datasets-acs-frame | direct-method-research |
| [`evaluate-crc-acs-results`](evaluate-crc-acs-results.md) | blocked | evaluate-crc-acs-benchmark, evaluate-confirmatory-results | direct-method-research |
| [`evaluate-syntheval-heart-disease`](evaluate-syntheval-heart-disease.md) | ready | -- | direct-method-research |
| [`evaluate-syntheval-results`](evaluate-syntheval-results.md) | blocked | evaluate-syntheval-heart-disease | direct-method-research |
| [`evaluate-synthetic-share-sweep`](evaluate-synthetic-share-sweep.md) | blocked | baselines-label-bootstrap, baselines-tabddpm | direct-method-research |
| [`evaluate-synthetic-share-results`](evaluate-synthetic-share-results.md) | blocked | evaluate-synthetic-share-sweep | direct-method-research |

### Composite and confirmation

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`evaluate-composite-arm`](evaluate-composite-arm.md) | blocked | evaluate-variant-dev-results, evaluate-quota-sweep-results | direct-method-research |
| [`evaluate-composite-dev-results`](evaluate-composite-dev-results.md) | blocked | evaluate-composite-arm | direct-method-research |
| [`evaluate-confirmatory-registration`](evaluate-confirmatory-registration.md) | blocked | evaluate-diabetes-control-results, evaluate-synthea-direct-results, evaluate-trace-attribution, baselines-tabddpm, datasets-support2-frame, datasets-nhanes-frame, datasets-acs-frame, datasets-physionet2012-frame, datasets-nhamcs-frame, evaluate-composite-dev-results | direct-method-research |
| [`evaluate-confirmatory-results`](evaluate-confirmatory-results.md) | blocked | evaluate-confirmatory-registration | direct-method-research |

### Follow-ups

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`ontology-auto-mapping`](ontology-auto-mapping.md) | blocked | generate-ancestor-shrunk-shares | direct-method-research |
| [`ontology-auto-mapping-results`](ontology-auto-mapping-results.md) | blocked | ontology-auto-mapping | direct-method-research |
| [`slm-lora-spike`](slm-lora-spike.md) | ready | -- | direct-method-research |
| [`slm-lora-trainer`](slm-lora-trainer.md) | blocked | slm-lora-spike | direct-method-research |
| [`slm-lora-results`](slm-lora-results.md) | blocked | evaluate-composite-dev-results, slm-lora-trainer, baselines-label-bootstrap, baselines-tabddpm | direct-method-research |

```mermaid
flowchart LR
    u3[generate-no-mapping-ablation] --> u6[evaluate-control-runs-registered]
    u3 --> u37[evaluate-covariate-bias-fractions]
    u4[baselines-label-bootstrap] --> u6
    u4 --> u37
    u5[evaluate-prior-shift-comparator] --> u6
    u5 --> u37
    u37 --> u38[evaluate-covariate-bias-results]
    u39[evaluate-tableshift-readmission] --> u40[evaluate-tableshift-results]
    u6 --> u7[evaluate-diabetes-control-results]
    u6 --> u8[evaluate-synthea-direct-results]
    u3 --> u11[generate-is-a-conditional-draws]
    u10[gaps-is-a-conditional-fragments] --> u11
    u9[evaluate-added-row-contradictions] --> u11
    u3 --> u13[generate-ancestor-shrunk-shares]
    u12[ontology-diabetes-diagnosis-codes] --> u13
    u3 --> u15[privacy-donor-limits]
    u14[privacy-donor-attacks] --> u15
    u15 --> u16[privacy-value-microaggregation]
    u3 --> u17[generate-coherent-free-fields]
    u9 --> u17
    u14 --> u17
    u11 --> u19[evaluate-variant-dev-results]
    u13 --> u19
    u15 --> u19
    u16 --> u19
    u17 --> u19
    u18[evaluate-trace-attribution] --> u19
    u6 --> u20[gaps-share-based-quota]
    u20 --> u21[evaluate-quota-sweep-results]
    u23[datasets-selection-rule] --> u25[datasets-support2-frame]
    u23 --> u26[datasets-nhanes-frame]
    u23 --> u27[datasets-acs-frame]
    u24[ontology-per-dataset-releases] --> u27
    u27 --> u41[evaluate-crc-acs-benchmark]
    u23 --> u47[datasets-physionet2012-frame]
    u23 --> u48[datasets-nhamcs-frame]
    u23 --> u49[datasets-non-health-selection]
    u49 --> u50[datasets-generalised-cell-axes]
    u43[evaluate-syntheval-heart-disease] --> u44[evaluate-syntheval-results]
    u19 --> u28[evaluate-composite-arm]
    u21 --> u28
    u28 --> u29[evaluate-composite-dev-results]
    u7 --> u30[evaluate-confirmatory-registration]
    u8 --> u30
    u18 --> u30
    u22[baselines-tabddpm] --> u30
    u4 --> u45[evaluate-synthetic-share-sweep]
    u22 --> u45
    u45 --> u46[evaluate-synthetic-share-results]
    u25 --> u30
    u26 --> u30
    u27 --> u30
    u47 --> u30
    u48 --> u30
    u29 --> u30
    u30 --> u31[evaluate-confirmatory-results]
    u41 --> u42[evaluate-crc-acs-results]
    u31 --> u42
    u13 --> u32[ontology-auto-mapping]
    u32 --> u33[ontology-auto-mapping-results]
    u34[slm-lora-spike] --> u35[slm-lora-trainer]
    u29 --> u36[slm-lora-results]
    u35 --> u36
    u4 --> u36
    u22 --> u36
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
  already exist lands with its code unit, recorded as a Runs row in the code unit's PR.
- **Which data is development and which is confirmatory is the plan's to say**
  ([`docs/RESEARCH.md`](../docs/RESEARCH.md)).

## Where results go

[`docs/EVIDENCE.md`](../docs/EVIDENCE.md): a new entry per question.

## Spend

As the experiments set's ([`experiments-index.md`](experiments-index.md#spend)).

## Conventions

As the experiments set's ([`experiments-index.md`](experiments-index.md#conventions)).
