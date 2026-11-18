# Direct method: work-unit index

**What this set is.** The units that make `bko-direct` the primary method of the study, each one
pull request. The plan is [`docs/RESEARCH.md`](../docs/RESEARCH.md); a unit that changes a
criterion, a definition or a decision an ADR records edits that document in its own PR. Each
unit's `milestone` is `direct-method-research`, the
[ROADMAP milestone](../docs/ROADMAP.md#direct-method-research) this set serves.

## Graph and status

### Foundation

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`plan-direct-method-research`](plan-direct-method-research.md) | ready | -- | direct-method-research |
| [`refactor-evaluate-size-splits`](refactor-evaluate-size-splits.md) | ready | -- | direct-method-research |

### Controls

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`generate-no-mapping-ablation`](generate-no-mapping-ablation.md) | blocked | plan-direct-method-research, refactor-evaluate-size-splits | direct-method-research |
| [`baselines-label-bootstrap`](baselines-label-bootstrap.md) | blocked | plan-direct-method-research | direct-method-research |
| [`evaluate-prior-shift-comparator`](evaluate-prior-shift-comparator.md) | blocked | plan-direct-method-research, refactor-evaluate-size-splits | direct-method-research |
| [`evaluate-control-runs-registered`](evaluate-control-runs-registered.md) | blocked | generate-no-mapping-ablation, baselines-label-bootstrap, evaluate-prior-shift-comparator | direct-method-research |
| [`evaluate-diabetes-control-results`](evaluate-diabetes-control-results.md) | blocked | evaluate-control-runs-registered | direct-method-research |
| [`evaluate-synthea-direct-results`](evaluate-synthea-direct-results.md) | blocked | evaluate-control-runs-registered | direct-method-research |

### Method and measurement

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`evaluate-added-row-contradictions`](evaluate-added-row-contradictions.md) | blocked | refactor-evaluate-size-splits | direct-method-research |
| [`gaps-is-a-conditional-fragments`](gaps-is-a-conditional-fragments.md) | blocked | refactor-evaluate-size-splits | direct-method-research |
| [`generate-is-a-conditional-draws`](generate-is-a-conditional-draws.md) | blocked | generate-no-mapping-ablation, gaps-is-a-conditional-fragments, evaluate-added-row-contradictions | direct-method-research |
| [`ontology-diabetes-diagnosis-codes`](ontology-diabetes-diagnosis-codes.md) | blocked | refactor-evaluate-size-splits | direct-method-research |
| [`generate-ancestor-shrunk-shares`](generate-ancestor-shrunk-shares.md) | blocked | generate-no-mapping-ablation, ontology-diabetes-diagnosis-codes | direct-method-research |
| [`privacy-donor-attacks`](privacy-donor-attacks.md) | blocked | refactor-evaluate-size-splits | direct-method-research |
| [`privacy-donor-limits`](privacy-donor-limits.md) | blocked | generate-no-mapping-ablation, privacy-donor-attacks | direct-method-research |
| [`privacy-value-microaggregation`](privacy-value-microaggregation.md) | blocked | privacy-donor-limits | direct-method-research |
| [`generate-coherent-free-fields`](generate-coherent-free-fields.md) | blocked | generate-no-mapping-ablation, evaluate-added-row-contradictions, privacy-donor-attacks | direct-method-research |
| [`evaluate-trace-attribution`](evaluate-trace-attribution.md) | blocked | refactor-evaluate-size-splits | direct-method-research |
| [`evaluate-variant-dev-results`](evaluate-variant-dev-results.md) | blocked | generate-is-a-conditional-draws, generate-ancestor-shrunk-shares, privacy-donor-limits, privacy-value-microaggregation, generate-coherent-free-fields, evaluate-trace-attribution | direct-method-research |
| [`gaps-share-based-quota`](gaps-share-based-quota.md) | blocked | refactor-evaluate-size-splits | direct-method-research |
| [`evaluate-quota-sweep-results`](evaluate-quota-sweep-results.md) | blocked | gaps-share-based-quota | direct-method-research |
| [`baselines-tabddpm`](baselines-tabddpm.md) | ready | -- | direct-method-research |

### Datasets

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`datasets-selection-rule`](datasets-selection-rule.md) | blocked | plan-direct-method-research | direct-method-research |
| [`ontology-per-dataset-releases`](ontology-per-dataset-releases.md) | blocked | refactor-evaluate-size-splits | direct-method-research |
| [`datasets-support2-frame`](datasets-support2-frame.md) | blocked | datasets-selection-rule | direct-method-research |
| [`datasets-nhanes-frame`](datasets-nhanes-frame.md) | blocked | datasets-selection-rule | direct-method-research |
| [`datasets-acs-frame`](datasets-acs-frame.md) | blocked | datasets-selection-rule, ontology-per-dataset-releases | direct-method-research |

### Composite and confirmation

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`evaluate-composite-arm`](evaluate-composite-arm.md) | blocked | evaluate-variant-dev-results, evaluate-quota-sweep-results | direct-method-research |
| [`evaluate-composite-dev-results`](evaluate-composite-dev-results.md) | blocked | evaluate-composite-arm | direct-method-research |
| [`evaluate-confirmatory-registration`](evaluate-confirmatory-registration.md) | blocked | evaluate-diabetes-control-results, evaluate-synthea-direct-results, evaluate-trace-attribution, baselines-tabddpm, datasets-support2-frame, datasets-nhanes-frame, datasets-acs-frame, evaluate-composite-dev-results | direct-method-research |
| [`evaluate-confirmatory-results`](evaluate-confirmatory-results.md) | blocked | evaluate-confirmatory-registration | direct-method-research |

### Follow-ups

| Unit | Status | Depends on | Milestone |
|---|---|---|---|
| [`ontology-auto-mapping`](ontology-auto-mapping.md) | blocked | generate-ancestor-shrunk-shares | direct-method-research |
| [`ontology-auto-mapping-results`](ontology-auto-mapping-results.md) | blocked | ontology-auto-mapping | direct-method-research |
| [`slm-lora-spike`](slm-lora-spike.md) | ready | -- | direct-method-research |
| [`slm-lora-trainer`](slm-lora-trainer.md) | blocked | slm-lora-spike | direct-method-research |
| [`slm-lora-results`](slm-lora-results.md) | blocked | evaluate-composite-dev-results, slm-lora-trainer | direct-method-research |

```mermaid
flowchart LR
    u1[plan-direct-method-research] --> u3[generate-no-mapping-ablation]
    u2[refactor-evaluate-size-splits] --> u3
    u1 --> u4[baselines-label-bootstrap]
    u1 --> u5[evaluate-prior-shift-comparator]
    u2 --> u5
    u3 --> u6[evaluate-control-runs-registered]
    u4 --> u6
    u5 --> u6
    u6 --> u7[evaluate-diabetes-control-results]
    u6 --> u8[evaluate-synthea-direct-results]
    u2 --> u9[evaluate-added-row-contradictions]
    u2 --> u10[gaps-is-a-conditional-fragments]
    u3 --> u11[generate-is-a-conditional-draws]
    u10 --> u11
    u9 --> u11
    u2 --> u12[ontology-diabetes-diagnosis-codes]
    u3 --> u13[generate-ancestor-shrunk-shares]
    u12 --> u13
    u2 --> u14[privacy-donor-attacks]
    u3 --> u15[privacy-donor-limits]
    u14 --> u15
    u15 --> u16[privacy-value-microaggregation]
    u3 --> u17[generate-coherent-free-fields]
    u9 --> u17
    u14 --> u17
    u2 --> u18[evaluate-trace-attribution]
    u11 --> u19[evaluate-variant-dev-results]
    u13 --> u19
    u15 --> u19
    u16 --> u19
    u17 --> u19
    u18 --> u19
    u2 --> u20[gaps-share-based-quota]
    u20 --> u21[evaluate-quota-sweep-results]
    u1 --> u23[datasets-selection-rule]
    u2 --> u24[ontology-per-dataset-releases]
    u23 --> u25[datasets-support2-frame]
    u23 --> u26[datasets-nhanes-frame]
    u23 --> u27[datasets-acs-frame]
    u24 --> u27
    u19 --> u28[evaluate-composite-arm]
    u21 --> u28
    u28 --> u29[evaluate-composite-dev-results]
    u7 --> u30[evaluate-confirmatory-registration]
    u8 --> u30
    u18 --> u30
    u22[baselines-tabddpm] --> u30
    u25 --> u30
    u26 --> u30
    u27 --> u30
    u29 --> u30
    u30 --> u31[evaluate-confirmatory-results]
    u13 --> u32[ontology-auto-mapping]
    u32 --> u33[ontology-auto-mapping-results]
    u34[slm-lora-spike] --> u35[slm-lora-trainer]
    u29 --> u36[slm-lora-results]
    u35 --> u36
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
  results unit runs it and records Runs, Result and Read against.
- **Which data is development and which is confirmatory is the plan's to say**
  ([`docs/RESEARCH.md`](../docs/RESEARCH.md)).

## Where results go

[`docs/EVIDENCE.md`](../docs/EVIDENCE.md): a new entry per question.

## Spend

As the experiments set's ([`experiments-index.md`](experiments-index.md#spend)).

## Conventions

As the experiments set's ([`experiments-index.md`](experiments-index.md#conventions)).
