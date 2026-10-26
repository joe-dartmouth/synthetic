---
id: detect-synthea-mappings
title: Map more Synthea conditions to MONDO
status: ready
depends_on: --
blocks: evaluate-synthea-downstream
parallel_safe: false
owner: Partiban
milestone: final-report
paid: false
adrs: --
---

# Map more Synthea conditions to MONDO

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

E2 maps one Synthea code, ischemic heart disease, to one MONDO concept with 9 statements, so every
pair count in its Result is a multiple of 9, the `cell` ablation is one cell's decision, and the
frequency-aware score has nothing to be wrong about
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e2-synthea-gap-detection-ablation), Limits). A richer
truth needs more mapped concepts, and the Synthea downstream experiment needs them as features.

## Scope

**In.**

- Mapped rows for the Synthea SNOMED CT codes of hypertension, type 2 diabetes, hyperlipidemia,
  myocardial infarction and stroke, and any further condition the pinned population shows in at
  least one in twenty patients, each row sourced under the conventions of ADR
  [`research/ontology-mapping-design`](../docs/adr/research/ontology-mapping-design.md) and
  checked by `python -m bko.ontology check`.
- E2 rerun over the richer scope: the `uci` reading unchanged, the `synthea` reading with every
  mapped concept's statements in the truth; its Configuration, a Runs row and the Result in
  `docs/EVIDENCE.md`. The `condition` ablation still removes ischemic heart disease cases; the
  other concepts give the detectors cells and statements the removal does not touch.
- `parallel_safe: false`: it rewrites E2's entry and `bko.gaps.experiment`, which
  [`detect-partial-removal`](detect-partial-removal.md) also edits, so it lands alone.

**Out.** Ablating another condition; HPO rows for Synthea observations.

## Acceptance

- `uv run --no-sync python -m bko.ontology check` passes on the pinned releases with the new rows.
- `uv run --no-sync python -m bko.gaps synthea` prints the reading over every mapped concept,
  identical on a second run, recorded in E2's entry.
- `make check`.
