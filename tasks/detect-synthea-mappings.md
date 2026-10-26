---
id: detect-synthea-mappings
title: Map more Synthea conditions to MONDO
status: ready
depends_on: --
blocks: evaluate-synthea-downstream
parallel_safe: true
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
scope gives every detector pairs outside the truth to flag or not, and the Synthea downstream
experiment needs the same concepts as features.

## Scope

**In.**

- Mapped rows for the Synthea SNOMED CT disorder codes of hypertension, type 2 diabetes,
  hyperlipidemia, myocardial infarction and stroke, each to a MONDO class, sourced under the
  conventions of ADR
  [`research/ontology-mapping-design`](../docs/adr/research/ontology-mapping-design.md) and
  checked by `python -m bko.ontology check`; a code with no MONDO class is recorded as excluded
  with its reason, and no finding or situation code joins.
- E2 rerun over the richer scope, with its Configuration, a Runs row and the Result in
  `docs/EVIDENCE.md`, and the `uci` reading shown unchanged by `cmp` of its output before and
  after. The `condition` ablation still removes ischemic heart disease patients whole, so the
  conditions those patients also hold leave with them; the truth then holds statements the
  remaining patients still ground, such as `is_a` ancestors the new concepts share with
  myocardial ischemia, which no grounding definition can recover, and the Result reads that
  shortfall as the richer scope's consequence, not a defect.

**Out.** Ablating another condition; HPO rows for Synthea observations.

## Acceptance

- `uv run --no-sync python -m bko.ontology check` passes on the pinned releases with the new rows,
  and a test pins that each of the five named codes has a row.
- `uv run --no-sync python -m bko.gaps synthea` prints the reading over every mapped concept,
  identical on a second run, recorded in E2's entry.
- `make check`.
