---
id: proposal-gap-detection
title: BKO and count-based gap detection
status: blocked
depends_on: proposal-mondo-do-fusion, proposal-datasets
blocks: proposal-llm-generation
parallel_safe: true
owner: Partiban
milestone: status-update-1
paid: false
adrs: research/bko-formal-gap-definition
---

# BKO and count-based gap detection

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

Detect: statements the fused ontology implies but the dataset fragment does not support are the gaps, and count-based gaps are evaluated separately ([RESEARCH, Detect](../docs/RESEARCH.md#detect), [two kinds of gap](../docs/RESEARCH.md#two-kinds-of-gap-evaluated-separately)). Status Update 1 asks for gap detection end to end.

## Scope

**In.**

- Add the dataset as a fragment weighted by observed frequency and list ungrounded statements in `src/bko/gaps/`.
- List count-based gaps from the cell counts.
- Run both on the Synthea ablation, where the removed cases are the ground truth.

**Out.** Generating records for the gaps.

## Acceptance

- Both gap lists reproduce with one command; the Synthea ablation's recovery of removed cases is recorded in `docs/EVIDENCE.md`.
- `make check`.
