---
id: evaluate-second-classifier
title: A second downstream classifier
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: --
---

# A second downstream classifier

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

Every arm is scored with one logistic regression (`bko.evaluate.classifier`, ADR
[`research/e1-evaluation-protocol`](../docs/adr/research/e1-evaluation-protocol.md)). A linear
model may not exploit 20 added rows in one cell, so a null result on it does not show that the rows
carry nothing. Scoring `e1-3`'s arms with a second model costs no API call.

## Scope

**In.**

- A second downstream model beside the first: scikit-learn's histogram gradient boosting
  classifier at fixed, seeded settings, named in `bko.evaluate.classifier`; the criteria keep
  reading logistic regression, and the second model's measures are reported beside them with no
  verdict.
- `score --replay` over `e1-3`'s call log and `compare` write both models' measures: recall and
  AUC-PR over the whole test split and the flagged subgroups per arm, and each contrast's corrected
  interval, for both.
- ADR `research/e1-evaluation-protocol` records the second model and why the criteria still read
  the first.
- A new E1 Runs row and a Result addition in `docs/EVIDENCE.md`.

**Out.** Changing which model the criteria read; tuning either model.

## Acceptance

- Both models' measures regenerate from `score --replay` over `e1-3`'s call log and `compare`.
- `make check`.
