---
id: evaluate-second-classifier
title: A second downstream classifier
status: ready
depends_on: --
blocks: generate-prompt-training-bounds, generate-effort-sweep
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
model may not exploit the 18 and 2 rows E1's gap fill adds to its two cells, so a null result on it
does not show that the rows carry nothing. Scoring `e1-3`'s arms with a second model costs no API
call, and it must land before any unit changes what a call sends, since `e1-3`'s log replays only
at a tree that sends its requests.

## Scope

**In.**

- A second downstream model beside the first: scikit-learn's histogram gradient boosting
  classifier with early stopping off, so no seed moves a score (the ADR rules out a random forest
  because its seed does), and a leaf size below the gap fill's smallest cell, every setting fixed
  in `bko.evaluate.classifier`; the criteria keep reading logistic regression, and the second
  model's measures are reported beside them with no verdict.
- `score --replay` over `e1-3`'s call log and `compare` write both models' measures: recall and
  AUC-PR over the whole test split and the flagged subgroups per arm, and each contrast's corrected
  interval, for both.
- ADR `research/e1-evaluation-protocol` records the second model, its settings and why the
  criteria still read the first.
- A new E1 Runs row and a Result addition in `docs/EVIDENCE.md`, noting that the replay's scores
  file differs from `e1-3`'s by the added rows.

**Out.** Changing which model the criteria read; tuning either model.

## Acceptance

- Both models' measures regenerate from `score --replay` over `e1-3`'s call log and `compare` at
  the landing commit.
- `make check`.
