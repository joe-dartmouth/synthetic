---
id: evaluate-more-classifiers
title: Score E6 under more classifiers, and cite what row-adding does to AUC-PR
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: research/classifier-panel
---

# Score E6 under more classifiers, and cite what row-adding does to AUC-PR

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

The instructor's feedback on the 2026-11-04 status update asked for two things:
- results under classifiers beyond logistic regression and histogram gradient boosting
  (`bko.evaluate.classifier.Model`);
- citations for the claim that row-adding methods lower AUC-PR on the group they fill.

E6 shows why the classifier matters. At f = 1.0, `bko-direct` reaches removed-cell recall 0.5377
under LR and 0.3591 under HGB, and the copula's falls from 0.3507 under LR to 0.0451 under HGB
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)). E4 and E6
report the AUC-PR cost, but neither places it in the literature on oversampling and
calibration.

## Scope

**In.**

- **Phase 1, literature.** Find and register in [`docs/literature.md`](../docs/literature.md) the
  work on how oversampling and synthetic minority rows shift a classifier's scores and affect
  precision-recall measures: SMOTE's and its successors' evaluations, and the class-imbalance
  calibration literature. Add a paragraph to [`docs/RESEARCH.md`](../docs/RESEARCH.md) that
  reads E4's and E6's AUC-PR results against it.
- **Phase 2, the panel.**
  - Add to `bko.evaluate.classifier` a random forest, extra trees, a multilayer perceptron and
    naive Bayes over binned fields, all from scikit-learn, so no new dependency is needed.
  - Each is opt-in per command (`panel`), so no existing run's scores change and every recorded run
    still replays byte for byte.
  - A new ADR under `docs/adr/research/` fixes each model's settings and why.
- **Phase 3, the reading.** Rescore E6 at both fractions under the panel, on each arm's rows
  read from the store that `python -m bko.evaluate rows` writes, with no API call.
  Each set of rows must hash to its `arm` line's `sha256` in the slice's `scores.log`.
  `evaluate-membership-inference` reads the same store. The result is a new
  `docs/EVIDENCE.md` entry.

**Out.**
- New dependencies (XGBoost, LightGBM, CatBoost).
- Tuning any classifier per arm.
- Changing E1's judged classifier.

## Acceptance

- The panel's models pass a test that fits each on a toy frame and scores it, and a test shows an
  existing named run's scores unchanged.
- The E6 reading comes from a committed command with no API call, its rows matching the logged
  `sha256` on every holdout read.
- The literature paragraph cites registered works only, and `tests/test_docs_citations.py`
  passes.
- `make check`.
