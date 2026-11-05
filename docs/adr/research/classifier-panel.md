---
status: accepted
topic: research
created: 2026-11-05
updated: 2026-11-05
---
# A classifier panel reads recorded runs under four more models, with calibration, and no verdict

## Context

Every run scores each arm with a logistic regression, which E1's criteria judge, and a histogram
gradient boosting model reported beside it (`bko.evaluate.classifier.MODELS`, ADR
[`research/e1-evaluation-protocol`](e1-evaluation-protocol.md)). The instructor's feedback on the
2026-11-04 status update asked for results under more classifiers, and for the literature on what
row-adding does to AUC-PR (unit `evaluate-more-classifiers`). E6 shows that the classifier
matters: at f = 1.0 `bko-direct` reaches removed-cell recall 0.5377 under the regression and
0.3591 under boosting ([`docs/EVIDENCE.md`](../../EVIDENCE.md#e6-diabetes-130-subgroup-removal)).
Row-adding also moves a model's predicted risk more than its ranking
([`docs/RESEARCH.md` section 7](../../RESEARCH.md#7-augmentation-versus-resampling)), and neither
recall nor AUC-PR reads that shift. Every recorded run must still replay byte for byte, and E1
ruled out a random forest as a judged model because its own seed moved subgroup recall by one of
5 positives.

Unless noted, every measurement below was taken on 2026-11-05 on E6 holdout 0 at f = 0.985
(54,177 training rows, 13,856 test rows, 1,250 removed rows, 34 columns), on one thread under
`VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1`. Test prevalence is 0.0901.

## Decision Drivers

* **Must leave every recorded run's bytes unchanged**, so `score`, `shared` and `rows` keep their
  models and their output.
* **Must make no API call and regenerate no rows**, reading each arm's fills from the store that
  `rows` writes (ADR [`research/rebuilt-rows-store`](rebuilt-rows-store.md)).
* **Must take each arm's weights**, since `class-weight` reaches the model through them.
* **Must be deterministic at one thread**, so a rerun of the reading reproduces it.
* **Should read calibration beside ranking**, since the literature's harm from rebalancing is a
  calibration harm.

## Decision

We will add a **classifier panel**: a random forest, extra trees, a multilayer perceptron and a
categorical naive Bayes, scored beside the regression and the boosting model by a separate `panel`
command, reported with no verdict.

- **The models** (`bko.evaluate.classifier.PANEL`, each value of `Model` suffixing its measure
  names as `hgb` does):
  - `rf`: `RandomForestClassifier(n_estimators=500, criterion="gini", max_features="sqrt",
    min_samples_leaf=1, bootstrap=True, class_weight=None, n_jobs=1, random_state=0)`.
  - `et`: `ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1,
    bootstrap=False, class_weight=None, n_jobs=1, random_state=0)`.
  - `mlp`: `MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
    alpha=1e-4, batch_size=200, learning_rate_init=1e-3, early_stopping=False, max_iter=15,
    random_state=0)`.
  - `nb`: `KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform", subsample=None)` on
    the measured fields and `OrdinalEncoder` over the codebook on the coded fields, then
    `CategoricalNB(alpha=1.0, min_categories=...)` with 10 categories per measured field and
    each coded field's codebook size.
  - The trees read measured fields unscaled and the perceptron reads them standardised; every
    model but naive Bayes one-hot encodes coded fields over the codebook.
- **MLP epochs**: fixed at 15, the epoch of lowest validation log-loss on holdout 0's `none`
  training split, split 90/10 stratified with seed 0, over epochs 1 to 30. The perceptron's
  `ConvergenceWarning` is recorded rather than raised (`FIXED_EPOCHS`), and its fit's log line
  gives `n_iter_`.
- **Calibration**: for every model, in the panel reading only, `brier`, mean((p - y)^2), and
  `calibration-in-the-large`, mean(p) - mean(y), per scope, each NaN on an empty scope, suffixed
  by model as `recall` is. `bko.evaluate.utility.utility(..., calibration=True)` writes them, so
  the default output is unchanged.
- **Opt-in per command**: only `panel` scores the panel and the calibration measures. Its
  `experiment` line logs `models=lr,hgb,rf,et,mlp,nb`; identity check 2 accepts it against a
  folder whose `models=` list is its prefix (ADR
  [`research/shared-test-split-reading`](shared-test-split-reading.md)).
- **Two stages**: stage 1, `rows`, rescores the folder at its own models and writes the store,
  which is also the self-check at the commit that reads. Stage 2, `panel` (module
  `bko.evaluate.panel`), scores from the store: `python -m bko.evaluate panel <run> --folder
  runs/<id> --rows runs/<id>-rows/rows.jsonl`. `docs/DEVELOPMENT.md` holds the guarded command.
- **Stored fills**: each BKO arm is a `StoredArm`, rebuilt from the folder's candidates file as
  `shared` and `rows` rebuild it. `none` and `class-weight` add no rows and are regenerated, since
  `class-weight` reaches the model through weights the store does not hold. `StoredBaseline`
  answers every other arm from `rows.jsonl`, for gap and replica fills alike, with unit weights
  and the logged params, refusing each attempt before the logged one as `StoredArm` does, so the
  harness retries as the run did and logs the same `arm` line. A partial fill, at its last attempt
  or a structural one, returns its stored rows as a whole fill. `panel` also refuses a store whose
  header names another folder or `log_sha256` than `--folder`'s (`bko.evaluate.rows_read.bind`).
- **Strict check 7**: `panel` applies the check `rows` does (ADR
  [`research/shared-test-split-reading`](shared-test-split-reading.md)), admitting as keys the
  folder lacks the panel models' suffixed measures and the calibration measures; any other
  difference exits 3.
- **Threads**: `panel` runs under `threadpool_limits(limits=1)` across every thread-pool API,
  beside `VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1` in its environment.
- **Seeds**: the forests and the perceptron take `random_state=0`, and their scores are reported,
  never judged.
- **Log**: the panel's log drops `shortfall` lines, since a stored fill runs no generator.

## Rationale

**The forests at 500 trees.** Over 10 seeds of the `real` fill, the random forest's removed-cell
recall has SD 0.0194 at 100 trees and 0.0142 at 500, and its AUC-PR over all records 0.0022 and
0.0006; extra trees read 0.0113 and 0.0096, and 0.0011 and 0.0004. A fit at 500 trees takes
14.5 s (random forest) and 19.6 s (extra trees) on one thread, against 2.9 s and 3.7 s at 100.
This answers E1's ruling: at E6's 317 test positives in the removed cell, the seed SD of cell
recall at 500 trees is 0.0142, and one fixed seed with no verdict reports the model without
letting a seed decide a criterion.

**The perceptron at 15 epochs.** On the `none` split the validation log-loss is lowest at epoch 15
(0.23905) and higher by epoch 30 (0.24415); on the `real` fill's training rows it is lowest at epoch 2 (0.29604,
0.30892 at epoch 30). A fixed count read from the `none` split is the same for every arm, where
early stopping would choose each arm's count from its own validation noise and weights. Trained to
convergence (276 to 329 epochs, 15.7 to 18.7 s) it overfits: AUC-PR over all records 0.1006 on
`none` and 0.1107 on `real`, against the regression's 0.1256 and 0.1429. At 15 epochs a fit takes
about 0.9 s.

**Categorical naive Bayes over 10 equal-width bins.** It reads AUC-PR 0.1222 on `none` and 0.1371
on `real`, Brier 0.0831 and 0.0819, with mean predicted risk 0.0732 and 0.0964 at a test
prevalence of 0.0901. `min_categories` fixes every field's category count, so a bin or code that
the training rows lack still has a category at prediction.

**Calibration.** Rebalancing a class leaves a regression's AUROC no higher and overestimates the
minority class's risk
([`vandengoorbergh-2022`](../../literature.md#augmentation-versus-resampling)), and over many
learners an uncorrected model calibrates equal to or better than a corrected one
([`carriero-2025`](../../literature.md#augmentation-versus-resampling)); learners also differ in
how well their scores serve as probabilities
([`niculescu-mizil-2005`](../../literature.md#evaluation-and-failure-modes)). The Brier score and
calibration-in-the-large read that shift where recall at one cutoff and AUC-PR do not. Reading
them in the panel only keeps every recorded run's bytes.

**Threads.** The boosting model already fits on one OpenMP thread; the panel's scalers, the
perceptron's matrix products and the regression call BLAS too, so the panel limits every API and
sets the environment variables that macOS Accelerate and OpenMP read before import.

## Consequences

**Gained**: E6's arms are read under six models with calibration, from the store, with no API call
and no regeneration, and every recorded run still replays byte for byte.
**Accepted**: the panel's forests take most of its compute (14.5 s and 19.6 s per fit at 500
trees); its scores hold one seed per model; a folder whose `rows` stage has not run, or whose
rescore no longer reproduces it, cannot be read.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Gaussian naive Bayes | Mean predicted risk 0.62 at a test prevalence of 0.09, Brier 0.4768 on `none` | 2026-11-05 |
| Early stopping for the perceptron | Its stopping epoch follows validation noise and depends on each arm's weights (Rationale) | 2026-11-05 |
| Training the perceptron to convergence | Overfits: AUC-PR 0.1006 on `none` and 0.1107 on `real` (Rationale) | 2026-11-05 |
| Forests at 100 trees | Seed SD of the random forest's cell recall 0.0194, against 0.0142 at 500 (Rationale) | 2026-11-05 |
| k-nearest neighbours, linear and quadratic discriminant analysis | None takes `sample_weight`, which `class-weight` needs | 2026-11-05 |
| A support vector classifier | Its probabilities need an internal cross-validated Platt fit, and its kernel fit grows faster than linearly in 54,177 training rows | 2026-11-05 |
| Judging a criterion on any panel model | E1's criteria judge the regression; a seeded model's verdict would depend on its seed | 2026-11-05 |
| Adding the panel to `score`'s default models | Every recorded run's scores would change | 2026-11-05 |

## Enforcement

- Owner: `src/bko/evaluate/classifier.py` (the models, `PANEL`, `FIXED_EPOCHS`),
  `src/bko/evaluate/utility.py` (the calibration measures), `src/bko/evaluate/panel.py` (the
  command, `StoredBaseline` and the keys it admits), `src/bko/evaluate/run_folder.py` (check 2's
  `models=` prefix rule, `_extends`), and `src/bko/evaluate/compare.py` with
  `src/bko/evaluate/unjudged.py` (each model's means and contrasts)
- Pinned by: `tests/test_evaluate_classifier.py`, `tests/test_evaluate_utility.py`,
  `tests/test_evaluate_panel.py`, `tests/test_evaluate_compare.py`

## Reversal Conditions

**The perceptron's validation log-loss on E6 holdout 0's `none` training split** (90/10
stratified, seed 0, epochs 1 to 30) reaches its minimum at an epoch other than 15 under the locked
scikit-learn: the epoch count is read again.

**The random forest's removed-cell recall SD over 10 seeds of the `real` fill at 500 trees exceeds
0.0284**, twice its measured 0.0142: one seed no longer reports the model, and the panel averages
seeds.

**A `panel` run exits 3 on the strict check 7** for a key outside the panel's suffixes and the
calibration measures: the scoring path moved, and the panel waits until `rows` reproduces the folder
again.

## Related

- [E1's evaluation](e1-evaluation-protocol.md)
- [A run is read beside another on the test split neither copies](shared-test-split-reading.md)
- [A recorded run's rows are rebuilt once](rebuilt-rows-store.md)
- [Diabetes 130 subgroup removal](diabetes-subgroup-removal-design.md)
