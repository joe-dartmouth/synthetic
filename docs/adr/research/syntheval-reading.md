---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# UCI Heart Disease fills scored under SynthEval's metrics, outside the lock

## Context

[`docs/RESEARCH.md`](../../RESEARCH.md#published-comparisons) runs four protocols beside
published numbers. SynthEval ([`lautrup-2024`](../../literature.md#evaluation-and-failure-modes))
is a metric library for tabular synthetic data, and Lautrup et al.
([`lautrup-2026`](../../literature.md#tabular-generators)) report heart-disease results under it
in figures, with MIT code at
[disjoint-synthetic-data-generation](https://github.com/notna07/disjoint-synthetic-data-generation).
Their heart-disease files (`experiments/datasets/heart_train.csv`, 242 rows, and
`heart_test.csv`, 61 rows) recode UCI's processed Cleveland file. A `rows` store
(ADR [`research/rebuilt-rows-store`](rebuilt-rows-store.md)) holds every fill of a recorded run,
checked against the run's log.

## Decision Drivers

* **Must score our arms under the metrics Lautrup et al. report**, so a reading sits beside their
  heart-disease figures.
* **Must keep the gate runnable with nothing but the interpreter and the lock.**
* **Must score fills the harness already checked**, not fills drawn again.
* **Must spend no money.**

## Decision

1. **The environment.** The scratch-lock probe, run 2026-11-24 in a detached worktree of
   `origin/main` at `15da6ed`: `syntheval = ["syntheval==1.7.3"]` added as an extra to
   `pyproject.toml`, then `uv lock`. It exited 0, resolving 141 packages and adding 10
   (contourpy, cycler, fonttools, kiwisolver, lightgbm, matplotlib, pcametric, pyparsing,
   seaborn, syntheval); `git diff uv.lock` changed no existing package's version line. SynthEval
   resolves inside the lock, and it still runs outside it, under ADR
   [`tooling/benchmark-tools-outside-the-lock`](../tooling/benchmark-tools-outside-the-lock.md),
   because LightGBM, which it imports, loads Homebrew's `libomp` at import on macOS.
2. **The input.** Each arm's replica fill of a holdout, read from the run's `rows` store, is the
   synthetic table. It is scored against that holdout's training split as the real data and the
   test split the run scored as the holdout data: when the run decontaminates
   (`Experiment.decontaminate`), the test split without each record some stored fill of the
   holdout, gap or replica, copies, under the rule the run's scoring uses
   (`bko.evaluate.copies.decontaminated`). A gap fill is not scored.
3. **The metric set.** The keyword dictionary Lautrup et al.'s code passes to `SynthEval.evaluate`
   at commit `a07cd9c007b58bee37b0837cfdc65252303afadc`
   (`experiments/auxiliaries/03_baselines.py:103-112`, re-read 2026-11-24): `pca`, `h_dist`,
   `corr_diff` with `mixed_corr` true, `auroc_diff` with `model` `rf_cls`, `cls_acc` with
   `F1_type` `macro`, `eps_risk`, `dcr`, and `mia` with `num_eval_iter` 5.
4. **The seed.** SynthEval 1.7.3's membership-inference metric draws its splits and samples from
   NumPy's global random state, unseeded. The script seeds Python's and NumPy's global state with
   the fill's holdout seed before scoring, so one store gives one `syntheval.csv`.
5. **The reference split.** E1's holdouts. Regenerating Lautrup et al.'s numbers on their own
   242 and 61 rows is the results unit's (`tasks/evaluate-syntheval-results.md`).
6. **The `hd` mapping.** `bko.datasets.hd.from_hd` reads their files as records of the UCI
   frame. A match of every row of theirs to the pinned `processed.cleveland.data` on age, sex,
   `trestbps`, `chol`, `thalach` and `oldpeak` (2026-11-24; the key is unique among UCI's 303
   records) gives the maps from their codes to UCI's: `cp` 0 to 4, 1 to 2, 2 to 3, 3 to 1;
   `restecg` 0 to 2, 1 to 0, 2 to 1; `slope` 0 to 3, 1 to 2, 2 to 1; `thal` 1 to 6, 2 to 3,
   3 to 7. A missing `ca` is coded 4 and a missing `thal` 0. `from_hd` checks every code of every
   row, then drops a row holding either missing code. Their `target` is 1 where UCI's `num` is 0,
   so ours is 1 minus theirs. UCI record 303, which lacks `ca`, appears in both files, and UCI
   record 185, a complete record, in neither. Each of the 296 rows `from_hd` keeps (237
   training, 59 test) equals its UCI record in every column.
7. **The register.** [P2](../../EVIDENCE.md#p2-uci-heart-disease-scored-by-syntheval) holds the
   Configuration.
8. **The arms.** Free arms only: the comparison set, the classical synthesizers, the Bayesian
   network, TabDDPM, the label bootstrap and `bko-direct`.

## Rationale

The replica fill is the one fill of each arm that is a synthetic table of the training split's
whole shape, which is what SynthEval's metrics compare against real data; a gap fill holds a few
targeted rows. A `rows` store's fills are the rows the run's `arm` lines hash, so reading them scores the fills
the run scored, and scoring them against the decontaminated test split scores them against the
records the run's classifiers were tested on. A seeded script makes the reading a function of the
store.
Scoring on E1's holdouts keeps the reading inside the harness's split and its many holdouts; their
single split is regenerated separately, so the two readings sit side by side.

## Consequences

**Gained**: every free arm scored under the metrics a published comparison uses, from fills
checked against their run's log, with no dependency added to the lock.
**Accepted**: the reading needs `make syntheval-env`, which on macOS needs `brew install libomp`.
Our split is not theirs, so the reading sits beside their figures rather than replicating them.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A locked extra for SynthEval | LightGBM loads `libomp` at import on macOS, so every gate machine would need it (ADR [`tooling/benchmark-tools-outside-the-lock`](../tooling/benchmark-tools-outside-the-lock.md)) | 2026-11-24 |
| Their split as the harness split | One split gives no interval, and the harness's holdouts already fix E1's split | 2026-11-24 |
| Scoring gap fills | A few targeted rows are not a synthetic table | 2026-11-24 |
| A paid arm | The unit spends no money | 2026-11-24 |

## Enforcement

- Owner: `src/bko/datasets/hd.py`, `src/bko/evaluate/rows_read.py`,
  `src/bko/evaluate/syntheval_inputs.py` (`METRICS`, `fills`), `src/bko/evaluate/syntheval_reading.py`,
  `src/bko/evaluate/runs_published.py` (`P2`), `bench/syntheval/score.py`
- Pinned by: `tests/test_datasets_hd.py` (the maps, every code read as UCI's, the inverted target,
  the missing codes dropped and counted, a code outside a map refused, in a row with a missing
  code too), `tests/test_evaluate_rows.py`, `tests/test_evaluate_syntheval.py` (each replica fill
  with its holdout's splits, the decontaminated test split, the metric set, P2's configuration,
  the seed passed, the command's CSV and log lines), `tests/test_named_runs_slices.py` (the run
  names in order); outside the gate, `bench/syntheval/tests/test_score.py` (one seed, one JSON)

## Reversal Conditions

**SynthEval joins the lock**: ADR
[`tooling/benchmark-tools-outside-the-lock`](../tooling/benchmark-tools-outside-the-lock.md)'s
reversal condition holds. **Lautrup et al. change their metric set**: a later commit's
`03_baselines.py` passes another dictionary.

## Related

- [Benchmark tools outside the lock](../tooling/benchmark-tools-outside-the-lock.md)
- [A recorded run's rows are rebuilt once and stored](rebuilt-rows-store.md)
- [TableShift's readmission split](tableshift-comparison.md)
