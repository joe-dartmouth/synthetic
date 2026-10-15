---
status: accepted
topic: research
created: 2026-10-15
updated: 2026-10-15
---
# The baselines: each method fills a given quota of added rows, stratum by stratum

## Context

E1 compares BKO-guided augmentation against no augmentation, class weighting, SMOTE, ADASYN, CTGAN
and TVAE at equal row counts (`docs/RESEARCH.md` section 4). The 2-pager says neither what an equal
row count is for a method that adds no rows, nor which records added rows are drawn for, nor how a
seed reaches a generator.

On the pinned archive's training split (measured 2026-10-15), the label is close to balance, 128
records without disease against 110 with, while the two underrepresented cells hold 2 and 18 records.
SMOTE and ADASYN need at least two records in a stratum, and imbalanced-learn's ADASYN rounds each
record's share, so it misses a requested count. SDV's CTGAN and TVAE cannot condition on an age band,
seed their sampling after every fit, and seed no fit.

## Decision Drivers

* **Must let E1 compare like with like**: any method can be given the same added rows per stratum as
  any other arm, on the same real training rows.
* **Must read the training split only** (ADR `research/uci-evaluation-frame`).
* **Must emit records the evaluation frame could hold**: codebook codes, whole numbers where the
  archive has whole numbers.
* **Must reproduce from a seed on one machine**, and record what it cannot promise.
* **Must leave the open readings, and so the allocation E1 compares at, to E1's configuration.**

## Decision

- **Rows means rows added.** A request names the rows each stratum gains. A method fills exactly the
  request it is given or is refused; it is never topped up from another method. The real training
  rows are unchanged.
- **A stratum is a cell or a label.** The command's default request water-fills its row count over
  the cells: each row goes to the non-empty stratum holding the fewest records so far, ties in the grid
  order of `bko.datasets.cells`. E1's configuration fixes the strata and rows it compares at.
- **The controls add nothing.** No augmentation returns the training split as it is. Class weighting
  weights each record n / (k * n_c) over the labels, scikit-learn's balanced heuristic, computed in
  the stage.
- **SMOTE is SMOTE-NC**, run once per stratum against the rest of the split with
  k = min(5, records - 1). A stratum of fewer than 2 records is refused.
- **ADASYN runs the same way on one-hot coded predictors**, each block decoded to its largest entry.
  It is asked for ceil(records / 2) more rows than the quota, and the quota is kept by sha256 rank.
- **SMOTE and ADASYN z-score the measured predictors** with the training split's mean and standard
  deviation for the neighbour search; restored values round to the archive's precision.
- **CTGAN and TVAE are SDV's synthesizers at SDV's defaults** on explicit metadata, on the CPU. Each
  draws unconditioned batches and keeps the rows whose own values fall in a stratum still short, until
  every quota is met or a cap on drawn rows refuses the run.
- **Seeds**: imbalanced-learn takes one seed per stratum, derived from the run's seed. CTGAN and TVAE
  fit with numpy's and torch's global generators seeded from it and one torch thread, all restored
  afterwards; SDV seeds sampling itself after every fit.
- **Output**: in process, the added rows (the evaluation frame's columns and dtypes, indexed -1, -2,
  ...) and one weight per training record. The command writes the training set they make to stdout
  as CSV and writes nothing under `data/` but the snapshot `bko.datasets` fetches.

`bko.baselines.request`, `bko.baselines.oversample` and `bko.baselines.synthesizers` hold the values.

## Rationale

A request that every method fills exactly makes "equal row counts" a property of the code rather than
of each run's bookkeeping, at whatever allocation E1 chooses. Cells are the command's default because
the underrepresented cells are what the project sets out to fill; the label is one flag away, since
RESEARCH section 4 leaves the reading open.

Running each oversampler on one stratum against the rest keeps its neighbours, and so every
interpolated row, inside the stratum: a cell's sex and label are constant, and its ages lie between
two of its own. SMOTE-NC takes each coded predictor from the neighbours, so codes stay in the
codebook. ADASYN has no such variant; one-hot decoding gives each row one parent's code, where
rounding a code would invent a third. Without z-scoring, cholesterol's spread decides every
neighbour.

SDV's defaults are the cited form of CTGAN and TVAE; on about 240 records they give 300 optimizer
steps. SDV's own conditional sampling is rejection sampling too, cannot name an age band, and reports
a shortfall as a warning; drawing and classifying rows by their own values is the same method with a
hard failure. Seeding the global generators is the only way to reach the fit without SDV's private
methods, and one thread on the CPU removes the device and the reduction order from the result.

Writing files under `data/` would give the harness nothing it lacks: it calls `augment` per split.

## Consequences

**Gained**: exact rows per stratum for any request; one contract the harness reads for every arm;
refusals that name the method and the stratum.
**Accepted**: SMOTE on the 2-record cell interpolates between its two records only; ADASYN's
per-stratum density counts same-label records of other cells as the majority; SDV's defaults train
for few steps on this split; cross-platform identity of CTGAN and TVAE output is not promised, so the
command logs its output's sha256; `make setup` downloads `torch` on every machine; importing SDV
creates SDV's user data directory.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Allocation inside each method (SMOTE's `auto`) | The request says where rows go, so an allocation E1 picks applies to every method alike | 2026-10-15 |
| Plain SMOTE on the coded predictors | Interpolates codes into values outside the codebook | 2026-10-15 |
| Rounding ADASYN's interpolated codes to the nearest code | Invents a third category between two parents | 2026-10-15 |
| SDV's `sample_from_conditions` | Rejection sampling that cannot name an age band and only warns on a shortfall | 2026-10-15 |
| SDV's private `_set_random_state` for a per-run sampling seed | A private method; the fit seed already varies the rows | 2026-10-15 |
| Cell-level balanced class weights | Weights the 2-record cell 238 / (8 * 2); not the class weighting the 2-pager names | 2026-10-15 |
| scikit-learn's `compute_class_weight` | scikit-learn is declared only in `eval`, for a one-line formula | 2026-10-15 |
| Writing baseline datasets under `data/` | No consumer reads them; the harness calls `augment` per split | 2026-10-15 |
| Tuning CTGAN and TVAE inside the stage | An inner split of 238 records leaves the 2-record cell nothing to validate on | 2026-10-15 |
| Cross-platform byte identity | torch does not promise it across platforms | 2026-10-15 |

## Enforcement

- Owner: `src/bko/baselines/`
- Pinned by: nothing -- convention only

## Reversal Conditions

**The command's default strata become the label if E1's configuration fixes the strata at the
label**; this record is then edited.

**CTGAN and TVAE leave SDV's defaults if E1's configuration fixes other training parameters**; this
record and `bko.baselines.synthesizers` change together.

**The fit stops being seeded through the global generators if SDV's synthesizers accept a fit seed.**

## Related

- [The UCI evaluation frame](uci-evaluation-frame.md)
- [Python 3.14, without synthcity](../tooling/python-314-without-synthcity.md)
- [uv, ruff and ty](../tooling/uv-ruff-ty.md)
- [src layout](../tooling/src-layout.md)
