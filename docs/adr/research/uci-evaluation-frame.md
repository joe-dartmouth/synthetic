---
status: accepted
topic: research
created: 2026-10-14
updated: 2026-10-15
---
# The UCI evaluation frame: Cleveland records, a binary label and a cell-stratified split

## Context

UCI Heart Disease is E1's evaluation setting, and the 2-pager defines underrepresentation as fewer
than 20 training records per age-sex subgroup-label combination (`docs/RESEARCH.md` section 2). The
2-pager names neither the archive's site, the label coding, the age bands, the handling of missing
values nor the train/test split, and every E1 number and the count-based gap list depend on all
five.

The archive holds four sites' `processed.*.data` files. In the archive whose size and sha256
`bko.datasets.sources` holds (measured 2026-10-14), Cleveland's has 303 records, 6 of them with a
`?` (4 in `ca`, 2 in `thal`); the Hungarian, Switzerland and VA files have a `?` in 293 of 294, 123
of 123 and 199 of 200 records. `num` grades disease from 0 to 4, and the archive's `heart-disease.names`
documents only absence (0) against presence.

## Decision Drivers

* **Must make "fewer than 20 training records" discriminate**: some cells under it, most over.
* **Must keep the test records out of every stage that shapes generation.**
* **Must reproduce byte for byte from the pinned archive**, across library upgrades.
* **Should follow the dataset's standard use**, so results compare with the literature.

## Decision

- **Cleveland only**: the frame is `processed.cleveland.data`.
- **Binary label**: `target` is 1 where `num` is 1 to 4 and 0 where it is 0; `num` is dropped.
- **Complete records only**: a record with a `?` is dropped before anything else reads the frame.
  A `?` in `age`, `sex` or `num` refuses the archive instead, since that record's cell is unknown.
- **Two age bands, split at 55**: `<55` and `>=55`, so 8 cells with the 2 sexes and 2 labels.
- **The split**: one record in five of every cell is held out, rounded half up, the lowest-ranked
  by the sha256 of `<seed>:<record_id>`; the seed is 0, and an experiment's configuration may fix
  others (`bko.evaluate.experiment.E1` does).
- **Training records only before Evaluate**: the cell counts, the BKO dataset fragment, count-based
  gaps, baselines and generators read the training split; only `bko.evaluate` reads the test split.

`bko.datasets.uci`, `bko.datasets.cells` and `bko.datasets.split` hold the values.

## Rationale

The names file says the Cleveland file is the only one ML researchers have used, and the other
three sites would need their missing values imputed, which puts model-made values in the real
baseline and makes site composition part of every cell.

The names file says Cleveland experiments distinguish presence (1 to 4) from absence (0). Five
labels would spread about 240 training records over 20 cells.

Dropping the 6 incomplete records costs 2% of the data. Keeping them would leave each baseline to
handle missing values its own way (SMOTE rejects them), an uncontrolled difference between methods.

With B age bands there are 4B cells sharing about 238 training records, so the mean cell falls to
19.8 at B = 3 and most cells drop under 20 by construction; the threshold discriminates only at
B = 2, and B = 1 has no age dimension. The cut sits at 55, near Cleveland's median age, so the two
bands are about equal in size. These choices were made with the cell counts in view: on the pinned
archive's training split, two bands at 55 flag 2 of 8 cells (women with heart disease in both
bands), three bands at 45 and 60 flag 6 of 12, and four at 45, 55 and 65 flag 11 of 16. No model
result was in view.

Stratifying on the full cell makes a cell's training count its size minus one fifth, rounded half
up, so the flagged set is the same for every seed and repeated holdouts in E1 do not move the gap
list. A cell of one or two records keeps them all in training, where scikit-learn's stratified
split raises on a one-record class. Ranking by sha256 depends on no library's random stream.

Test records that reach the gap list or the BKO's frequency weights would inflate any measured
improvement.

## Consequences

**Gained**: a gap list that is the same for every seed; a frame with no missing values and no label
leak; a split every unit recomputes identically from the pinned archive.
**Accepted**: 6 records and three sites unused; two coarse age bands; about five test records of
women with heart disease per split, so E1 needs repeated holdouts and intervals to resolve its
criteria.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Pooling the four sites | Missing values in almost every non-Cleveland record; needs imputation and confounds cells with site | 2026-10-14 |
| `num` as a five-class label | 20 cells for about 240 training records; only presence against absence is documented | 2026-10-14 |
| Imputing missing values | Model-made values in the real baseline; see Rationale | 2026-10-14 |
| Three or more age bands | Flag half or more of the cells on the pinned archive; see Rationale | 2026-10-14 |
| Stratifying on the label only | A cell of 24 or 25 records lands on either side of 20 depending on the seed | 2026-10-14 |
| scikit-learn `train_test_split` | Raises on a one-record stratum and brings the `eval` extra into the data stage | 2026-10-14 |
| A numpy `Generator` permutation | numpy does not keep `Generator` method streams stable across releases | 2026-10-14 |

## Enforcement

- Owner: `src/bko/datasets/`
- Pinned by: `tests/test_datasets_cells.py`, `tests/test_datasets_uci.py`

## Reversal Conditions

**The age bands are re-chosen before E1 runs if a re-pinned archive's training split flags no cell
or all 8 cells**; `python -m bko.datasets cells` shows which.

**The test share changes if E1's configuration fixes another one**; this record is then edited.

## Related

- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
- [src layout](../tooling/src-layout.md)
