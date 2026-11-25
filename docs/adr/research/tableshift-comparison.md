---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# TableShift's readmission split: reproduced in numpy on the pinned Diabetes 130 archive, scored as two datasets

## Context

[`docs/RESEARCH.md`](../../RESEARCH.md#published-comparisons) runs four protocols beside
published numbers. TableShift
([`gardner-2023`](../../literature.md#evaluation-and-failure-modes)) publishes a readmission task
on Diabetes 130, called Hospital Readmission in the paper and `diabetes_readmission` in the code,
with a fixed in-domain and out-of-domain partition and baseline accuracies (its Table 7) and split
sizes (its Table 12). The repository holds the same archive under `uci-diabetes-130`. The
`tableshift` package is not in the lock, and `bko.datasets` imports no scikit-learn, so the split
is rebuilt from TableShift's source. The harness splits records by a hash per holdout seed
(`bko.datasets.split.split`), which cannot give a published partition, and it scores one test
split per run.

## Decision Drivers

* **Must reproduce TableShift's partition row for row**, so a reading sits beside its published
  numbers on the same test records.
* **Must leave every existing dataset, frame and run folder byte for byte**: the `diabetes` frame,
  its field rows and `bko-direct`'s draw are re-read under identity checks.
* **Must draw with `bko-direct`**, the study's candidate, which needs a donor showing exactly the
  drawn set of concepts.
* **Must add no dependency**: neither `tableshift` nor scikit-learn in `bko.datasets`.

## Decision

1. **Source.** The task is read from `mlfoundations/tableshift` at commit
   `fca9429814703a07e3902d005d46563a207b7f0a` (MIT).
2. **Rows and label.** Rows whose `race` is `?` are dropped (2,273 rows; TableShift reads the
   CSV with `na_values="?"`, then `df.dropna(subset=["race"], inplace=True)`,
   `tableshift/datasets/diabetes_readmission.py:374`). The label is any readmission,
   `readmitted != "NO"` (`:377`), so it is not E6's 30-day label. Out of domain is
   `admission_source_id == 7` (`domain_split_ood_values=[7, ]`,
   `tableshift/configs/benchmark_configs.py:127`). No other row is dropped: TableShift's
   row-wise `dropna` finds nothing, since every categorical column is cast with
   `x.apply(str)` (`tableshift/core/features.py:28`) and the numeric columns hold no gap.
3. **Partition.** Rows stay in archive file order. The in-domain positions are split by
   `RandomState(264738).permutation(n)`: the first `ceil(0.2 n)` are held, the rest train. That
   held part, in its permuted order, is split again by a fresh `RandomState(264738)`: the first
   `ceil(0.5 n)` are the in-domain test, the rest validation. The out-of-domain positions take the
   first `ceil(0.1 n)` of a fresh permutation as validation and the rest as test. This is what
   scikit-learn's `train_test_split` does in TableShift's three calls
   (`tableshift/core/splitter.py:263-283`), written in numpy alone, with `ceil` taken of the same
   float products.
4. **Frame.** The frame holds TableShift's 47 features less the domain column, which TableShift
   drops after the split (`drop_domain_split_col: bool = True`, `tableshift/core/splitter.py:157`),
   so 46 model inputs. `age` is the decade index, `[0-10)` 0 to `[90-100)` 9, every bracket kept.
   `sex` is 1 for `Male` against every other value and `race` is 1 for `Caucasian` against every
   other value, TableShift's grouper (`tableshift/configs/benchmark_configs.py:129-130`). The
   diagnosis columns are under `bko.datasets.diagnoses.code_number`, and the lab and drug columns
   use the `diabetes` frame's codes. `weight`, `payer_code` and `medical_specialty` are coded by
   the position of each value among the kept rows' sorted values; `admission_type_id` and
   `discharge_disposition_id` keep their ids.
5. **Field rows.** The field rows are the lab rows only (`LAB_FIELD_ROWS`). `bko-direct` cannot
   draw a coded field of hundreds of exclusive concepts, since no donor shows a drawn set that
   names two codes for one field, so the diagnosis columns are free fields here.
6. **Datasets.** `tableshift-id` holds the in-domain train and the in-domain test, the test held;
   `tableshift-ood` holds the in-domain train and the out-of-domain test, the test held. The
   validation partitions are built and counted, never loaded.
7. **Holdouts.** Every holdout seed reuses the fixed partition, so a run's intervals read the arms'
   draw variance only.
8. **Gap request.** E1's rule on the in-domain training split with edges `(1, 2, 4, 6, 8)`: a cell
   with fewer than 20 training rows is filled to 20. It flags the cells the `--releases` test
   names.
9. **Runs.** Experiment P1, runs `p1-id` and `p1-ood`, with `decontaminate=False`, since the
   partition is TableShift's. Accuracy at the 0.5 cutoff is derived from recall, precision and the
   test prevalence.

**Differences from TableShift.** No tuning on validation; our classifiers and encodings; one-hot
coding without its 0.01 frequency floor (`PreprocessorConfig(min_frequency=0.01)`); partition
membership follows the same algorithm on the same file bytes, but TableShift itself was not run.
Table 7's figures are the best accuracy pair per model over TableShift's hyperparameter sweep, not
one fit.

## Rationale

A numpy permutation under the same seed is the draw `train_test_split` makes, so the partition
needs no new dependency, and Table 12's counts check it on the archive. Two datasets keep the
harness's one test split per run and give each published column, in-domain and out-of-domain, its
own run. Lab rows alone keep `bko-direct` drawing: the diagnosis rows of `diabetes-codes` name a
concept per code, and a draw of two codes for one field has no donor. Building a new frame rather
than reusing `DIABETES_FRAME` keeps TableShift's cohort, which Table 12 counts, and leaves the
`diabetes` frame untouched.

## Consequences

**Gained**: a published partition and baseline beside which the direct arm and its comparators are
scored, with no dependency added and no existing dataset moved.
**Accepted**: our scores are not TableShift's protocol (no validation tuning, other classifiers),
so the comparison is read beside Table 7, not as a replication of it. The intervals carry no split
variance. The diagnosis columns carry no ontology concept.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| `tableshift` in the lock | It pulls in torch, ray and its own preprocessing; the unit keeps it out | 2026-11-24 |
| scikit-learn in `bko.datasets` | The datasets stage stays free of it; numpy gives the same permutation | 2026-11-24 |
| The seed-based `split` | Its hash ranking per cell cannot give TableShift's partition | 2026-11-24 |
| `DIABETES_FRAME` | Its cohort (first encounters, adults, no hospice) does not give Table 12's counts | 2026-11-24 |
| One dataset with two test sets | The harness scores one test split per run | 2026-11-24 |
| The code rows of `diabetes-codes` as field rows | `bko-direct` finds no donor for a draw naming two codes for one field | 2026-11-24 |
| A new accuracy `Measure` | Accuracy at the cutoff follows from recall, precision and the test prevalence | 2026-11-24 |

## Enforcement

- Owner: `src/bko/datasets/diabetes.py`
- Pinned by: `tests/test_datasets_diabetes_pinned.py`

## Reversal Conditions

**A partition that stops reproducing**: the `--releases` test's counts differ from Table 12's on
the pinned archive. **A TableShift revision that changes the task**: a later commit of
`mlfoundations/tableshift` changes the row filter, the label, the domain value or the split
shares, and the study moves to it.

## Related

- [The diabetes-codes dataset keeps each ICD-9 diagnosis code and maps it through MONDO's equivalentTo xrefs](diabetes-diagnosis-codes.md)
- [The Diabetes 130 subgroup-removal design: the full adult frame, one thinned cell at the prevalence cutoff, every arm](diabetes-subgroup-removal-design.md)
