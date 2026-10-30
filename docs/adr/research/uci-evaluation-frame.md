---
status: accepted
topic: research
created: 2026-10-14
updated: 2026-10-29
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

Dropping `slope`, `ca` and `thal`, where most of those `?` sit, leaves 740 complete records across
the four sites, and 661 once a cholesterol of 0 is read as missing (measured 2026-10-16): no living
patient has one, though `heart-disease.names` gives -9.0 as the missing-value code. All 123
Switzerland records and 49 of VA's 200 hold a cholesterol of 0. Among the 661, disease is present in
0.459 of Cleveland's records, 0.375 of Hungarian's and 0.794 of VA's; VA holds 5 women among its 97;
and 40 records are of women with heart disease, 25 of them Cleveland's.

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
- **Training records only before Evaluate**: the cell counts, the BKO dataset fragments, count-based
  gaps, baselines and generators read the training split; only `bko.evaluate` reads the test split.

- **One frame value**: `bko.datasets.frames.FrameSpec` holds what the stages read of a frame: its
  name, its features, which of them are numeric, fractional and categorical, the fractional
  fields' decimals, the codebook, each column's description, the numeric fields' ranges, the label
  with its codes and meanings, the age band edges, and the study and cohort the system prompt
  opens with. `bko.datasets.uci.UCI_FRAME` is UCI's, built from the module's constants.
  The generate stage, the evaluate stage's scoring helpers and `bko.validate`'s entry points take
  the frame as a required keyword, the call log requires the frame's name, and the cell helpers
  (`bko.datasets.split.split`, `bko.datasets.cells.cell_keys` and `cell_counts`,
  `bko.gaps.observe.record_cells`, `bko.fusion.join.join`) require the band edges, so none of
  them falls back to UCI's for a caller holding another frame. The baselines read the frame from
  their `Request`, whose field defaults to `UCI_FRAME`.
- **One frame value per named run**: `bko.evaluate.named_runs.NamedRun.dataset` chooses a
  `bko.evaluate.data.Data` (the records, the frame, the field rows its guidance reads, the
  dataset's name, the data use agreement it is held under and the sha256 of its records), and
  every stage of `score` reads its frame from that value.
- **The source's agreement**: each `bko.datasets.sources.Source` names the data use agreement its
  download is held under (`Agreement`, with the `sending` permission it grants, if any), None for
  open data; UCI Heart Disease's is None, and Synthea's frame carries None. Every Messages arm
  refuses data whose agreement records no `sending` permission (ADR
  [`research/cllm-arm-design`](cllm-arm-design.md), Egress).

`bko.datasets.uci`, `bko.datasets.cells`, `bko.datasets.frames` and `bko.datasets.split` hold the
values.

## Rationale

The names file says the Cleveland file is the only one ML researchers had used, the dataset's
standard use. Pooling the other sites costs E1 more than it adds. Imputing their missing values
would put model-made values in the real baseline. Dropping the three columns instead (measured
2026-10-16 on the 661 records, with the bands and split above) raises every holdout's test records
from 59 to 132, but the flagged subgroup's positive test records fall from 5 to 3, the gap request
falls from 20 rows to 10 in one cell, three predictors go, and site rides along with every cell.
Default TVAE fit on the pooled training rows generates almost no women and, in each of four
measurements, exhausted E1's 5 attempts at the `<55 female 1` request on most of the first 20
holdouts, so `score` stops early. Alone, VA (5 women) and Switzerland (1 record without disease
among the 46 complete once cholesterol is set aside) cannot test a classifier, and Hungarian (261
records) loses the same three predictors and flags 5 of 8 cells at the 55 edge, so the threshold
stops discriminating. The open heart-disease sets in this schema copy these sites:
[`fedesoriano-2021`](../../literature.md#datasets) is their records and Statlog's less duplicates,
and [`statlog-nd`](../../literature.md#datasets) repeats Cleveland records in every row checked
(checked 2026-10-16); the independent ones need a data request or change the schema and the label.
No arrangement of this archive adds many records of women with heart disease, the records E1's
secondary criterion lacks.

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

One frame value lets a second frame, such as a Synthea one, run through every stage without
editing it; a stage that bound UCI's constants at import could read no other. The cell is always
an age band, a sex and the label (`bko.datasets.cells`), so only the band edges vary between
frames. The keyword default keeps every UCI caller, and so every UCI request and score,
unchanged. Choosing the data by named run fixes a run's frame in code beside its experiment and
settings, so `score <run>` reads the same records every time, and one `Data` value hands every
stage the same frame, records and digest, which the run's logs then name.

The agreement sits on the `Source`, beside the archive's size and sha256, so a dataset's terms have
one home, and `Data` hands them to the stage that sends records' values to the API.

## Consequences

**Gained**: a gap list that is the same for every seed; a frame with no missing values and no label
leak; a split every unit recomputes identically from the pinned archive.
**Accepted**: 6 records and three sites unused; two coarse age bands; about five test records of
women with heart disease per split, so E1 needs repeated holdouts and intervals to resolve its
criteria, and no other frame from this archive adds records of women with heart disease.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Pooling the four sites | Needs imputation, or drops three predictors for fewer flagged test positives, half the gap rows, site in every cell and a TVAE that cannot fill the gap request; see Rationale | 2026-10-16 |
| Each site as its own frame | VA and Switzerland cannot test a classifier alone; Hungarian loses three predictors and flags 5 of 8 cells; see Rationale | 2026-10-16 |
| Another open dataset | The open sets in this schema copy these four sites, and the independent ones need a data request or change the schema and the label; see Rationale | 2026-10-16 |
| `num` as a five-class label | 20 cells for about 240 training records; only presence against absence is documented | 2026-10-14 |
| Imputing missing values | Model-made values in the real baseline; see Rationale | 2026-10-14 |
| Three or more age bands | Flag half or more of the cells on the pinned archive; see Rationale | 2026-10-14 |
| Stratifying on the label only | A cell of 24 or 25 records lands on either side of 20 depending on the seed | 2026-10-14 |
| scikit-learn `train_test_split` | Raises on a one-record stratum and brings the `eval` extra into the data stage | 2026-10-14 |
| A numpy `Generator` permutation | numpy does not keep `Generator` method streams stable across releases | 2026-10-14 |
| A `cell_columns` field on the frame | Every cell is an age band, a sex and the label, so only the band edges vary between frames; see Rationale | 2026-10-26 |
| A dataset flag on `score` | A flag would vary a run's values, which a named run fixes in code | 2026-10-27 |
| The frame on `Experiment` | `compare` reads only the experiment, which needs no frame | 2026-10-27 |
| A keyword defaulting to `UCI_FRAME` on the generate stage, the scoring helpers and the cell helpers | A caller holding the Synthea frame that omits it sends a UCI prompt and schema, or bands by UCI's edges, with no error | 2026-10-29 |
| A frame field on `Settings` | Every call log's header records the settings, and a run is refused when they differ from its log's, so the field would change what a logged run reads as | 2026-10-26 |
| The name `Frame` | The package calls a pandas DataFrame a frame throughout | 2026-10-26 |
| A map from dataset to its terms | A second home for a dataset's terms beside its `Source` | 2026-10-29 |

## Enforcement

- Owner: `src/bko/datasets/`
- Pinned by: `tests/test_datasets_cells.py`, `tests/test_datasets_uci.py`,
  `tests/test_datasets_frame.py`, `tests/test_evaluate_frame.py`, `tests/test_evaluate_egress.py`

## Reversal Conditions

**The age bands are re-chosen before E1 runs if a re-pinned archive's training split flags no cell
or all 8 cells**; `python -m bko.datasets cells` shows which.

**The test share changes if E1's configuration fixes another one**; this record is then edited.

**The frame is re-chosen if an open archive in this schema adds as many complete records of women
with heart disease as Cleveland's frame holds**; the counts in Context, taken with three predictors
dropped, show how far the other sites fall short.

## Related

- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
- [src layout](../tooling/src-layout.md)
- [The CLLM arms and the egress guard](cllm-arm-design.md)
