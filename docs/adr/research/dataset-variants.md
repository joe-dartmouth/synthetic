---
status: accepted
topic: research
created: 2026-11-26
updated: 2026-11-26
---
# A mapping change or a field projection is a new dataset name over its base

## Context

A named run chooses its records, frame and field rows through one `Dataset` value
(`bko.evaluate.data`; ADR [`research/uci-evaluation-frame`](uci-evaluation-frame.md) records why
the run, not a flag, chooses them). The matched knowledge controls need Diabetes 130's records on
a subset of its columns, the fields the mapping table can generate (ADR
[`research/matched-knowledge-controls`](matched-knowledge-controls.md), Decision 10), and later
units need the same records under another mapping table. Every existing run of record reads its
dataset's records, frame and rows as they are, and its logs name the dataset.

## Decision Drivers

* **Must leave every existing dataset's records, frame and rows byte for byte**, so runs of record
  re-read under identity checks.
* **Must share the base's split**, so a variant's holdouts pair with the base's.
* **Must name the variant in the run's logs**, so a folder says which data it scored.
* **Must keep one place that says what a dataset is.**

## Decision

1. **A variant is a dataset name.** A mapping change or a field projection of an existing dataset
   is a new `Dataset` value, held in `VARIANTS`, a read-only map from each variant to its
   `Variant`: its `base`, its `rows` (None keeps the base's) and whether it is `supported`,
   projected onto `supported_frame`. `frame`, `field_rows`, `dataset_releases` and `load` read a
   variant through its `VARIANTS` entry alone, so its match case in each holds no rule of its own.
2. **Records, releases and source name are the base's.** Loading a variant reads its base's
   records, then swaps the rows or projects the records; `Data.name` stays the base's source name,
   and `dataset_releases` of a variant is its base's.
3. **The frame.** A row swap keeps the base's frame object; a supported variant's frame is one
   module-level `FrameSpec`, built once at import (`diabetes-supported` for
   `Dataset.DIABETES_SUPPORTED`).
4. **The field rows.** A row swap gives the variant's rows; a supported variant keeps the base's.
5. **Logging.** The `data` line's `dataset=` is the variant's value, and loading a variant logs no
   line for its base.
6. **The first variant** is `diabetes-supported`, `diabetes` on `age`, `sex`, the label and every
   field of a `Codes` clause of a non-label row.

## Rationale

A dataset name is already the unit a run, its folder and its logs carry, so a variant needs no
new field on `Experiment` and no new reader. Reading the base's records and splitting them as the
base does keeps every holdout's training and test rows equal to the base's on the columns they
share, which is what a matched comparison needs. One frame object per variant keeps frame
identity checks working as they do for every other dataset.

## Consequences

**Gained**: a projection or a mapping change runs under the existing run, split and logging code;
existing datasets and their runs stay as they are.
**Accepted**: each variant is a `Dataset` member, a `VARIANTS` entry and a case in each reader's
match that defers to the entry; a variant's digest differs
from its base's whenever its records do, so its folders never merge with the base's.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Editing a base's rows or frame | Moves every run of record that reads the base | 2026-11-26 |
| A run-time flag selecting columns or rows | One run name would then name two datasets, and its logs would not say which | 2026-11-26 |
| An `Experiment.features` field | Needs placeholder values for every other experiment and a second reader of the frame | 2026-11-26 |

## Enforcement

- Owner: `src/bko/evaluate/data.py`
- Pinned by: `tests/test_evaluate_variants.py` (the supported frame's fields and its one object,
  the base's field rows, the base split, the `data` line naming the variant, the base's source
  name and releases, no base line when a variant loads, and a row swap's base frame and own rows
  through a patched `VARIANTS` entry)

## Reversal Conditions

**A variant's split differs from its base's**: the train and test index sets of
`diabetes-supported` and `diabetes` at one holdout seed are unequal.

## Related

- [The matched knowledge controls change one input of bko-direct's draw at a time](matched-knowledge-controls.md)
- [The UCI evaluation frame: Cleveland records, a binary label and a cell-stratified split](uci-evaluation-frame.md)
