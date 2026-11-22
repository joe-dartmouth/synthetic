---
id: evaluate-tableshift-readmission
title: Reproduce TableShift's diabetes_readmission split on the pinned Diabetes 130 archive
status: ready
depends_on: --
blocks: evaluate-tableshift-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/tableshift-comparison
---

# Reproduce TableShift's diabetes_readmission split on the pinned Diabetes 130 archive

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

[`docs/RESEARCH.md`](../docs/RESEARCH.md#published-comparisons) runs four protocols beside
published numbers. [TableShift](https://arxiv.org/abs/2312.07577) publishes a readmission task on
the same Diabetes 130 data with a fixed in-domain and out-of-domain partition and baseline
numbers. The paper calls it Hospital Readmission; the code calls it `diabetes_readmission`. The
source is MIT, pinned at commit `fca9429814703a07e3902d005d46563a207b7f0a` of
[mlfoundations/tableshift](https://github.com/mlfoundations/tableshift).

## Scope

**In.**

- Read the task from that source, pinned at the commit: rows with missing race are dropped (2,273
  rows in the code); the label is any readmission (`!= "NO"`), the code's rule, where the paper's
  text says 30 days, so it is not E6's label; 47 predictors; out-of-domain is
  `admission_source_id == 7`; seed 264738 with 0.1 test and validation shares.
- An in-repo split that reproduces its partition on the pinned Diabetes 130 archive, as a new
  dataset name with `DIABETES_FRAME`'s rows unchanged, tested against the paper's Table 12: train
  34,288; validation 4,286; in-domain test 4,287; out-of-domain validation 5,664; out-of-domain
  test 50,968.
- The gap request each row-adding arm fills, as named cells of the in-domain training split.
- Named runs.
- The EVIDENCE Configuration, `not yet run`, quoting TableShift's Table 7 baselines, for example
  CatBoost at 0.659 in-domain and 0.618 out-of-domain accuracy.
- A new ADR under `docs/adr/research/` that records the split and what differs from TableShift.

**Out.**

- `tableshift` in the lock.
- Its other tasks.
- Generalised cell axes.

## Acceptance

- A test behind `--releases` checks the split's counts on the pinned archive against the
  documented ones.
- The gate test uses a fixture.
- The named runs, the ADR and the `docs/EVIDENCE.md` entry's Configuration, status `not yet
  run`, exist.
- `make check` passes.
