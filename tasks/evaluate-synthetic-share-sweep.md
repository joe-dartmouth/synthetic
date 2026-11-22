---
id: evaluate-synthetic-share-sweep
title: Register the synthetic-share ablation on E6's frame
status: blocked
depends_on: baselines-label-bootstrap, baselines-tabddpm
blocks: evaluate-synthetic-share-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/synthetic-share-ablation
---

# Register the synthetic-share ablation on E6's frame

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

[`docs/RESEARCH.md`](../docs/RESEARCH.md#synthetic-share-ablation) asks whether `bko-direct`'s
utility at each synthetic share is non-inferior to `real`, and whether its privacy readings are no
worse than each comparator's. The sweep needs the label-bootstrap control, which resamples real
rows, and the TabDDPM baseline.

## Scope

**In.**

- On E6's frame, training sets with 2%, 10%, 25%, 50% and 100% of rows synthetic. The total is
  held at the training split's size, real rows are kept by sha256 rank, and the test set is
  unchanged.
- Arms `bko-direct`, `ctgan`, `tvae`, `cart`, `bayes-net`, `tabddpm` and `label-bootstrap`
  (resampled real rows, the control for synthetic adds nothing), beside `real`.
- New named runs; no existing `RUNS` entry is edited.
- The readings of the research plan's synthetic-share ablation: the flagged measures, membership
  inference, NNDR and DCR.
- The two contrasts RESEARCH section 11 registers: `bko-direct` minus `real` on utility,
  non-inferior within the recall margin; `bko-direct` minus each comparator on each privacy
  reading, no worse.
- The EVIDENCE Configuration, `not yet run`.
- A new ADR under `docs/adr/research/` that fixes how a share's synthetic rows are requested (a
  replica of the training split's cells), what a method does when it cannot fill a cell, and which
  recall the margin reads at a share, since a replica fill of every cell has no removed cell.

**Out.**

- The LoRA classifier at each share: unit [`slm-lora-results`](slm-lora-results.md), at the mixes
  the trainer's ADR allows.

## Acceptance

- A test pins the arms, the shares and that the test split is untouched.
- The ADR names the recall the margin reads.
- The named runs and the `docs/EVIDENCE.md` entry's Configuration, status `not yet run`, exist.
- `make check` passes.
