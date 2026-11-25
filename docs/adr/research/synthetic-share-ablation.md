---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# The synthetic-share ablation on E6's frame

## Context

[`docs/RESEARCH.md`](../../RESEARCH.md#synthetic-share-ablation) rebuilds each row-adding arm's
training set with a share of its rows synthetic, the total held at the training split's size and
the test set unchanged. It asks whether `bko-direct`'s utility at each share is non-inferior to
`real` within the recall margin of the control reading rule, and whether its privacy readings are
no worse than each comparator's.

E6 ([`docs/EVIDENCE.md`](../../EVIDENCE.md#e6-diabetes-130-subgroup-removal)) removes a share of
one cell's training rows, `E6_CELL`, `(6-7, male, 1)`, and reads the margin's recall on that
cell's test rows. A sweep that replaces rows in every cell has no removed cell, so the request,
the arm that cannot fill a cell and the recall the margin reads all need a rule.

`bko.datasets.split` already ranks records per cell by sha256 under a tag: the split itself and
`subsampled`, whose `kept_count` keeps at least two rows of every cell.

## Decision Drivers

* **Must hold the total at the training split's size** at every share.
* **Must leave the test split untouched**, so every share scores the same test rows.
* **Must read the margin on the test rows it was calibrated on.**
* **Must add no score row to any other run**: `rows` refuses a key only the rescore holds.
* **Must spend no money.**

## Decision

1. **The shares.** 1/50, 1/10, 1/4, 1/2 and 1, one named run each (`e18-two`, `e18-ten`,
   `e18-quarter`, `e18-half`, `e18-all`), each on E6's holdouts in ten slices.
2. **The dropped rows.** Per cell of `n` training rows, `floor(n * share + 1/2)` real rows are
   dropped, with no floor: the lowest under `bko.datasets.split`'s ranking with the tag
   `synthetic-share:`. The count rises with the share and the ranking is one order per holdout,
   so the dropped sets nest across shares. Share 1 drops every row.
3. **The request.** Each arm is asked for exactly the dropped count per cell, a replica of those
   cells, leaving out each cell that drops none, so the kept rows plus a full fill hold the
   training split's size. The request is a gap request, so its reader rebuilds it from the whole
   training split.
4. **What a generator fits on.** Every generator fits on the whole training split at every share,
   so a fill can carry values of the rows it replaces. Membership inference counts the whole split
   as members, so it measures that.
5. **The training set.** The classifier trains on the kept rows plus the fill. `real` adds the
   dropped rows back, so it trains on the whole split at every share. The test split is untouched.
6. **A short arm.** An arm short of a cell's count scores the partial fill, and its `rows` measure
   shows the realised count. A fill that leaves the training set empty or of one label, as a short
   fill at share 1 may, stops the run with an error naming the holdout and the arm.
7. **The recall the margin reads.** Recall on E6's cell, flagged by name
   (`Experiment.flagged_cell`), the same test rows the margin was calibrated on. All-rows recall
   is reported beside it with no margin. Since the run names its flagged cell, it also writes
   `auc-roc` on both scopes.
8. **The arms.** `real`, `bko-direct`, `ctgan`, `tvae`, `cart`, `bayes-net`, `tabddpm` and
   `label-bootstrap`; `bko-direct` the candidate, every other arm but `real` a comparator, no
   replica and no criterion. `prior-shift` and `class-weight` are refused under a share: their
   offsets and weights read the whole training split, not the kept rows. So is `bko-tree`: its
   code-share error reads the removed rows, and a share removes none.
9. **Privacy.** The readings are over the synthetic rows against the whole split, with
   `bko-direct` the reference arm, so each comparator's reading is its contrast with
   `bko-direct`, and E6's cell the cell scope.

## Rationale

Dropping by sha256 rank under its own tag reuses the split's ranking, so a share's dropped rows
are fixed per holdout and need no stored list, and nesting makes each share's real rows a subset
of a smaller share's. Asking each cell for its dropped count makes the synthetic rows a replica
of the cells they replace, so the label and age-sex mix of the training set is the same at every
share. Fitting each generator on the whole split keeps one fit per holdout and share comparable
across arms, at the cost of letting a fill carry values of the rows it replaces; membership
inference reads exactly that leak. Reading the margin on E6's cell keeps it on the rows its
half-width was measured on, which an all-rows recall would not.

## Consequences

**Gained**: the five shares read on a real frame against `real`, with every privacy reading
contrasted with `bko-direct`, at no API spend.
**Accepted**: five runs fit each generator once per share and holdout, so a TabDDPM fit's 435.9
seconds on the whole split and a CTGAN fit of over an hour repeat at every share (ADR
[`research/diffusion-baseline`](diffusion-baseline.md)); the runs run as parallel slices.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| One run with nested shares | The scores carry no share key, so the shares' rows could not be told apart | 2026-11-25 |
| `kept_count`'s floor of two rows per cell | At share 1 it would keep real rows | 2026-11-25 |
| The LoRA classifier at each share | It is unit `slm-lora-results`'s, at the mixes ADR [`research/slm-comparison-design`](slm-comparison-design.md) allows | 2026-11-25 |
| A replica request, which stops the run on a shortfall | An arm that cannot fill a cell would stop every share; a gap request scores its partial fill | 2026-11-25 |

## Enforcement

- Owner: `src/bko/datasets/split.py`, `src/bko/evaluate/requests.py`,
  `src/bko/evaluate/experiment.py`, `src/bko/evaluate/prepare.py`, `src/bko/evaluate/run.py`,
  `src/bko/evaluate/runs_direct.py`
- Pinned by: `tests/test_datasets_share.py` (the dropped count's rounding with no floor, the
  nesting, the hash ranking, share 1 dropping every row), `tests/test_evaluate_share.py` (a share
  refused beside a removal or with no flagged cell, the test split untouched at every share, the
  request's quota, the training size at every share, `real` on the whole split, a partial fill,
  a fill of fewer than two labels at share 1, `prior-shift` and `class-weight` refused,
  the `experiment` line's key and its assumed value), `tests/test_evaluate_share_runs.py` (the
  shares, the arms, no paid arm, each run's slices, the reference arm and cell),
  `tests/test_evaluate_share_readers.py` (`rows`, `panel`, `shared`, `merge`, `trace`, `privacy`
  and `contradictions` on a share run's folder)

## Reversal Conditions

**A share key joins the scores**: one run could then hold every share. **An arm cannot fit on the
whole split at a share**: each generator would fit on the kept rows instead, and membership
inference would read only the kept rows as members.

## Related

- [The Diabetes 130 subgroup-removal design](diabetes-subgroup-removal-design.md)
- [Membership inference, NNDR and DCR](membership-inference-reading.md)
- [The TabDDPM baseline](diffusion-baseline.md)
- [The LoRA classifier comparison](slm-comparison-design.md)
