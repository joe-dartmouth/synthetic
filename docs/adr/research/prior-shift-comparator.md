---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# The prior-shift arm moves each requested subgroup's logits to its post-fill prevalence and adds no rows

## Context

E6's gain for `bko-direct` sits at the default cutoff with ranking unchanged: its flagged AUC-PR
reads level with `none`'s ([EVIDENCE.md](../../EVIDENCE.md#e6-diabetes-130-subgroup-removal)).
That is the pattern a shift in the subgroup's risk level produces. The control reading rule of
[RESEARCH section 11](../../RESEARCH.md#11-direct-method-study) contrasts `bko-direct` with
`prior-shift`, an arm that makes only that shift and adds no rows, so the contrast shows how much
of the gain the shift explains. Shifting an uncorrected model's threshold has given a balance of
sensitivity and specificity similar to resampling's
([`vandengoorbergh-2022`](../../literature.md#augmentation-versus-resampling)).

## Decision Drivers

* **Must use the information a row-adding arm uses and no more**: the training split and the gap
  request's counts, never a test record.
* **Must be finite at an emptied cell**, since E6 reads full removal.
* **Must leave every other arm's scores byte for byte**, since their runs of record are re-read
  under identity checks.
* **Must be scored by every model the run names**, the panel's included.

## Decision

1. **An arm, not a method.** The arm `prior-shift` (`PRIOR_SHIFT_ARM` in
   `bko.evaluate.experiment`) has a runner in `bko.evaluate.run.ARMS` that adds no rows, keeps
   unit weights and logs its targets in the `arm` line's params. It is not a
   `bko.baselines.request.Method` member.
2. **Subgroups.** The subgroups are the (age band, sex) pairs of the gap request's cells; a
   request not keyed by cell is refused.
3. **Target.** A subgroup's target is its prevalence after the fill,
   (n1 + q1) / (n0 + n1 + q0 + q1), where n0 and n1 are its training records per label and q0 and
   q1 the request's rows per label. The `arm` params hold the integer counts as
   `{"shift": {"<band>/<sex>": [negatives, positives]}}`.
4. **Offset.** Per model, the offset is the delta at which the mean of
   `expit(logit(clip(p)) + delta)` over the subgroup's training records equals the target. The p
   are 5-fold out-of-fold predictions on the training split from the same model factory at unit
   weights (`StratifiedKFold`, shuffled at the holdout seed; fewer folds when a label holds fewer
   than 5 records, refused below 2). The solver bisects on [-64, 64] for 100 iterations with the
   probabilities clipped to [1e-12, 1 - 1e-12]. A fold fit's refusal names the holdout, the arm,
   the model and the fold, and a perceptron fold fit logs whether it converged.
5. **Scoring.** The scored fit is `none`'s. The offset is added to the logits of the subgroup's
   test records only; every other record keeps its probability, and the cutoff is unchanged.
6. **No training record.** A subgroup with no training record gets offset 0, and its `shift` line
   logs it. Its target is still defined: `bko.baselines.request.Request` refuses a zero-row
   quota, so every subgroup the request names holds at least one requested row, and the target's
   denominator is at least 1.
7. **Readers.** The arm cannot be a replica arm, since it adds no rows. `panel` regenerates it, as
   it does `none` and `class-weight`.

## Rationale

The arm re-estimates the intercept of each subgroup, the parsimonious update that Steyerberg et
al. found worked better than extensive revision
([`steyerberg-2004`](../../literature.md#augmentation-versus-resampling)), with the design's count
as the target in place of a validation sample's outcome. Its target counts the same rows a
row-adding arm adds, so the contrast holds the information fixed and varies only what the rows
carry beyond the subgroup's prevalence. A mean over predictions, solved by bisection, stays finite
when the cell is empty, where a closed form does not. Out-of-fold predictions read the model as it
predicts unseen records; in-sample tree predictions reproduce the training labels and inflate the
offset. On E6 the flagged set is one subgroup, and a constant logit shift keeps the order within
it, so the arm's flagged AUC-PR equals `none`'s and it moves recall and Brier only.

## Consequences

**Gained**: a free control whose contrast with `bko-direct` measures how much of a gain a shift in
the subgroup's risk level explains; a `shift` line per model and subgroup that shows each offset
with the target's counts and the fold count, so the target can be recomputed from the line
([`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#evaluation) holds its format).
**Accepted**: five more fits per model per holdout for this arm; outside the requested subgroups
the arm scores as `none` does.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The closed form -log(1 - f) from the removal fraction | Unbounded at f = 1, which E6 reads | 2026-11-24 |
| Weighting the subgroup's records | Undefined at an empty cell, whose label has no record to weight | 2026-11-24 |
| A shift estimated from test records | The test split is what the arms are scored on; no arm reads it | 2026-11-24 |
| A per-subgroup cutoff move | Flagged Brier reads probabilities, so it cannot see a moved cutoff | 2026-11-24 |
| In-sample training predictions | Tree models reproduce training labels, which inflates the offset | 2026-11-24 |
| A `Method` member | A method returns rows or weights for a training set; the offset acts at scoring time | 2026-11-24 |

## Enforcement

- Owner: `src/bko/evaluate/run_scores.py`, with the solver in `src/bko/evaluate/shift.py`
- Pinned by: `tests/test_evaluate_shift.py` (the hand-computed offsets, the finite offset at an
  empty cell, out-of-fold fits that exclude the records they predict, the `shift bound` WARNING, the
  arm's runner in `bko.evaluate.run.ARMS`, and every other arm's rows byte for byte with and without
  the arm), `tests/test_evaluate_panel_shift.py`, `tests/test_evaluate_diabetes.py` (the arm in
  E6-shaped rehearsed runs)

## Reversal Conditions

**A `shift bound` WARNING**: no offset reaches the subgroup's target, a target of 0 or 1, which a
mean of logistic values never equals, or one the moved mean misses by more than 1e-6. **The control
reading rule stops naming `prior-shift`**; the check is the Contrasts line of
[RESEARCH section 11](../../RESEARCH.md#11-direct-method-study).

## Related

- [The Diabetes 130 subgroup-removal design: the full adult frame, one thinned cell at the prevalence cutoff, every arm](diabetes-subgroup-removal-design.md)
- [A classifier panel reads recorded runs under four more models, with calibration, and no verdict](classifier-panel.md)
- [The no-mapping ablation draws bko-direct's records with an empty mapping table](no-mapping-ablation.md)
