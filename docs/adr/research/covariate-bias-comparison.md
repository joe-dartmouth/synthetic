---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# Juwara et al.'s removal fractions read on E6's cell

## Context

[`docs/RESEARCH.md`](../../RESEARCH.md#published-comparisons) runs four protocols beside published
numbers. Juwara et al. ([`juwara-2024`](../../literature.md#augmentation-versus-resampling)) remove
a share of one covariate level's records from a training set, add synthetic records of that level
back (synthetic minority augmentation, SMA) and read the full model's AUC, the odds ratio of the
biased covariate `Z` and `I_Z`, the overlap of its confidence interval with the ground truth's
([`karr-2006`](../../literature.md#evaluation-and-failure-modes)). Their generator is a sequential
synthesizer whose every model is a gradient-boosted tree
([`elemam-2021`](../../literature.md#tabular-generators)). Their code is on the OSF node `rkf9t`
([doi:10.17605/OSF.IO/RKF9T](https://doi.org/10.17605/OSF.IO/RKF9T), MIT licence), one archive,
`bias-w-sma for juwara et al paper.zip` (`https://osf.io/download/jk2qv/`, 139,595 bytes, read
2026-11-25). Its synthesis wrappers call `sdgs`, which a comment in the real-data script calls
not publicly available, so the scripts do not run as released.

E6 ([`docs/EVIDENCE.md`](../../EVIDENCE.md#e6-diabetes-130-subgroup-removal)) removes a share of
one cell's training rows on the Diabetes 130 frame: `E6_CELL`, `(6-7, male, 1)`, which holds
1,269 training rows, and its control runs E12 hold the controls `bko-nomap`, `label-bootstrap`
and `prior-shift`.

## Decision Drivers

* **Must read the fractions Juwara et al. read**, so the runs sit beside their table.
* **Must size each fill as their code sizes it**, since the paper states the rule only
  qualitatively.
* **Must add no score row to any other run**: `rows` refuses a key only the rescore holds.
* **Must spend no money.**

## Decision

1. **The fractions.** 0.15, 0.30, 0.50, 0.80 and 0.95 of `E6_CELL`'s training rows, removed per
   holdout at the holdout's seed by `bko.datasets.ablation.thinned`. Juwara et al. use 15%, 30%,
   50% and 80% on real data and 15% to 95% in simulation. One run per fraction, `p3-fifteen`,
   `p3-thirty`, `p3-fifty`, `p3-eighty` and `p3-ninetyfive`, each on E6's holdouts in ten slices.
2. **The arms.** `none`, `real`, `smote`, `ctgan`, `cart`, `bko-direct`, `bko-nomap`,
   `label-bootstrap` and `prior-shift`; `bko-direct` the candidate, every other arm but `real` a
   comparator, no replica and no criterion.
3. **The sizing.** Their marginal-bias augmentation, `1_Simulated_data_SMA.R` lines 432 to 435:

   ```
     countminor<-length(which(biased_data$Z%in%levels(biased_data$Z)[1]))
     prop_remain<-1-props
     total_size<-floor(countminor/prop_remain)
     synth.sel<-synth[sample(x = which(synth$Z==0),size = (total_size-countminor), replace = T),]
   ```

   `countminor` is the minority count the removal leaves, so `total_size` is about the count
   before it.
   With `r` the cell's rows the removal leaves, each run's `fill_to` is `floor(r / (1 - fraction))`
   in exact arithmetic (`juwara_total`), their `total_size`, and each row-adding arm's quota is
   `fill_to - r`, their `total_size-countminor`. Their removal takes the floor of the share
   (`samps<-floor(table(data$X1)[1]*prop_out)`) and ours its ceiling (`removed_count`). On
   `E6_CELL`'s 1,269 rows, `fill_to` is 1,268 at 0.15, 0.30 and 0.50, 1,265 at 0.80 and 1,260 at
   0.95.
4. **The flagged cell.** `Experiment.flagged_cell` is `E6_CELL`: it is flagged whatever its
   count, since at 0.15 it keeps 1,078 rows, far above the threshold, and the gap request asks for
   it alone. With a removal, it must be the removal's cell.
5. **The reading.** ROC AUC (`auc-roc`) on the scope `flagged-subgroups`, the test rows of the
   cell's age band and sex at both labels, and on all test rows, beside the set's measures. The
   removed cell holds one label, so ROC AUC is undefined on its own test rows. `utility` writes
   `auc-roc` only for a run whose `flagged_cell` is set.
6. **The reference arm.** `real` adds the removed rows, as it does in every run with a removal.
7. **The register.** [P3](../../EVIDENCE.md#p3-juwara-removal-fractions-on-the-diabetes-130-frame)
   holds the Configuration and quotes Table 2.

## Rationale

The fractions and the sizing are the two things their protocol fixes that ours can take
unchanged. Their code is the only statement of the sizing, so the quota follows its lines. ROC
AUC is the measure their table reports; reading it on the cell's age-sex group, where both labels
occur, is the nearest scope the harness defines, and gating it on `flagged_cell` leaves every
other run's scores file as it is.

What differs from the paper:

* **A cell against a covariate level.** They remove records of one level of one covariate; the
  removal here takes one label of one age-sex group, so it is a label shift within a subgroup.
* **One fill against 100.** Each holdout here fits one seeded fill per arm; their estimates
  average m = 100 synthetic copies.
* **`cart` for their synthesizer.** Their sequential gradient-boosted trees run through `sdgs`,
  which is not public; `cart`, sequential CART, is the nearest free arm. Their synthesizer fits
  the whole biased training set and samples the minority level's rows from its output, with
  replacement; each arm here answers a quota for the one cell.
* **Simulation against real data.** Table 2 holds their simulation results, a cohort of 5,000
  records and 500 repetitions, and its AUC is the full model's on all records, not the minority
  group's.

## Consequences

**Gained**: the five fractions read on a real frame, each fill sized by their code's rule, beside
the controls of the control reading rule, at no API spend.
**Accepted**: no reading here replicates Table 2; each sits beside it. The runs cost CPU time:
each CTGAN fit takes over an hour on this frame, so the five runs run as parallel slices.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Their request-only datasets | Access is by request, so no reader could rebuild the reading | 2026-11-25 |
| Their UCI breast-cancer data | It has no sex column, so no age-sex cell exists on it | 2026-11-25 |
| A new `Scope` for the removed cell | Every run's scores file would gain rows, and the cell holds one label, so ROC AUC is undefined there | 2026-11-25 |

## Enforcement

- Owner: `src/bko/evaluate/runs_published.py` (`P3`, `P3_FRACTIONS`, `P3_ARMS`,
  `juwara_total`), `src/bko/evaluate/requests.py` (`flagged`, `gap_request`),
  `src/bko/evaluate/experiment.py` (`Experiment.flagged_cell`), `src/bko/evaluate/utility.py`
  (`roc`), `src/bko/evaluate/run.py`
- Pinned by: `tests/test_evaluate_flagged_cell.py` (the cell flagged above the threshold, its
  quota, a cell other than the removal's refused, a cell at `fill_to` refused, the `experiment`
  line's key and its assumed value), `tests/test_evaluate_auc_roc.py` (a hand-computed ROC AUC,
  NaN on one label, no `auc-roc` row without a flagged cell, both scopes with one),
  `tests/test_evaluate_covariate.py` (the fractions, the arms, the sizing on a toy cell and at
  1,269 rows, no paid arm, each run's slices), `tests/test_named_runs_slices.py` (the run names
  in order); outside the gate, `tests/test_datasets_diabetes_pinned.py` with `--releases` (each
  P3 run asks for `E6_CELL` alone, `juwara_total(1269, fraction)` less the rows the removal
  leaves, on holdout seeds 0 to 2)

## Reversal Conditions

**`sdgs` becomes public**: their synthesizer can run as an arm in place of `cart`. **Their
appendix reports the minority group's AUC on real data**: the reading moves beside that table.

## Related

- [The Diabetes 130 subgroup-removal design](diabetes-subgroup-removal-design.md)
- [The prior-shift arm](prior-shift-comparator.md)
- [The no-mapping ablation](no-mapping-ablation.md)
- [The `bko-coherent` arm's sequential CART](free-field-synthesis.md)
