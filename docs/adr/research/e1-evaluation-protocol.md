---
status: accepted
topic: research
created: 2026-10-15
updated: 2026-10-15
---
# E1's evaluation: repeated holdouts, one classifier, separation judged on paired intervals

## Context

E1 compares augmentation arms on UCI Heart Disease (`docs/RESEARCH.md` section 4). The 2-pager
states its criteria but no protocol, and gives no basis for its primary threshold, a 10% gain over
no augmentation. On the archive `bko.datasets.sources` names (measured 2026-10-15), every holdout
tests 59 records, 27 of them positive. The underrepresented cells are `<55 female 1`, 2 records in
all and never held out, and `>=55 female 1`, with 5 test records per holdout. Baseline classifiers
were scored on the real test split while this protocol was designed, and no BKO arm existed: their
whole-split AUC-PR sat near 0.91, which leaves a 10% gain no room.

## Decision Drivers

* **Must compare arms on identical splits**, so every difference is paired.
* **Must read the test split only in `bko.evaluate`** (ADR `research/uci-evaluation-frame`).
* **Must give intervals that hold for overlapping holdouts**, with about 5 positive test records
  in the underrepresented cells.
* **Must fix every reading before the first run that includes the candidate.**
* **Must reproduce from seeds on one machine**, with no unseeded randomness in a judged statistic.
* **Should run unchanged on a larger frame in the same schema.**

## Decision

- **Configuration in code.** `bko.evaluate.experiment.E1` holds the holdout seeds, the arms, the
  candidate (`bko`), the comparators, the replica arms, the fill level, the two margins, the
  confidence and the attempts per fill. `docs/EVIDENCE.md` (E1) states how each criterion is
  read. No flag changes a value; `compare --candidate` judges another arm as a dry run, never as
  E1's verdict.
- **Repeated holdouts.** Each of `E1`'s seeds gives one cell-stratified split from
  `bko.datasets.split`; every request on that split takes the same seed.
- **The gap request.** Every arm is asked to fill each cell under the threshold up to it, the same
  quota on every holdout; a control adds no rows. The BKO arm fills the same request: its pipeline
  runs on each holdout's training rows and returns exactly the requested rows per cell, guided by
  the ontology. BKO-detected gaps are judged by the completeness criterion and E2, not here.
- **The replica request.** The row-adding baselines also fill the training split's own cell
  counts. Those rows alone train a classifier (train on synthetic, read against the
  no-augmentation arm, which trains on real) and are compared with the training split for
  fidelity.
- **One classifier.** Logistic regression (lbfgs, L2, C = 1) on standardised measured predictors
  and coded predictors one-hot encoded over their codebook. The weights reach the regression only.
  A record is called positive at a probability of 0.5 or more. A fit that does not converge stops
  the run.
- **Measures.** Recall, precision and AUC-PR over the whole test split and over the flagged
  subgroups, the age-sex subgroups that hold an underrepresented cell; a measure a scope cannot
  define is empty, and a criterion that reads an empty value refuses to judge. Fidelity is
  SDMetrics' quality report with every column pair counted. Distance to closest record is SDMetrics'
  `calculate_dcr`, read as a 5th percentile and a median, for the added rows and for the test rows,
  both against the training rows.
- **Intervals.** Every contrast is paired by holdout and gets Nadeau and Bengio's corrected
  resampled t interval ([`nadeau-2003`](../../literature.md#evaluation-and-failure-modes)), the
  variance of the mean being (1/J + n2/n1) S^2, with n2/n1 the largest held-out-to-training ratio
  of any cell.
- **Criteria.** Primary: in both recall and AUC-PR over the whole test split, the candidate minus
  each comparator minus the margin has an interval above zero. Secondary: the same over the
  flagged subgroups. Privacy: the candidate's added rows' 5th-percentile distance minus the privacy
  share of the test rows' has an interval above zero. A criterion is met when every interval it
  reads lies above zero, not met when one lies below zero, and inconclusive otherwise.
- **Retries, then all or nothing.** An arm that refuses a request tries it again with a seed
  derived from the holdout seed and the attempt, up to `E1`'s attempts, and each retry is logged.
  A refusal on the last attempt stops the run with no output; a method output that breaks its
  contract stops it at once.

## Rationale

The holdouts overlap, so a plain t interval over them is too narrow and narrows to nothing as
holdouts are added; Nadeau and Bengio's correction is the standard for repeated random holdouts.
Taking n2/n1 from the cell where it is largest keeps the flagged scope's interval as wide as its 5
test records against 18 training records need. Past about 30 holdouts the interval barely narrows,
since n2/n1 dominates, but corrected resampled tests replicate their verdicts across seed sets at
about 100 runs ([`bouckaert-2004`](../../literature.md#evaluation-and-failure-modes)), and the
baselines are cheap to rerun.

Requiring the candidate to clear every comparator is an intersection of tests, each at the
interval's level, so it needs no multiplicity correction and never picks the best comparator after
seeing results. E1 makes the stricter claim and judges ADASYN and the controls too. An interval
that straddles its threshold is neither a pass nor evidence of no effect, hence a third verdict. A
margin in the measure's own units means the same on any dataset and baseline level. Recall is
read at a fixed threshold, which any arm that adds positive rows can move without ranking records
better; AUC-PR, which no threshold moves, must clear the same margin, so a shifted threshold alone
cannot pass.

Every arm fills the same per-cell request so that equal row counts hold by construction; letting
the BKO arm fill its own detected gaps would make the baselines' request depend on its output and
send SMOTE cells it refuses.

The whole split resolves small effects. The flagged subgroups are what the project targets, but
they hold 5 positive test records per holdout, so their verdict is secondary and likely
inconclusive until the data grows. AUC-PR inside the underrepresented cells is not a scope: both
cells hold one label, so every scorer gets 1.0.

The gap request follows the threshold of `docs/RESEARCH.md` section 2 and is the same on every
holdout; on holdout 0 it equals `python -m bko.baselines <method> --rows 20`. The gap rows hold
one label in two cells, so they can neither train a classifier alone nor be compared with the
training split; a replica of the training split's cells is the conventional train-on-synthetic set
([`xu-2019`](../../literature.md#tabular-generators)). The BKO arm's replica,
a paid generation per holdout, is left to the unit that adds the arm.

Logistic regression is deterministic, takes weights and has no seed of its own; a random forest
scored the same on the baselines while its own seed moved subgroup recall by one of 5 positives.
The test rows are real records no generator saw, the natural real-data baseline for distance to
closest record, and the 5th percentile is the reading that catches near-copies
([`zhao-2021`](../../literature.md#evaluation-and-failure-modes)). An ideal generator ties the test
rows in expectation, so a rule at a share of 1 would fail it about half the time; the share of 0.9
allows for the sampling noise of 59 test records and still fails a generator that copies training
rows. A refused holdout would leave the comparison unpaired. Default TVAE refuses the gap request
on 2 of the 100 holdouts (measured 2026-10-15): its rejection sampling draws 1,000,000 rows and
still lacks a few of the 18 rows asked of `<55 female 1`, a cell of 2 training records. That is a
property of one sampling seed, not of the split: with derived seeds, one of the two fills on its
third attempt and the other on its fourth. A retry keeps every holdout paired and keeps the
request the same; dropping the holdout would select splits on how one generator behaved. The run
still stops when every attempt is refused.

## Consequences

**Gained**: every criterion computed by code from one scores file; verdicts that do not depend on
which arm looked best; a harness that takes any frame in the evaluation schema.
**Accepted**: the 2-pager's 10% criterion is replaced (`docs/RESEARCH.md` section 9);
`<55 female 1` holds 2 records, none ever tested, yet takes 18 of the 20 gap rows, so its effect
shows only through other test records; subgroup verdicts are likely inconclusive at this size;
distance to closest record is weak evidence of privacy, blind to attacks that need no near-copy;
`calculate_dcr` is not a public SDMetrics export; baseline results on the real test split were in
view when the readings were fixed, no BKO result was; on a retried holdout, that arm's rows
come from a derived seed rather than the holdout seed; the BKO pipeline runs once per holdout, so
its paid generation grows with the holdout count; E1 does not exercise BKO gap detection.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The 2-pager's 10% gain over no augmentation | No stated basis; whole-split AUC-PR leaves it no room | 2026-10-15 |
| Non-overlap of each arm's own interval | Discards the pairing and asks for roughly p below 0.006 | 2026-10-15 |
| Judging on the mean over holdouts | A true effect at the threshold passes half the time | 2026-10-15 |
| Non-inferiority for "matches" with an upper bound at or above zero | Passes almost always at these interval widths | 2026-10-15 |
| A plain t interval over holdouts | Treats overlapping holdouts as independent | 2026-10-15 |
| Cell-stratified repeated k-fold | Leaves the 2-record cell one training record in a fold, which SMOTE refuses | 2026-10-15 |
| `DCROverfittingProtection` as the privacy measure | At 238 training and 59 test rows an ideal generator scores about 0.27 | 2026-10-15 |
| `DCRBaselineProtection` | A uniform random baseline, not a real one, seeded only through a private attribute | 2026-10-15 |
| AUC-PR within the underrepresented cells | Both cells hold one label, so every scorer gets 1.0 | 2026-10-15 |
| Fidelity and train-on-synthetic on the gap rows | The gap rows hold one label in two cells | 2026-10-15 |
| A random forest | Its own seed moves subgroup recall by one of 5 positives | 2026-10-15 |
| Recording a refused arm and continuing | Unpairs the comparison at that holdout | 2026-10-15 |
| Dropping, for every arm, a holdout where one arm refuses | Selects splits on one generator's behaviour, and lowers the holdout count | 2026-10-15 |
| Flags for the readings or the holdout count | Lets a run choose its reading after the fact | 2026-10-15 |
| A TOML configuration file | A parser for one reader, and no type checking | 2026-10-15 |
| The BKO arm filling its own detected gaps | The baselines' request would depend on its output, and SMOTE refuses cells under 2 records | 2026-10-15 |
| Equal gap rows per underrepresented cell | Not derived from the threshold; chosen only to move rows toward the cell with test records | 2026-10-15 |

## Enforcement

- Owner: `src/bko/evaluate/`
- Pinned by: `tests/test_evaluate_criteria.py`, `tests/test_evaluate_experiment.py`,
  `tests/test_evaluate_run.py`, `tests/test_evaluate_classifier.py`,
  `tests/test_evaluate_utility.py`, `tests/test_evaluate_fidelity.py`,
  `tests/test_evaluate_privacy.py`, `tests/test_evaluate_scores.py`, `tests/test_evaluate_cli.py`

## Reversal Conditions

**A reading changes only while E1's Runs field in `docs/EVIDENCE.md` has no row**; after the first
run, a changed reading is a new experiment entry.

**The holdout count and the margins are re-chosen if `proposal-evaluation-data` grows the frame
before E1's first run**; this record and `E1` change together.

**`calculate_dcr` is replaced if an SDMetrics release removes
`sdmetrics.single_table.privacy.dcr_utils.calculate_dcr`.**

**The classifier changes if any holdout logs a fit that did not converge within `MAX_ITER`.**

**The BKO arm joins `E1.replica_arms` only if `proposal-final-evaluation` records its replica
spend in E1's Configuration before the run.**

**The gap request stops binding the BKO arm if its generation cannot target a cell**; the request
then becomes the BKO arm's realised rows per cell, and this record changes before E1's first run.

## Related

- [The UCI evaluation frame](uci-evaluation-frame.md)
- [The baselines](baseline-augmentation-design.md)
- [Python 3.14, without synthcity](../tooling/python-314-without-synthcity.md)
