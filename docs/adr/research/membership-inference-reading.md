---
status: proposed
topic: research
created: 2026-11-05
updated: 2026-11-05
---
# Membership inference and NNDR read an arm's added rows against a null arm, outside the removed cell

## Context

`docs/RESEARCH.md` names three privacy measures: distance to closest record (DCR), nearest-neighbour
distance ratio (NNDR) and membership-inference risk. Only DCR is measured (ADR
`research/e1-evaluation-protocol`). A low DCR says an added row sits near some training record.
It does not say whether the added rows let an attacker tell training records from held-out ones.
`tasks/evaluate-membership-inference.md` scopes the reading; this record fixes it.

Distance-based attacks score a record by its distance to a generated sample, as the work of Chen
and colleagues does for generative models ([`chen-2020`](../../literature.md#privacy)), and
Hilprecht and colleagues' Monte Carlo attack counts generated samples near the record
([`hilprecht-2019`](../../literature.md#privacy)). Stadler and colleagues' title frames the privacy of
synthetic data as an anonymisation question ([`stadler-2022`](../../literature.md#privacy)), and
van Breugel and colleagues' attack detects overfitting of the generator
([`vanbreugel-2023`](../../literature.md#privacy)). None of these works is fetched, so the
sentences above are each work's report as its title and the plan give it, not a checked claim.

E6 removes most or all of one cell's training records and lets each arm add rows back
(ADR `research/diabetes-subgroup-removal-design`). The records removed are neither training
records nor held-out records. The `real` arm adds exactly those removed records back, which are
real records the attacker has not seen as members.

## Decision Drivers

* **Must separate a copy of a training record from a draw of the population.**
* **Must give a null**, so an AUC above 0.5 caused by the removal's shifted distribution is not read
  as leakage.
* **Must make no API call**, reading only the rows store (ADR `research/rebuilt-rows-store`).
* **Must report every arm that adds rows, and say when an arm added none.**
* **Must not turn a descriptive measure into a pass or fail**, since the plan fixes no privacy
  threshold for these readings.

## Decision

- **Score.** A record's score is minus its distance to the nearest added row, under the DCR metric
  of `bko.evaluate.privacy` with each column's span taken from the training split. A higher score
  predicts membership.
- **Members and non-members.** Members are the prepared training split, the records the arm's
  generator or classifier could have seen. Non-members are the whole test split. The removed
  records are neither and enter no group.
- **Scopes.** Each statistic is read in three scopes:
  - outside the removed cell;
  - the removed cell, at f = 0.985 only, where it keeps members;
  - label 1 outside the cell.
- **Statistics.** AUC from `sklearn.metrics.roc_auc_score`, which gives tied scores their midrank,
  and the true-positive rate (TPR) at false-positive rates (FPR) 0.01 and 0.001, by linear
  interpolation on `sklearn.metrics.roc_curve(drop_intermediate=False)`. The cell scope has no
  0.001 reading.
- **Reference arm.** E6 reads each arm against `real`, the empirical null: its added rows are real
  records the attack has not seen as members, so its AUC shows what the removal alone does to the
  statistic. E10 has no `real` arm and reads against `bko-direct`.
- **The rule.** Each arm's per-holdout contrast with the reference arm, on each statistic, goes
  to `bko.evaluate.criteria.corrected_interval` with `confidence` set by
  `refuting_confidence(0.95, k)`, `family=1` and `ratio` the held-out ratio of the sized split. `k`
  is the arms with a defined contrast times 2 statistics, per fraction and scope. An arm exceeds
  the reference on a statistic when its interval lies above 0. The rule is a reading rule: it
  sets how the contrasts are read together and is no criterion that a run passes or fails.
- **No rows.** An arm that adds no row on a holdout has an undefined (NaN) statistic there. The
  reading counts these holdouts and leaves them out of the contrast, and never reads them as 0.
- **Cell scope.** At f = 1.0 the cell keeps no training record, so it has no members and the scope
  is undefined. At f = 0.985 it keeps 19 members, so its TPR takes steps of 1/19 and the scope is
  descriptive only: no interval there is a verdict.
- **Holdouts.** All 100 holdouts of each fraction.
- **NNDR.** For each added row, d1 / d2 for its nearest and second-nearest training records under
  the DCR metric. A pair with d1 = d2 = 0 reads 1, and d1 = 0 < d2 reads 0. The summary is the 5th
  percentile and the median, with counts of the 0/0 and the d1 = 0 rows. The held-out rows, read
  the same way against the training records, are the baseline. The ratio of nearest to second
  nearest follows the match test of Lowe ([`lowe-2004`](../../literature.md#privacy)), as that
  work reports it for image features.
- **DCR and NNDR stay descriptive.** The reading recomputes each arm's DCR 5th percentile and
  median beside its NNDR and logs their contrasts without a verdict. Ganev and De Cristofaro report
  that similarity-based metrics of this kind do not establish privacy
  ([`ganev-2025`](../../literature.md#privacy)); that work is not fetched.
- **Command.** `bko.evaluate.membership` holds the attack, and the `privacy` command of
  `python -m bko.evaluate` reads each run's rows stores and writes `privacy.csv` and `privacy.log`
  ([`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#evaluation)). It makes no API call.

## Rationale

**The all-records scope is dropped.** A scope over every record mixes the removal with the arm. The
planning measurements of 2026-11-05 on E6 holdout 0, which E8's recorded run supersedes, put its AUC
within 0.4941 to 0.5031 for every arm tried. Among them are copies of the 19 cell members, whose
AUC is 1.0000 in the cell, and 1,250 copies of label-1 members, whose AUC is 0.6590 on label 1
outside the cell. A scope on which a copier of the cell reads 0.4965 cannot show copying, so
the reading uses the three scopes above.

**`real` is the null, and a no-row arm is NaN.** An arm's cell AUC can sit above 0.5 with no copying of a member. In the same planning
measurements, `real` thin read 0.5708 in the cell where `bko-direct` thin read 0.6784. Read alone,
neither number separates leakage from what the removal does to the statistic; the contrast with
`real` reads each arm beside the arm that adds real records no attacker has seen as members. An arm with no rows has no nearest added row, so any number would be
invented; a 0 would read as a perfect inversion. NaN with a count keeps the holdout visible and
out of the mean.

**The cell is coarse.** With 19 members and a test cell of several hundred records (ADR
`research/diabetes-subgroup-removal-design`), the cell TPR moves in steps of 1/19 and FPR in steps
above 0.001, so an interval over it would claim more than the data holds. The planning measurements
on holdouts 0 to 9, `bko-direct` minus `real` in the cell, had a mean of +0.1242 and an SD of
0.0656, against a half-width of 0.1059 at k = 24 and J = 100. A subset of holdouts only widens the
interval.

**TPR at 0.001.** Carlini and colleagues report that an attack's average-case AUC hides what a
low false-positive rate shows ([`carlini-2022`](../../literature.md#privacy)); that work is not
fetched. A copy of one record leaks that record with certainty, and an AUC averaged over thousands
of records does not move. The two rates, 0.01 and 0.001, read the strongest scores.

**The ratio is the held-out ratio.** `corrected_interval` takes n2 / n1, the held-out over training
records, as E1's criteria do; the reading passes the same value (ADR `research/e1-evaluation-protocol`).

**Answer to the `DCROverfittingProtection` ruling.** E1 rules out that measure because its value
depends on the training and test sizes: at 238 training and 59 test rows an ideal generator scores
about 0.27. AUC is a rank statistic and does not change with the sizes of its two groups, and the
contrast with `real` removes what a size ratio would add. The membership AUC answers the question
that measure asked.

## Consequences

**Gained**: a privacy reading that sees a copied training record where DCR's 5th percentile does
not; a null arm every E6 contrast is read beside; NNDR beside DCR for every row-adding arm.
**Accepted**: the cell scope is descriptive and undefined at f = 1.0; no arm passes or fails; a
distance-based attack is the weakest attack that reads copying, and misses an attacker with a
reference model; the figures in this record's Rationale are planning measurements, and the run of
record supersedes them.

## Ruled Out

| Idea | Why ruled out | When |
|---|---|---|
| The all-records scope | Reads 0.4941 to 0.5031 for every arm tried, including a full copier of the cell (Rationale) | 2026-11-05 |
| Subsampling members | Discards training records the attack can score, and a subsample's size would set the tail rates | 2026-11-05 |
| The removed records as non-members | The removal selects them, so the test shifts to a population the training split lacks | 2026-11-05 |
| A reference-model attack | Needs a second generator fit per arm and holdout, beyond this reading's no-new-fit scope | 2026-11-05 |

## Enforcement

- Owner: `src/bko/evaluate/privacy.py` (the distances and NNDR) and `bko.evaluate.membership` (the
  attack, its scopes and the `privacy` command's reading), which this record's code adds
- Pinned by: `tests/test_evaluate_privacy.py` for the distances and NNDR; the attack's
  calibration, identity-refusal and rule tests sit in the `bko.evaluate.membership` test module

## Reversal Conditions

A recorded run in which `real`'s statistic cannot be told from an arm that copies training records
would show the null fails and send the reading to a reference-model attack. A privacy criterion in
`docs/RESEARCH.md` that fixes a threshold would turn the reading rule into a criterion and need its
own record. A cell that keeps enough members for a stable TPR would give the cell scope an interval.

## Related

- [Evaluation protocol](e1-evaluation-protocol.md)
- [Rebuilt rows store](rebuilt-rows-store.md)
- [Diabetes subgroup removal](diabetes-subgroup-removal-design.md)
