---
status: accepted
topic: research
created: 2026-11-05
updated: 2026-11-06
---
# Membership inference and NNDR read an arm's added rows against a null arm, outside the removed cell

## Context

`docs/RESEARCH.md` names three privacy measures: distance to closest record (DCR), nearest-neighbour
distance ratio (NNDR) and membership-inference risk. ADR `research/e1-evaluation-protocol` reads
DCR alone. A low DCR says an added row sits near some training record. It does not say whether the
added rows let an attacker tell training records from held-out ones. This record fixes how the
other two are read.

Chen and colleagues report distance-based attacks that score a record by its distance to a
generated sample ([`chen-2020`](../../literature.md#privacy)), and Hilprecht and colleagues report a
Monte Carlo attack that counts generated samples near the record
([`hilprecht-2019`](../../literature.md#privacy)). Stadler and colleagues' title frames the privacy
of synthetic data as an anonymisation question ([`stadler-2022`](../../literature.md#privacy)), and
van Breugel and colleagues report an attack that detects overfitting of the generator
([`vanbreugel-2023`](../../literature.md#privacy)). None of these works is fetched, so each sentence
above is that work's report, not a checked claim.

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
  - the removed cell, defined only where the removal keeps members;
  - label 1 outside the cell.
- **Statistics.** AUC from `sklearn.metrics.roc_auc_score`, which gives tied scores their midrank,
  and the true-positive rate (TPR) at false-positive rates (FPR) 0.01 and 0.001, by linear
  interpolation on `sklearn.metrics.roc_curve(drop_intermediate=False)`. The cell scope has no
  0.001 reading. A rate below 1 / (the scope's non-members), the least FPR one non-member makes,
  is NaN: the curve does not resolve it, and interpolating toward (0, 0) would invent a value.
- **Reference arm.** Each run family reads every arm against the arm `REFERENCE` in
  `bko.evaluate.membership` names; for E6 that is `real`, the empirical null. Its added rows are
  real records the attack has not seen as members, so its AUC shows what the removal alone does to
  the statistic. For the perturbation pilots, `e10-pilot-thin` and `e10-pilot-empty`, it is
  `bko-direct`: both perturbation arms rewrite its rows (ADR `research/identifier-perturbation`), so
  each contrast reads what the perturbation changes against the unperturbed rows, not against the
  null. A slice reads against the reference of the run it slices.
- **The rule.** Each arm's per-holdout contrast with the reference arm, on each statistic, goes
  to `bko.evaluate.criteria.corrected_interval` with `confidence` set by
  `refuting_confidence(0.95, k)`, `family=1` and `ratio` the largest held-out ratio across the
  holdouts (`criteria.held_out_ratio_of`). Each scope's attack statistics (AUC and TPR at 0.01)
  form one family, and the DCR and NNDR quantiles another; each family is read at family-wise
  0.95. `k` is the family's pairs of arm and statistic with two or more finite contrasts, so at
  most the arms times 2 for an attack scope and the arms times 4 for the distances. A pair with
  fewer, a NaN rate among them, leaves the family. TPR at 0.001 is written to the scores but read
  by no rule line.
- **Direction.** An arm leaks more than the reference (`leaks_more=true`) when, for an attack
  statistic, its interval lies above 0, and, for a DCR or NNDR quantile, below 0: a nearer added
  row leaks more. A pair with fewer than two finite contrasts reads `leaks_more=undefined`. The
  rule is a reading rule: it sets how the contrasts are read together and is no criterion that a
  run passes or fails.
- **No rows.** An arm that adds no row on a holdout has an undefined (NaN) statistic there. The
  reading counts these holdouts and leaves them out of the contrast, and never reads them as 0.
- **Cell scope.** Where the removal keeps no training record of the cell, as at E6's f = 1.0, the
  scope has no members and is undefined. At f = 0.985 it keeps R = 19 members (ADR
  `research/diabetes-subgroup-removal-design`), so its TPR takes steps of 1/19 and the scope is
  descriptive only: its rule lines are logged, and no interval there is a verdict.
- **Holdouts.** Every holdout of the run, its slices' stores read together.
- **NNDR.** For each added row, d1 / d2 for its nearest and second-nearest training records under
  the DCR metric. A pair with d1 = d2 = 0 reads 1, and d1 = 0 < d2 reads 0. The summary is the 5th
  percentile and the median, with counts of the 0/0 and the d1 = 0 rows. The held-out rows, read
  the same way against the training records, are written as the baseline and enter no contrast.
  The ratio of nearest to second nearest follows the match test of Lowe
  ([`lowe-2004`](../../literature.md#privacy)), as that work reports it for image features.
- **DCR and NNDR stay descriptive.** The reading recomputes each arm's DCR 5th percentile and
  median beside its NNDR and logs their contrasts against the reference arm without a verdict.
  Ganev and De Cristofaro report that similarity-based metrics of this kind do not establish
  privacy ([`ganev-2025`](../../literature.md#privacy)); that work is not fetched.
- **Command.** `bko.evaluate.membership` holds the attack and `bko.evaluate.privacy_rule` the rule;
  the `privacy` command of `python -m bko.evaluate` reads each run's rows stores and writes
  `privacy.csv` and `privacy.log` ([`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#evaluation)). It
  makes no API call.

## Rationale

**The all-records scope is dropped.** A scope over every record mixes the removal with the arm.
The removed cell's test records stay non-members while its training records are mostly gone, so
the shifted population moves the all-records AUC whatever the arm does. An arm that copies every
cell member moves that AUC only by the cell members' small share of all records, though it reads
them perfectly in the cell. A scope on which a copier of the cell can read as chance cannot show
copying, so the reading uses the three scopes above.

**`real` is the null.** An arm's cell AUC can sit above 0.5 with no copying of a member, since the
removal alone shifts the statistic. Read alone, an arm's number does not separate leakage from what
the removal does. The contrast with `real` reads each arm beside the arm that adds real records no
attacker has seen as members.

**A no-row arm is NaN.** An arm with no rows has no nearest added row, so any number would be
invented, and a 0 would read as a perfect inversion. NaN with a count keeps the holdout visible and
out of the mean.

**The cell is coarse.** With 19 members and a test cell of several hundred records, the cell TPR
moves in steps of 1/19 and FPR in steps above 0.001, so an interval over it would claim more than
the data holds. A subset of holdouts only widens the interval.

**TPR at low FPR, and its floor.** Carlini and colleagues report that an attack's average-case AUC
hides what a low false-positive rate shows ([`carlini-2022`](../../literature.md#privacy)); that
work is not fetched. A copy of one record leaks that record with certainty, and an AUC averaged
over thousands of records does not move. The two rates, 0.01 and 0.001, read the strongest
scores. Below one non-member's share of the FPR axis the curve has no point to read, so the rate is
NaN rather than a value interpolated from (0, 0).

**Each group is its own family.** The attack scopes ask different questions of different records,
and the distances are descriptive; one family over all of them would let a crowded scope widen the
others' intervals. Counting only the pairs with a defined interval keeps an undefined rate from
widening its neighbours'.

**The ratio is the held-out ratio.** `corrected_interval` takes n2 / n1, the held-out over training
records, as E1's criteria do; the reading passes the same value, the largest across the holdouts
(ADR `research/e1-evaluation-protocol`).

**Answer to the `DCROverfittingProtection` ruling.** E1 rules out that measure because its value
depends on the training and test sizes (ADR `research/e1-evaluation-protocol`, Ruled Out). AUC is
a rank statistic and does not change with the sizes of its two groups, and the contrast with
`real` removes what a size ratio would add. The membership AUC answers the question that measure
asked.

## Consequences

**Gained**: a privacy reading that sees a copied training record where DCR's 5th percentile does
not; a null arm every E6 contrast is read beside; NNDR beside DCR for every row-adding arm.
**Accepted**: the cell scope is descriptive and undefined at f = 1.0; a scope with fewer than 1,000
non-members has no TPR at 0.001, and one with fewer than 100 none at 0.01; no arm passes or
fails; a distance-based attack is the weakest attack that reads copying, and misses an attacker
with a reference model.

## Ruled Out

| Idea | Why ruled out | When |
|---|---|---|
| The all-records scope | Mixes the removal with the arm, and dilutes a copier of the cell to the cell's share of all records (Rationale) | 2026-11-05 |
| Subsampling members | Discards training records the attack can score, and a subsample's size would set the tail rates | 2026-11-05 |
| The removed records as non-members | The removal selects them, so the test shifts to a population the training split lacks | 2026-11-05 |
| A reference-model attack | Needs a second generator fit per arm and holdout, beyond this reading's no-new-fit scope | 2026-11-05 |
| Interpolating TPR below one non-member's FPR | Reads a value the curve does not hold | 2026-11-05 |

## Enforcement

- Owner: `src/bko/evaluate/privacy.py` (the distances and NNDR), `src/bko/evaluate/membership.py`
  (the attack, its scopes and the `privacy` command's reading) and
  `src/bko/evaluate/privacy_rule.py` (the rule)
- Pinned by: `tests/test_evaluate_privacy.py` for the distances and NNDR;
  `tests/test_evaluate_membership.py` for the attack's calibration, its rate floor, the rule and
  the command's identity refusals; `tests/test_named_runs_slices.py` for the slices a store's run
  may name

## Reversal Conditions

A recorded run in which `real`'s statistic cannot be told from an arm that copies training records
would show the null fails and send the reading to a reference-model attack. A privacy criterion in
`docs/RESEARCH.md` that fixes a threshold would turn the reading rule into a criterion and need its
own record. A cell that keeps enough members for a stable TPR would give the cell scope an interval.

## Related

- [Evaluation protocol](e1-evaluation-protocol.md)
- [Rebuilt rows store](rebuilt-rows-store.md)
- [Diabetes subgroup removal](diabetes-subgroup-removal-design.md)
