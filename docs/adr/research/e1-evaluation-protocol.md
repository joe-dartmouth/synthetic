---
status: accepted
topic: research
created: 2026-10-15
updated: 2026-10-22
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
* **Must fix every reading before the run it judges.**
* **Must reproduce every judged statistic on one machine**: from seeds where an arm takes one, and
  from the run's call log where it cannot (ADR `research/call-log-replay`).
* **Should run unchanged on a larger frame in the same schema.**

## Decision

- **Configuration in code.** `bko.evaluate.experiment.E1` holds the holdout seeds, the arms, the
  candidate (`bko`), the comparators, the replica arms, the fill level, the two margins, the
  confidence, the attempts per fill, the BKO arms' budget per attempt, the validity threshold and
  the run's spend ceiling. It refuses a candidate that is not one of its arms, and a replica arm
  that adds no rows. `docs/EVIDENCE.md` (E1) states how each criterion is read. No flag changes a
  value: `score --log`, `--replay` and `--resume` choose where calls are recorded and answered
  from, and `compare --candidate` judges another arm as
  a dry run, never as E1's verdict.
- **Repeated holdouts.** Each of `E1`'s seeds gives one cell-stratified split from
  `bko.datasets.split`; every request on that split takes the same seed.
- **The gap request.** Every arm is asked to fill each cell under the threshold up to it, the same
  quota on every holdout; a control adds no rows. Three BKO arms fill the same request, one per gap
  definition of ADR `research/bko-formal-gap-definition`: each runs its pipeline on each holdout's
  training rows and returns exactly the requested rows per cell, guided by its definition's gaps.
  `bko`, guided by per-subgroup gaps, is the candidate. `bko-population` and `bko-frequency` are
  arms but not comparators: no criterion compares the candidate with them or judges them, and their
  measures are reported beside it. BKO-detected gaps themselves are judged by the completeness
  criterion and E2, not here.
- **The BKO arms' added rows.** Each BKO arm is a `bko.evaluate.guided.GuidedArm`. It reads gaps and
  guidance from the holdout's training rows only, asks the Messages API for each requested cell's
  shortfall, cell by cell in the quota's order and round by round, and validates every reply with
  `bko.validate`. It drops each accepted candidate `bko.baselines.records.check` would refuse, by
  `bko.baselines.records.drops`, which tests:
  - a numeric value outside the training split's minimum and maximum, with the rule `check` uses;
  - then a cell the request does not ask for;
  - then a row past its cell's quota.

  A candidate counts toward the cell its own values name, as `check` reads it. This differs from
  fusion's join, which drops a record whose values name another cell (ADR
  `research/gap-closure-design`), because E1's rows are bound by `check`. The arm returns the kept
  rows cell by cell in the quota's order and in generation order within a cell. It never clips,
  rounds or repairs one, and it calls `check` before it returns. An attempt asks for at most `E1`'s
  budget, three times the request's rows, and raises a plain `BaselineError` as soon as the rest of
  that budget cannot cover the rows still short; the retries below repeat it. It logs one `guided`
  line per attempt with its outcome, its call range, the records asked, the candidates generated and
  passed, those dropped for each reason and those kept. Its refusal's message carries the same
  counts. A run stops, unretried, once its call log's spend reaches `E1`'s spend ceiling; the arm
  reads the spend before every call. That spend is the call log's whole spend, a resumed log's
  replayed calls included.
- **Validity.** Each BKO arm's validity rate is `passed` summed over `generated` summed, from every
  `bko.validate` call it makes in the run: every holdout, and every refused and retried attempt. It
  fills two requests on each holdout, the gap request and the replica request, and the rate reads
  both. A refused attempt returns no `Baseline`, so the arm keeps its counts itself; the harness
  drains them after the gap fills and again after the replica fills, and writes them per holdout as
  `candidates` and `replica-candidates` scores (`generated`, `passed`, and one count per drop
  reason). `compare` sums both over the holdouts for the rate, refuses a replica arm's scores
  without its replica counts, and logs each fill's counts and rate apart, with no verdict on a fill.
  The criterion reads `bko`'s rate: it is met when the rate reaches `E1`'s validity threshold, read
  as a point with no interval, and not met below it. A candidate dropped after validation stays in
  the rate, having passed every check. The other BKO arms' rates are reported beside it, with no
  verdict. Each arm logs its sums at the end of the run, including a run that stops.
- **The replica request.** Every row-adding arm, the four baselines and the three BKO arms, also
  fills the training split's own cell counts; a BKO arm fills it as it fills the gap request. Those
  rows alone train a classifier (train on synthetic, read against the no-augmentation arm, which
  trains on real) and are compared with the training split for fidelity. `compare` logs each
  replica arm's mean replica utility and fidelity, with no verdict.
- **One classifier.** Logistic regression (lbfgs, L2, C = 1) on standardised measured predictors
  and coded predictors one-hot encoded over their codebook. The weights reach the regression only.
  A record is called positive at a probability of 0.5 or more. A fit that does not converge stops
  the run.
- **Measures.** Recall, precision and AUC-PR over the whole test split and over the flagged
  subgroups, the age-sex subgroups that hold an underrepresented cell; a measure a scope cannot
  define is empty, and a criterion that reads an empty value refuses to judge. Fidelity is
  SDMetrics' quality report with every column pair counted. Distance to closest record is SDMetrics'
  `calculate_dcr`, read as a 5th percentile and a median, for the added rows and for the test rows,
  both against the training rows. Each arm's added and replica rows that equal a test record in
  every column are counted (`test-copies`).
- **A test split without copied records.** On each holdout every arm fills before any is scored. A
  test record that some arm's added or replica rows copy in every column leaves the test split every
  arm's utility is then scored on. No classifier is then scored on a record it trained on, and the
  comparison stays paired. The held-out ratio and the test rows' distance to closest record read the
  whole split.
- **Intervals.** Every contrast is paired by holdout and gets Nadeau and Bengio's corrected
  resampled t interval ([`nadeau-2003`](../../literature.md#evaluation-and-failure-modes)), the
  variance of the mean being (1/J + n2/n1) S^2, with n2/n1 the largest held-out-to-training ratio
  of any cell.
- **Criteria.** Validity is read as above; the rest read intervals. Primary: in both recall and
  AUC-PR over the whole test split, the candidate minus each comparator has an interval above zero
  (`E1`'s margin is zero). Secondary: the same over the flagged subgroups. Privacy: the candidate's
  added rows' 5th-percentile distance minus the privacy share of the test rows' has an interval
  above zero. An interval criterion is met when every interval it reads lies above zero, not met
  when one lies below zero at the family-wise level, each of its k intervals read at confidence 1 -
  (1 - c)/k for `E1`'s confidence c, and inconclusive otherwise.
- **Retries, then all or nothing.** An arm that refuses a request tries it again with a seed
  derived from the holdout seed and the attempt, up to `E1`'s attempts, and each retry is logged.
  A refusal on the last attempt stops the run with no output; a method output that breaks its
  contract stops it at once. A call that gets no response stops the run at once, as do a failed
  gap query and a spend at the ceiling.
- **The latest configuration is judged.** E1 judges its latest configuration: after a change to
  the system, the design or any configuration, E1 runs again, and only runs of the new
  configuration count. A configuration with one complete paid run (exit 0, call log ending
  `complete`) is judged on it. A resumed run whose log ends `complete` is one complete paid run,
  and its paid spend is the stopped log's `usd` plus its own `live_usd`. Runs of one
  configuration are combined into one result, and the method is fixed, as a reading, before a
  configuration's second paid run. Every run's readings
  are fixed before it starts. Every run keeps its Runs row in `docs/EVIDENCE.md`, a failed one
  with its reason; reporting earlier configurations elsewhere is optional.

## Rationale

The holdouts overlap, so a plain t interval over them is too narrow and narrows to nothing as
holdouts are added; Nadeau and Bengio's correction is the standard for repeated random holdouts.
Taking n2/n1 from the cell where it is largest keeps the flagged scope's interval as wide as its 5
test records against 18 training records need. Past about 30 holdouts the interval barely narrows,
since n2/n1 dominates, but corrected resampled tests replicate their verdicts across seed sets at
about 100 runs ([`bouckaert-2004`](../../literature.md#evaluation-and-failure-modes)), and the
baselines are cheap to rerun.

Requiring the candidate to clear every comparator is an intersection of tests, each at the
interval's level, so met needs no multiplicity correction and never picks the best comparator after
seeing results. E1 makes the stricter claim and judges ADASYN and the controls too. Not met is the
union: one interval below zero refutes the claim, so read at each interval's own level it fires
falsely, at the boundary, up to 30% of the time across 12 intervals under any dependence (26% for
independent ones). Reading each at 1 - (1 - c)/k holds the family's false not-met rate to one
interval's one-sided rate under any dependence. Intervals that decide neither way are neither a pass
nor evidence of no effect, hence a third verdict. The margin is zero: met claims the candidate is
reliably better than every comparator however small the gain, and each contrast's mean and interval
keep its size in view. No basis fixes a larger margin in advance, and at zero not met means reliably
worse than some comparator rather than short of an arbitrary gain. Recall is read at a fixed
threshold, which any arm that adds positive rows can move without ranking records better; AUC-PR,
which no threshold moves, must clear zero too, so a shifted threshold alone cannot pass.

Every arm fills the same per-cell request so that equal row counts hold by construction; letting
the BKO arm fill its own detected gaps would make the baselines' request depend on its output and
send SMOTE cells it refuses.

Three BKO arms ask whether the gap definition changes what generation adds; judging only `bko`,
fixed before the run, keeps one look at each criterion in each run, and comparing it with its own
ablations
would test the definitions against each other under criteria written for the baselines. Dropping
and generating again keeps equal row counts, as rejection sampling keeps CTGAN's and TVAE's
strata; clipping, which SDV applies to those two, would repair a record (ADR
`research/record-validation-design`) and pile rows on the split's bounds. The budget bounds a paid
loop: validity near 100% is expected, and three times the request leaves room for two candidates
in three to fall outside the split's range or the request before a refusal. A plain refusal lets
the retries apply, where a contract break would stop the run on one candidate. Summing over refused
attempts keeps validity from selecting on success.

The whole split resolves small effects. The flagged subgroups are what the project targets, but they
hold 5 positive test records per holdout, so their verdict is secondary and likely inconclusive, and
no other frame from the archive adds records of women with heart disease (ADR
`research/uci-evaluation-frame`). AUC-PR inside the underrepresented cells is not a scope: both
cells hold one label, so every scorer gets 1.0.

The gap request follows the threshold of `docs/RESEARCH.md` section 2 and is the same on every
holdout; on holdout 0 it equals `python -m bko.baselines <method> --rows 20`. The gap rows hold one
label in two cells, so they can neither train a classifier alone nor be compared with the training
split; a replica of the training split's cells is the conventional train-on-synthetic set
([`xu-2019`](../../literature.md#tabular-generators)). The BKO arms fill a replica too, so every
row-adding arm gets the same readings and the BKO arms a fidelity and replica utility reading beside
the baselines'. Their replica candidates enter the validity rate, which is read over every candidate
`bko.validate` sees, with the gap and replica counts reported apart. The replica asks for most of a
run's records, so the rate is mostly the replica fill's, and the gap fill's own rate shows in its
counts. The replica's spend dominates the run's, so `E1.spend_ceiling` is set above an estimate
recorded in E1's Configuration before the run.

Logistic regression is deterministic, takes weights and has no seed of its own; a random forest
scored the same on the baselines while its own seed moved subgroup recall by one of 5 positives. The
test rows are real records no generator saw, the natural real-data baseline for distance to closest
record, and the 5th percentile is the reading that catches near-copies
([`zhao-2021`](../../literature.md#evaluation-and-failure-modes)). The gap cells are cells of women
with heart disease, and a training row of the other label or sex already sits 1/14 (0.071) away
through that one column, against the 0.9 x 0.0225 = 0.020 the share asks of the added rows' 5th
percentile (the test rows' mean, measured 2026-10-16). A gap row falls under the share only by
nearly copying a training record of a woman with heart disease, so the share fails a generator that
copies or barely perturbs training rows and passes one that draws new records. Real records of the
gap cells agree (measured 2026-10-16 over `E1`'s holdouts, each training record of a gap cell
against the other training rows): their mean 5th percentile is 0.038 to 0.064, and the contrast
reads met at a share of 0.9 and of 1 whether each cell's records are taken once, cycled to its
requested rows or weighted to them; `<55 female 1` holds only 2 records, so this check rests mostly
on them. The share stays below 1 as room for the noise of a 5th percentile over 20 rows, which costs
nothing against near-copies. Records of the flagged subgroups' two labels, mostly without disease,
sit nearer the training rows and read inconclusive at 0.9; the gap request asks only for records
with disease. A refused holdout would leave the comparison unpaired. Default TVAE refuses the gap
request on 2 of the 100 holdouts (measured 2026-10-15): its rejection sampling draws 1,000,000 rows
and still lacks a few of the 18 rows asked of `<55 female 1`, a cell of 2 training records. That is
a property of one sampling seed, not of the split: with derived seeds, one of the two fills on its
third attempt and the other on its fourth. A retry keeps every holdout paired and keeps the request
the same; dropping the holdout would select splits on how one generator behaved. The run still stops
when every attempt is refused.

Paid generation has copied archive records in every column: in the completeness run of 2026-10-19, 5
of 20 records of one cell, 3 of them test records of that run's split. A copied test record would be
scored by a classifier that trained on it. The privacy reading measures distance to the training
rows, so it cannot see the copy. Removing such records from the test split for every arm keeps the
pairing and changes nothing on a holdout no arm copies. Counting alone would report the leak and
still judge on it. A call that gets no response is not a refusal of the request: retries come with
no delay, so they add no patience against an outage, and they would read an infrastructure failure
as the method's. Patience against an outage belongs to the SDK client that
`bko.generate.api.default_send` builds, whose retries (`MAX_RETRIES`) back off over minutes and
change no logged request: a stopped run resumes and repays its stopped call (ADR
`research/call-log-replay`), so the client still carries a brief outage. Budget and attempts bound the records asked, not the calls, since
each short cell's top-up is one call however few rows it asks, so a spend ceiling bounds the run.
Judging the latest configuration lets a change to the method or the design be judged on runs made in
it. Each run's readings are fixed before it starts, so each verdict is one look at its
configuration; a configuration may be chosen with an earlier configuration's results in view, and
every run stays in the register, so how many configurations were tried stays visible. Once a
configuration has several runs, combining them would use every paid draw of its generated rows. The
corrected interval's variance factor is 1/J + n2/n1, so more holdouts narrow it little once J is
large; combining would mainly average out the generated rows' draw-to-draw noise.

## Consequences

**Gained**: every criterion computed by code from one scores file; verdicts that do not depend on
which arm looked best; a harness that takes any frame in the evaluation schema.
**Accepted**: the 2-pager's 10% criterion is replaced (`docs/RESEARCH.md` section 9); `<55 female 1`
holds 2 records, none ever tested, yet takes 18 of the 20 gap rows, so its effect shows only through
other test records; subgroup verdicts are likely inconclusive at this size; distance to closest
record is weak evidence of privacy, blind to attacks that need no near-copy; `calculate_dcr` is not
a public SDMetrics export; baseline results on the real test split were in view when the readings
were fixed, no BKO result was; on a retried holdout, that arm's rows come from a derived seed rather
than the holdout seed; a BKO arm ignores the seed, and its retry repeats the same request; the test
split each arm is scored on depends on every arm's rows; a call with no response forfeits only that call,
and a spend at the ceiling forfeits the run; each BKO pipeline runs twice per holdout, for the gap request and
the replica request, so the three arms' paid generation grows with the holdout count and the
training split's size; the replica configuration's readings were fixed with `e1-1`'s BKO results in
view, and its generator was chosen for cost; a scores file from before the BKO arms filled a replica
holds no replica counts, so it is compared at the commit that wrote it; E1 judges no gap definition
directly, only what its guidance adds; the ablation arms' measures are reported, never judged; a
not-met verdict needs an interval below zero at the family-wise level, so more criteria read
inconclusive; that reading was fixed with baseline dry runs in view and no BKO result, and it only
ever moves not met to inconclusive, which favours the candidate; a met verdict certifies a reliable
gain, not a minimum size, and the zero margin, also fixed with baseline dry runs in view and no BKO
result, makes met easier to reach; a configuration may be chosen with an earlier configuration's
results in view, so each verdict is one look at its configuration, and every run stays in the
register so how many configurations were tried is visible.

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
| A privacy share of 1 | Fails no near-copy that 0.9 passes, and leaves no room for the noise of a 5th percentile over 20 rows | 2026-10-16 |
| Real gap-cell records as the privacy reference | `<55 female 1` holds 2 records, never held out, too few to serve as a reference | 2026-10-16 |
| A privacy share calibrated on records of the flagged subgroups | Mostly records without disease, nearer the training rows than the gap cells the request fills | 2026-10-16 |
| Reading not met at each interval's own level | A union of up to 12 tests: up to 30% false not-met at the boundary under any dependence | 2026-10-16 |
| Holm's step-down for not met | Its first step is Bonferroni's test, and not met needs one refuting interval, so it returns the same verdicts | 2026-10-16 |
| A not-met family per measure | Leaves the criterion's false not-met rate at up to twice one interval's | 2026-10-16 |
| An absolute margin of 0.02 | No stated basis, and not met then fires for a candidate better by less than it | 2026-10-16 |
| Judging every BKO arm against the criteria | Three looks at each criterion, the best picked after the fact | 2026-10-17 |
| Clipping a BKO candidate into the split's range | Repairs a record and piles rows on the split's bounds | 2026-10-17 |
| Loosening the added-row contract for the BKO arms | Compares arms that extend the split's support with baselines bound to it | 2026-10-17 |
| Counting test copies without removing them | Reports the leak and still scores utility on records the classifier trained on | 2026-10-19 |
| Dropping copied rows from an arm's output | The arm would read the test split, and equal row counts would break | 2026-10-19 |
| Retrying a call with no response as a refused attempt | Retries come with no delay, and an outage would read as the method's refusal; the SDK client's retries and timeout carry the patience | 2026-10-19 |
| An interval on the validity rate | The run's candidates are a census of what the arm generated, not a sample | 2026-10-19 |
| One call log per BKO arm | Three exclusive files and three spend totals for one run, and a ceiling split three ways | 2026-10-19 |
| Judging only the first paid run | Freezes the method at its first configuration, so a design change such as BKO replicas could never be judged | 2026-10-19 |
| BKO arms filling no replica | Leaves them no fidelity or replica utility reading beside the baselines' | 2026-10-19 |
| Validity over the gap fill's candidates only | One rule counts every candidate validation sees, replica candidates included | 2026-10-19 |
| A validity verdict per fill | Two looks at one criterion | 2026-10-19 |
| A control as a replica arm | It adds no rows, so its replica fit trains on nothing and score exits 1 | 2026-10-22 |

## Enforcement

- Owner: `src/bko/evaluate/`
- Pinned by: `tests/test_evaluate_criteria.py`, `tests/test_evaluate_experiment.py`,
  `tests/test_evaluate_run.py`, `tests/test_evaluate_classifier.py`,
  `tests/test_evaluate_utility.py`, `tests/test_evaluate_fidelity.py`,
  `tests/test_evaluate_privacy.py`, `tests/test_evaluate_scores.py`, `tests/test_evaluate_cli.py`,
  `tests/test_evaluate_guided.py`, `tests/test_baselines_records.py`,
  `tests/test_evaluate_copies.py`, `tests/test_evaluate_validity.py`,
  `tests/test_evaluate_bko_cli.py`

## Reversal Conditions

**A reading changes before the run that judges it**; a changed reading makes a new configuration
that E1 runs again (The latest configuration is judged).

**The holdout count and the margins are re-chosen if the frame changes before the run that judges
it**; this record and `E1` change together.

**`calculate_dcr` is replaced if an SDMetrics release removes
`sdmetrics.single_table.privacy.dcr_utils.calculate_dcr`.**

**The classifier changes if any holdout logs a fit that did not converge within `MAX_ITER`.**

**The BKO arms leave `E1.replica_arms`, all three or none, only with the spend estimate and
`E1.spend_ceiling` in E1's Configuration recomputed before a run, as they are whenever
`SETTINGS.model` or its `PRICES` row changes.**

**The gap request stops binding the BKO arms if their generation cannot target a cell**; the
request then becomes each BKO arm's realised rows per cell, and this record changes before the
run that judges it.

## Related

- [The UCI evaluation frame](uci-evaluation-frame.md)
- [The baselines](baseline-augmentation-design.md)
- [Python 3.14, without synthcity](../tooling/python-314-without-synthcity.md)
- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
- [Record validation: each field's number, bound and decimals, counted per check](record-validation-design.md)
- [A paid run's call log answers its rerun](call-log-replay.md)
