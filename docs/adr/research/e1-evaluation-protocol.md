---
status: accepted
topic: research
created: 2026-10-15
updated: 2026-11-06
---
# E1's evaluation: repeated holdouts, one judged classifier, separation judged on paired intervals

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
  candidate (`bko`), the comparators, the replica arms, the threshold that flags a cell and the
  fill target each flagged cell is filled up to (both 20 in `E1`; a target below the threshold is
  refused), the two margins, the
  confidence, the attempts per fill, the BKO arms' budget per attempt, the validity threshold and
  the run's spend ceiling. It refuses a candidate that is not one of its arms, and a replica arm
  that adds no rows. `docs/EVIDENCE.md` (E1) states how each criterion is read. No flag changes a
  value: `score --log`, `--candidates`, `--replay`, `--resume` and `--rehearse` choose where calls
  and candidates are recorded and where calls are answered from, `compare --candidate` judges
  another arm as a dry run, never as E1's verdict, and `compare --beside` reads an arm beside
  another run's arm, a reading with no verdict, over the holdouts both files hold where the held-out
  rows match, their flagged-cell fingerprints match when both files hold one, and no arm in either
  file copied a test record, unless the experiment keeps copies or both files hold equal
  `shared-split` rows there, each scored on the split neither run copies (ADR
  `research/shared-test-split-reading`). A variant is a named run:
  `bko.evaluate.named_runs.RUNS` maps a run name to its `Experiment`, generator `Settings` and
  dataset, `e1` being `E1` at `SETTINGS` over UCI and the default, and `score` and `compare` take
  the name as a positional. E1's comparison set is the literal `METHOD_ARMS` in
  `bko.evaluate.experiment`, docs/RESEARCH.md section 4's six methods, and the methods that add
  rows (`ADDS_ROWS`) follow from it; a method `bko.baselines` adds later joins
  `bko.evaluate.run.ARMS` alone, so E1, E3 and every run built on them keep their arms. A run that
  reads a new method names it as its candidate with `comparators=METHOD_ARMS`, and another new
  method is read from the same scores with `compare --candidate`. A named run is a value fixed in
  code and committed, so it replays at its
  commit as E1 does, and
  the name does not reopen the ruled-out flags: no flag sets a reading, a holdout count or a
  setting. A run lists its own `criteria`, which may be none, and `compare` judges only those,
  reporting every measure beside them; a run with none is reported with no verdict. A paid run also
  stops on two rules beside the spend ceiling, each value held by its `Experiment`: its whole spend
  after `projection_after` holdouts, scaled to every holdout, reaching the ceiling (a resume counts
  the holdouts it replays and the spend it carries and replays, so every automatic resume projects
  as the fresh run would), and a call with no reply within `idle_limit` seconds, timed by
  `asyncio.timeout` around each awaited send, retries included, so that on expiry the run sends no
  later call, lets the calls in flight beside it land under their own idle limits, records their
  replies and spend, and raises `StopRuleError("idle")` (ADR `research/concurrent-call-waves`);
  `launch` starts it detached, kept awake and refused on battery power (`docs/DEVELOPMENT.md`,
  Evaluation). `launch` supervises the run: when a segment ends on an idle stop (F3), on a failure
  the API caused (F4: its call log's `end` line says `retryable`) or is killed by a signal with no
  `end` line (F6), it resumes the run into `<id>-resume-<k>` after a backoff, at most `--retries`
  times, under launch's own folder checks and resume guard; it never resumes a projection or
  ceiling stop, a deterministic failure, a crash with no `end` line, an interrupt or a run the
  operator stopped. A call abandoned by the idle rule or by a client timeout may still be billed,
  and its spend is not in the call log, so the ceiling does not count it; each automatic resume can
  repeat that exposure, bounded by the calls in flight per segment and by `--retries`.
  `launch --resume STOPPED` resumes a run the idle rule stopped into a new folder under the same
  checks; a run the projection or the ceiling stopped resumes only under a ceiling raised for that
  resume alone (`--ceiling`), above the run's own projection, since otherwise resuming would spend
  past the rule that stopped it. A guard holds this: each rule raises a `StopRuleError` naming it,
  the call log's `end` line records it under `stop`, and a resume refuses a `projection` or
  `ceiling` stop with no raised ceiling or one that does not cover its projection (ADR
  `research/call-log-replay`); `launch --resume` also refuses a log whose settings differ from the
  named run's, one whose `run` line names another run, and one a later segment already resumed,
  before it claims a folder. `launch --rehearse` first runs
  the whole run on canned replies (`bko.evaluate.rehearse`) into the run folder's `rehearsal`
  directory, so the stop rules, the spend lines and the folder roles are checked for free before any
  spend. A rehearsal bills each record asked at the judged run `e1-4`'s tokens per record
  (`INPUT_PER_RECORD` 301 and `OUTPUT_PER_RECORD` 211 in `bko.evaluate.rehearse`: `e1-4`'s Runs
  row's input and output tokens over its records asked), priced at the named run's model's rates, so
  its spend and projection read on the scale of a real run at that model; its spend is a wiring check, not an estimate. Those tokens per record are
  recomputed from a judged
  run's Runs row when `SETTINGS.model` or its `bko.generate.spend.PRICES` row changes.
- **Repeated holdouts.** Each of `E1`'s seeds gives one cell-stratified split from
  `bko.datasets.split`; every request on that split takes the same seed.
- **A training fraction.** An experiment may keep a share of each cell's training rows before any
  arm runs (`Experiment.training_fraction`, 1 in `E1`; `bko.datasets.split.subsampled`): per cell,
  the fraction of its rows rounded half up exactly, never below 2 nor above the cell, ranked by the
  sha256 of the seed, the tag `train-fraction` and the record id, so a smaller fraction's rows nest
  in a larger one's. The test split is never thinned. A fraction below 1 and a removal cannot both
  thin the training rows, so each run varies one. E5 reads the curve this draws.
- **The gap request.** Every arm is asked to fill each cell under the threshold up to the fill
  target, the same quota on every holdout; the threshold alone decides which cells are flagged, so
  a dose run that raises the target (E4) fills the same cells further and its flagged scopes pair
  with `E1`'s; a control adds no rows. Three BKO arms fill the same request, one per gap
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
  `bko.baselines.records.drops`, which tests: - a numeric value outside the training split's minimum
  and maximum, with the rule `check` uses; - then a cell the request does not ask for, or, in a
  call that carries prescriptions, any cell but the call's own; - then, in such a call, a record
  whose concepts match no line of the call left unconsumed; - then, for a curating arm (ADR
  `research/cllm-arm-design`), a row its curator drops; - then a row past its cell's quota. A
  candidate counts toward the cell its own values name, as `check` reads it. The prescribing arm
  `bko-skeleton` draws each attempt's lines once, with `bko.generate.direct.sample` at the
  request's seed, keeps them in a queue per cell and assigns them at wave planning, in plan order.
  Within one call a record is matched by concept set, not by position: it matches the first line,
  in the call's order, whose concepts equal its own and that no earlier kept record of the call
  consumed, and only a kept record consumes its match (`bko.baselines.records.assess`). A record
  outside the call's cell is dropped as `dropped-cell`, not `dropped-concepts`: the reason is its
  cell, so `dropped-concepts` counts only records that ignored their lines, and no other cell's
  quota is filled with a line drawn for this one. Every line asked and not consumed returns to the
  end of its cell's queue, in call order, for the next top-up, never redrawn. This differs from
  fusion's join, which drops a record whose values name another cell (ADR
  `research/gap-closure-design`), because E1's rows are bound by `check`. The arm returns the kept
  rows cell by cell in the quota's order and in generation order within a cell. It never clips,
  rounds or repairs one, and it calls `check` before it returns. An attempt asks for at most `E1`'s
  budget, three times the request's rows, and raises a plain `BaselineError` as soon as the rest of
  that budget cannot cover the rows still short; the retries below repeat it. It logs one `guided`
  line per attempt with its outcome, its call range, the records asked, the candidates generated and
  passed, those dropped for each reason and those kept, and for a prescribing arm the lines
  re-queued. Its refusal's message carries the same counts. A run stops, unretried, once its call
  log's spend reaches `E1`'s spend ceiling; the arm reads the spend before every call and logs
  `stop rule=ceiling` before the call it refuses. With calls in flight, it first reserves each
  lower-numbered in-flight call's worst-case spend and, if the spend plus the reservations reaches
  the ceiling, waits for those calls and checks again exactly, so the stop lands at the call an
  in-flight 1 run stops at. That
  spend is the call log's whole spend, a resumed log's replayed calls included. Every candidate an
  arm generates, refused attempts included, goes to the run's candidates file
  (`bko.evaluate.holdout_candidates`) with its holdout, fill, attempt, call (null for the direct
  arm of ADR `research/bkb-direct-generation`, which makes none; a row a `bko-direct-perturbed`
  rewrite call returned, changed or unchanged, carries that call, and only a row whose rewrite was
  refused, or whose holdout the distribution check refused, carries `call: null`, ADR
  `research/identifier-perturbation`), cell and outcome when
  the harness drains the holdout's gap fills and then its replica fills, so a later reading of
  single candidates needs no new run; its header, schema 3, names the frame and the sha256 of the
  run's data, and a run with no Messages arm writes no settings; a curating or ranking arm's line
  also holds the `score` its curator read; the reader also takes a schema 2 file, whose lines hold
  no score, and a schema 1
  file, written before frames, whose header names the UCI snapshot's sha256 instead and is checked
  against it, so `e1-4`'s own file is read as written; a stopped run's file lacks the
  stopped holdout's undrained fills, whose candidates live only in its call log; `score` requires
  the file whenever a BKO arm
  runs, and a replay rewrites it byte for byte.
- **Validity.** Each BKO arm's validity rate is `passed` summed over `generated` summed, from every
  `bko.validate` call it makes in the run: every holdout, and every refused and retried attempt. It
  fills two requests on each holdout, the gap request and the replica request, and the rate reads
  both. A refused attempt returns no `Baseline`, so the arm keeps its counts itself; the harness
  drains them after the gap fills and again after the replica fills, and writes them per holdout as
  `candidates` and `replica-candidates` scores (`generated`, `passed`, and one count per drop
  reason; `dropped-concepts` only for a prescribing arm, `dropped-curation` only for a curating
  arm). `compare` sums both over the holdouts
  for the rate, refuses a replica arm's scores without its replica counts, and logs each fill's
  counts and rate apart, with no verdict on a fill.
  The criterion reads `bko`'s rate: it is met when the rate reaches `E1`'s validity threshold, read
  as a point with no interval, and not met below it. A candidate dropped after validation stays in
  the rate, having passed every check. The other BKO arms' rates are reported beside it, with no
  verdict. Each arm logs its sums at the end of the run, including a run that stops.
- **The replica request.** Every row-adding arm, the four baselines and the three BKO arms, also
  fills the training split's own cell counts; a BKO arm fills it as it fills the gap request. Those
  rows alone train a classifier (train on synthetic, read against the no-augmentation arm, which
  trains on real) and are compared with the training split for fidelity. `compare` logs each
  replica arm's mean replica utility and fidelity, with no verdict.
- **A judged classifier and a reported one.** The criteria read a logistic regression (lbfgs, L2,
  C = 1) on standardised measured predictors and coded predictors one-hot encoded over their
  codebook. The weights reach the regression only. A record is called positive at a probability of
  0.5 or more, E1's cutoff; `Experiment.cutoff`, a float strictly between 0 and 1, sets it for
  another run, and E1 and E3 keep 0.5. A fit that does not converge stops the run. Every arm's
  augmented training set also
  fits a histogram gradient boosting classifier (`bko.evaluate.classifier.build_hgb`): the same
  one-hot coded predictors, measured predictors unscaled, log loss, learning rate 0.1, 100
  iterations, 31 leaves, no depth limit, a leaf of at least 1 record, no L2 penalty, every feature,
  255 bins, no native categorical features, early stopping off and no class weight. It takes the
  same weights, fits on one OpenMP thread, and is scored as the regression is, into
  `recall-hgb`, `precision-hgb` and `auc-pr-hgb` rows; `compare` logs each primary and secondary
  contrast of them as a `contrast model=hgb` line, with no verdict. Replica utility reads the
  regression only.
- **Measures.** Recall, precision and AUC-PR over the whole test split and over the flagged
  subgroups, the age-sex subgroups that hold an underrepresented cell; a measure a scope cannot
  define is empty, and a criterion that reads an empty value refuses to judge. Fidelity is
  SDMetrics' quality report with every column pair counted. Distance to closest record is SDMetrics'
  `calculate_dcr`, read as a 5th percentile and a median, for the added rows and for the test rows,
  both against the training rows. `bko.evaluate.privacy.nearest` computes it in blocks of arrays
  rather than through `calculate_dcr`'s cross merge of every row pair, and
  `tests/test_evaluate_privacy.py` holds the two equal bit for bit. Each arm's added and replica
  rows that equal a test record in every column are counted (`test-copies`). Each holdout's held-out rows also carry a
  `flagged-fingerprint` score, a hash of the cells it flagged, so two scores files show whether
  their flagged subgroups are the same test records.
- **A test split without copied records.** On each holdout every arm fills before any is scored. A
  test record that some arm's added or replica rows copy in every column leaves the test split every
  arm's utility is then scored on. No classifier is then scored on a record it trained on, and the
  comparison stays paired. The held-out ratio and the test rows' distance to closest record read the
  whole split. An experiment with `decontaminate` false keeps copied records in the split, counts
  them only, and writes a `copies-kept` held-out row on each holdout so a scores file says which
  rule scored it.
- **Intervals.** Every contrast is paired by holdout and gets Nadeau and Bengio's corrected
  resampled t interval ([`nadeau-2003`](../../literature.md#evaluation-and-failure-modes)), the
  variance of the mean being (1/J + n2/n1) S^2, with n2/n1 the largest held-out-to-training ratio
  of any cell, read after the training fraction and before any removal. A smaller fraction trains
  each classifier on fewer rows, so its ratio grows and its intervals widen as the correction asks;
  a removal thins one cell to make a gap, not the training set, so E3 reads the ratio before it.
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
- **E3's additions.** `Experiment.removal`, `Experiment.decontaminate`, the `real` arm, whose
  added rows are the removed training rows, `pairable`'s `allow_copies` and holdout slices joined
  by `merge` exist for E3 (ADR [`research/synthea-downstream-design`](synthea-downstream-design.md));
  the held-out ratio reads the split before any removal. E1 removes no training rows, removes copied
  test records and runs no `real` arm.
- **The latest configuration is judged.** E1 judges its latest configuration: after a change to
  the system, the design or any configuration, E1 runs again, and only runs of the new
  configuration count. A configuration with one complete paid run (exit 0, call log ending
  `complete`) is judged on it. A resumed run whose log ends `complete` is one complete paid run,
  and its paid spend is the stopped log's paid spend (its `end` line's `usd`, or the sum of its
  `reply` lines' `usd` when a killed process wrote no `end` line) plus its own `live_usd`. Runs of one
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
scored the same on the baselines while its own seed moved subgroup recall by one of 5 positives. A
linear model may not exploit the 18 and 2 rows the gap fill adds to its two cells, so a null result
on it alone does not show the rows carry nothing; the boosting model reads the same rows with no
seed. Histogram gradient boosting with early stopping off and every feature draws no random number
below 200000 training rows, and its scores were bit-identical at OpenMP threads 1 to 18 and at
`random_state` 0, 1 and None. Under the gate's parallel workers unbounded OpenMP threads crawl: the
median fit took 41749 ms against 78 ms on one thread under 12 processes. Both are planning
measurements of 2026-10-26 whose script was not kept. At these settings it classifies every
training row correctly where the regression does not (measured 2026-10-27 on holdouts 0 to 2 of the
pinned archive, each training split scored by `bko.evaluate.classifier.probabilities` on itself), and
its recall and AUC-PR read lower than the regression's on
every arm in both scopes (EVIDENCE E1, `e1-3-replay-2`). It responds to added rows more often: SMOTE
against no augmentation changes flagged-subgroup recall on 31 of `e1-3`'s 100 holdouts under it and
on 10 under the regression, read from `e1-3-replay-2`'s scores with

    uv run --no-sync python -c "import pandas as pd; s=pd.read_csv('runs/e1-3-replay-2/scores.csv'); f=s[(s.source=='augmented')&(s.scope=='flagged-subgroups')].pivot_table(index=['holdout','measure'],columns='arm',values='value'); [print(m, (f.xs(m,level='measure')['smote']!=f.xs(m,level='measure')['none']).sum(), 'of', len(f.xs(m,level='measure'))) for m in ('recall','recall-hgb')]"

Judged, its intervals would move no verdict: read as each criterion reads the regression's, its
primary and secondary contrasts in `e1-3-replay-2`'s `criteria.log` are inconclusive, as E1's are. A leaf size below the gap fill's smallest cell is read
as below that cell's 2 rows, so a leaf holds 1 record; read as below the whole 20-row fill, a leaf of
19 would pass, and it cannot hold the 2-row cell's rows apart, as scikit-learn's default of 20
cannot. The criteria still read the regression: they were fixed on it before any run, and judging a
second model adds a look at each criterion after `e1-3`'s results were in view. Replica utility
stays on the regression, its only reading beside fidelity.

The test rows are real records no generator saw, the natural real-data baseline for distance to closest
record, and the 5th percentile is the reading that catches near-copies
([`zhao-2021`](../../literature.md#privacy)). The gap cells are cells of women
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
the same; dropping the holdout would select splits on how one generator behaved. A rejection
sampler (CTGAN, TVAE, the Gaussian copula, sequential CART or the Bayesian network) still short of
a gap fill at its row limit on the last attempt fills partially
(`bko.evaluate.shortfall`): it keeps the rows it drew, checked against the quota less the
shortfall, each short cell gets a `shortfall` line and its `arm` line's params name the shortfall,
and the holdout stays paired at fewer rows for that arm alone. A structural shortfall, a stratum
holding fewer records than its method needs (ADR
[`research/baseline-augmentation-design`](baseline-augmentation-design.md)), fills partially the
same way on attempt 1, since no later attempt changes it: on a gap fill, any arm's shortfall that
falls on strata of fewer than 2 training records alone fills partially on the attempt that meets
it, a rejection sampler's at its row limit included; a shortfall that also falls on another
stratum, and a refusal that is not a shortfall, are retried as on any other fill. A replica fill still refuses, since
its scores read it as a copy of every training cell. E4's fill targets of 40 and 80 ask TVAE for more rows of a
2-record cell than its draws reach (rehearsals of 2026-10-31), and refusing would drop the dose run,
not one holdout. Any other refusal on every attempt still stops the run.

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
record is weak evidence of privacy, blind to attacks that need no near-copy; the test oracle
`calculate_dcr` is not a public SDMetrics export; baseline results on the real test split were in view when the readings
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
register so how many configurations were tried is visible; the boosting model's measures are
reported, never judged, so a gain only it shows moves no verdict, and it costs one more fit per arm
and holdout.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A partial fill on attempt 1 whenever the request asks a structural stratum | A shortfall that also falls on a stratum a retry can fill would keep fewer rows than the retry gives | 2026-11-02 |
| Reporting a projection or ceiling stop rather than resuming it | Loses the holdouts a stopped run paid for; a resume under a raised ceiling recovers them and still stops at a ceiling | 2026-11-02 |
| Raising a stopped run's ceiling in its named run | A commit changes the resumed run's `versions` line, so `merge` refuses it beside its sibling slices, and an E6 fraction's ceiling sets all 10 of its slices'; `--ceiling` raises one resume's alone | 2026-11-02 |
| Distances to closest record through `calculate_dcr` itself | Its cross merge builds a pandas frame of every row pair per chunk before it reduces them; blocks of arrays reduce as they go and return its values bit for bit (`tests/test_evaluate_privacy.py`) | 2026-11-02 |
| `METHOD_ARMS` and `ADDS_ROWS` derived from every `Method` | A method added for B1 would join E1 and E3, changing their arms and the runs that replay them | 2026-10-29 |
| A separate enum for the comparison set | `augment` takes a `Method`, so a second enum breaks its typing | 2026-10-29 |
| One free run per new method | Doubles the CTGAN and TVAE fits the runs share; a second method reads from the same scores | 2026-10-29 |
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
| A random forest as the judged model or a default model | Its own seed moves subgroup recall by one of 5 positives; ADR `research/classifier-panel` scores one at a fixed seed, opt-in and with no verdict | 2026-10-15 |
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
| Judging the boosting model | A second look at each criterion, chosen after `e1-3`'s results were in view | 2026-10-26 |
| A `model` column in the scores file | Changes every row and the columns `parse` reads; suffixed measures leave the regression's rows byte for byte | 2026-10-26 |
| The boosting model's native categorical features | One-hot coding over the codebook gives both models the same columns whatever codes a fit's rows hold | 2026-10-26 |
| Scaling the boosting model's measured predictors | Trees split on order, so scaling changes no split | 2026-10-26 |
| Unbounded OpenMP threads for the boosting fit | 41749 ms against 78 ms median per fit under 12 processes, for bit-identical scores | 2026-10-26 |
| A leaf size below the whole 20-row gap fill | Passes a leaf of 19, which cannot hold the 2-row cell's rows apart | 2026-10-26 |
| Positional matching of a prescribed record to its line | One skipped or reordered record shifts every later record onto the wrong line | 2026-10-28 |
| Keeping a prescribed record in another asked cell | Fills that cell's quota with a line drawn for another cell, and consumes no line of its own | 2026-10-28 |
| Pairing scores across different flagged scopes | Two runs that flag different cells score their flagged subgroups on different test records, so the flagged-cell fingerprints must match | 2026-10-28 |

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

**The equality test's oracle is replaced if an SDMetrics release removes
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
- [The Synthea downstream design: a 40-and-over frame, one thinned cell and an in-run reference](synthea-downstream-design.md)
