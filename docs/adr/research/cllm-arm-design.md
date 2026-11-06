---
status: accepted
topic: research
created: 2026-10-29
updated: 2026-11-06
---
# The CLLM arms: training records shown per cell, a boosting curator's drop rule, and an egress guard on every Messages arm

## Context

The BKO arms ask the generator for records guided by the ontology and the training split's bounds
and spreads, and never show it a training record (ADR
[`research/record-generation-design`](record-generation-design.md)). The direct arm draws from the
fused BKB with no model (ADR [`research/bkb-direct-generation`](bkb-direct-generation.md)). So the
arms separate the BKB from the LLM, but no arm asks the same model for records the way the
literature does without an ontology, and nothing separates what the ontology adds from what the
LLM adds. CLLM ([`seedat-2024`](../../literature.md#tabular-generators)) is that way: it shows an
LLM real training records as in-context examples, then curates the generated rows by a classifier's
learning dynamics, each row's mean confidence in its own label and its aleatoric uncertainty over
the classifier's checkpoints, metrics its paper says Data-IQ
([`seedat-2022`](../../literature.md#tabular-generators)) inspired. The paper reports its gains as
"especially noticeable in the low-data regime (i.e. n < 100)", with the gap to other generators
narrowing as n grows to 200, its largest training size.

CLLM's code, read on 2026-10-29 at commit `4ce879df21f859b7d1b637d63b938e915bd6927d` of
`github.com/seedatnabeel/CLLM`:

- `src/cllm/llm_gen.py`: `ic_samples=20`, and each prompt's examples are
  `example_df.sample(n=ic_samples, replace=True)` from the whole training set (lines 85-87).
- `src/cllm/curation.py`, `data_centric_curation`: an `XGBClassifier(n_estimators=nest)` at
  XGBoost's other defaults fit on the training rows, each generated row read at iterations 1 to
  `nest - 1` (`for i in range(1, nest)`); `get_groups` (lines 248-250) puts a row in `hard_train`
  when its confidence is at most `curation_ythresh` and its aleatoric uncertainty at most
  `curation_xthresh`.
- `notebooks/compute_downstream.ipynb`, which computes the paper's results: `nest=100`,
  `curation_ythresh=0.2`, `curation_xthresh = 0.15` and `temp = 0.9`; the curated set is
  `easy_train` and `ambig_train` together, so a row is dropped only when it is hard. Its downstream
  models train on the generated rows alone and are tested on real records.

Smolyak et al. ([`smolyak-2024`](../../literature.md#tabular-generators)) follow CLLM's prompt to
generate records per demographic group and show 20 randomly selected records of the group per
prompt (their section 2.2).

## Decision Drivers

* **Must separate the ontology from the LLM**: the arm uses `bko`'s model, effort, limits, send
  path and call log, so the prompt and the curation are what differ.
* **Must follow CLLM where the harness allows**, and name each departure.
* **Must replay byte for byte**, as every Messages arm does, so a row's fate may not depend on the
  batch it came in or on the calls in flight.
* **Must keep data held under a data use agreement off the API** unless the agreement permits it,
  for every Messages arm, since the BKO arms' bounds and percentiles are training values too
  ([`docs/RESEARCH.md`](../../RESEARCH.md#6-datasets)).
* **Must leave E1's and E3's arms and comparators unchanged.**

## Decision

- **Two arms.** `cllm` and `cllm-uncurated` (`bko.evaluate.experiment.CLLM_ARM` and
  `CLLM_RAW_ARM`) are Messages arms in `MESSAGES_ARMS`, each a `bko.evaluate.guided.GuidedArm`
  in the run's shared flight, so they share its call log, spend ceiling, concurrent waves, idle
  rule, replay, resume and rehearsal. They differ in curation alone. Neither is an arm of E1 or
  E3; they run in B3's named runs.
- **The spec.** `bko.evaluate.experiment.MessagesArm(definition, prescribes, shows, curation,
  over_ask, budget)` describes every Messages arm: `definition` None guides by cell alone, `shows`
  makes each call show training records, `curation` curates the accepted rows, and `budget` None
  reads `Experiment.budget`; a spec that both prescribes and shows is refused.
  `over_ask` is above 1 with `Curation.LIKELIHOOD` alone. `cllm` is
  `MessagesArm(None, shows=True, curation=Curation.CLASSIFIER, budget=CLLM_BUDGET)` and
  `cllm-uncurated` the same with no curation; `bko-curated` is
  `MessagesArm(Definition.SUBGROUP, curation=Curation.LIKELIHOOD, over_ask=2, budget=2 * E1.budget)`,
  which ranks each call cell's valid rows by their BKB likelihood (ADR
  [`research/bkb-candidate-curation`](bkb-candidate-curation.md), Arm). A `GuidedArm` takes `classifier` curation, which no arm but `cllm` has, and `likelihood`
  curation with its `over_ask`, which no arm but `bko-curated` has.
- **Guidance by cell alone.** `bko.generate.guidance.cell_guidance` gives each cell its training
  record count and bounds, with no gap, no finding and no ontology.
- **The prompt.** `bko.generate.prompt.render_shown`: the system text every call shares, then a
  user turn of `render`'s first two components (the cell and the records asked, then each numeric
  field's training-split bounds), then the sentence `SHOWN` and the shown records, one JSON object
  per line with every frame column in the frame's order. It reads no guidance level: the call
  log's settings name `SETTINGS`' level, and the prompt carries none of its components.
  `bko.generate.calls.request` refuses a request with both a skeleton and shown records.
- **The shown records.** `bko.evaluate.cllm.Shower` gives each call up to `SHOW` (20) training
  records: the call's cell's own, then the rest of its sex-and-label group (the other age band),
  then the rest of its label group (the other sex), each tier in a fresh permutation, none twice in
  one call. A cell holding 20 records or more shows its own alone. The generator is
  `numpy.random.default_rng(derived_seed(seed, "cllm"))` at the attempt's seed, made once per
  attempt and advanced call by call in planning order, never at send, so every in-flight count
  and a replay show the same records.
- **The curator.** `bko.evaluate.curation.classifier_scores` fits scikit-learn's
  `HistGradientBoostingClassifier` (`curator()`: 100 iterations, learning rate 0.3, maximum depth
  6, L2 regularization 1.0, at least 5 rows per leaf, no early stop, `random_state` 0) once per
  attempt on the holdout's training rows, the frame's features against its label, on one OpenMP
  thread. It reads each row at the first `STAGES` (99) stages: its confidence is the mean
  probability of its own label, its aleatoric uncertainty the mean of p(1 - p). A row is dropped
  when its confidence is at most `CONFIDENCE_AT_MOST` (0.2) and its aleatoric uncertainty at most
  `ALEATORIC_AT_MOST` (0.15) (`discards`; `classifier_discard` reads the rule off the scores).
  A training split holding one label is refused. Each row that passed validation keeps its two
  scores as its candidates line's `score`, so a threshold sweep reads them without a refit.
- **Where curation sits.** Each call's rows that pass validation are curated, and a row the curator
  drops gets the drop reason `Drop.CURATION`, after range and cell and before the quota
  (`bko.baselines.records.assess`), so it never takes a quota slot and the arm asks again for
  it. It is the outcome `dropped-curation` in the candidates file, the `dropped-curation` score
  measure and `dropped_curation=` in the `guided` and `validity` lines, for a classifier-curating arm alone.
  It counts as passed in the validity rate, as every drop after validation does.
- **Budget.** `CLLM_BUDGET`, both CLLM arms' records asked per requested row, is
  `max(3, ceil(1.5 / s))` for the least curator keep share `s` measured: 22, from `ceil(1.5 / 0.0707)`
  at B3's replica `<65/female/1` fill on Synthea (Consequences). A curating call asks
  `min(per_call, limit - asked)` records, so a cell never asks past its limit.
- **Partial fill.** A curating arm gives each cell its own budget, `budget` times the cell's
  quota, and gives up a cell, with a `shortfall` line naming its `quota=`, once its records asked
  could not cover its rows still short; the other cells go on, and the attempt returns the rows it
  kept with outcome `partial` and `short` in its params, checked against its quota less each
  given-up cell's missing rows. An arm that does not curate still refuses once the attempt's
  budget cannot cover its short cells. A budget shared across the attempt let one rejected cell ask
  the rest of it one record per call: `b3-synthea`'s first launch reached call 3159 on its second
  holdout and was stopped at $1.64. CLLM keeps whatever its curator passes, and a curator that
  rejects nearly every record of a cell is the effect B3 measures: on Synthea's real `bko` records
  the curator kept none of `>=65/female/1`'s on 3 of 100 holdouts' gap fills and 29 of their replica
  fills (`scratchpad/cllm_pilot_synthea_holdouts.py`), so no budget fills that cell.
- **The pilot**, a proxy on each dataset: the curator applied, per holdout, to the `bko`
  candidates kept in returned attempts, each holdout's training rows rebuilt by
  `bko.datasets.split.split` then `bko.evaluate.prepare.prepared` (with E3's removal on Synthea).
  They are BKO records, not records written from shown examples. On UCI, over
  `runs/g5-1/holdout-candidates.jsonl` (2026-10-29, `scratchpad/cllm_pilot.py`):

  | Fill | Cell | Rows | Kept | Share |
  |---|---|---|---|---|
  | gap | `<55/female/1` | 1803 | 1400 | 0.7765 |
  | gap | `>=55/female/1` | 197 | 192 | 0.9746 |
  | replica | `<55/female/0` | 3000 | 2717 | 0.9057 |
  | replica | `<55/female/1` | 200 | 196 | 0.9800 |
  | replica | `<55/male/0` | 4702 | 4297 | 0.9139 |
  | replica | `<55/male/1` | 3216 | 2661 | 0.8274 |
  | replica | `>=55/female/0` | 2700 | 2545 | 0.9426 |
  | replica | `>=55/female/1` | 1800 | 1499 | 0.8328 |
  | replica | `>=55/male/0` | 2398 | 2077 | 0.8661 |
  | replica | `>=55/male/1` | 5784 | 5110 | 0.8835 |
  | all | | 25800 | 22694 | 0.8796 |

  On Synthea, over `e3-a`'s and `e3-b`'s `runs/e3-2/` and `runs/e3-3/` candidates files
  (2026-10-30, `scratchpad/cllm_pilot_synthea.py`):

  | Fill | Cell | Rows | Kept | Share |
  |---|---|---|---|---|
  | gap | `>=65/female/1` | 1500 | 311 | 0.2073 |
  | replica | `<65/female/0` | 25500 | 25495 | 0.9998 |
  | replica | `<65/female/1` | 2301 | 399 | 0.1734 |
  | replica | `<65/male/0` | 24200 | 24005 | 0.9919 |
  | replica | `<65/male/1` | 4500 | 1446 | 0.3213 |
  | replica | `>=65/female/0` | 5500 | 5481 | 0.9965 |
  | replica | `>=65/female/1` | 499 | 102 | 0.2044 |
  | replica | `>=65/male/0` | 2700 | 946 | 0.3504 |
  | replica | `>=65/male/1` | 8900 | 8769 | 0.9853 |
  | all | | 75600 | 66954 | 0.8856 |

  UCI's least share, 0.7765, would ask for E1's 3; Synthea's, 0.1734 at the replica fill of
  `<65/female/1`, asks for `ceil(1.5 / 0.1734)` = 9. On Synthea the curator drops about four of
  five records in every sick women's cell, the flagged gap cell `>=65/female/1` included (1189 of
  1500, 1902 of 2301 and 397 of 499 above): each dropped record is one to whose own label, sick,
  the curator gives a mean probability of at most 0.2. The pilot measures the drops, not why the
  fit makes them.
- **Sampling.** No temperature, `top_p` or `top_k` is sent, as for every Messages arm (ADR
  `research/record-generation-design`).
- **B3's runs.** `b3-uci` and `b3-synthea` in `bko.evaluate.named_runs.RUNS` hold `METHOD_ARMS`,
  `bko`, `cllm` and `cllm-uncurated` (`B3_ARMS`), every row-adding arm filling a replica too,
  `cllm` as the candidate, `METHOD_ARMS` as comparators and no criterion: `b3-uci` on E1's holdouts
  over UCI, `b3-synthea` with E3's removal, copy rule and Synthea frame, plus `real`; their
  ceilings and estimates are in [EVIDENCE](../../EVIDENCE.md#b3-cllm-arms). The in-run `bko` keeps
  `Experiment.budget`.
- **The readings.** `cllm` and `cllm-uncurated` are each read by `compare --candidate`. The
  ontology against in-context records is `cllm` read `--beside` the same run's `bko`, and curation
  is `cllm` read `--beside` the same run's `cllm-uncurated`; both pairs share every holdout, split
  and call log, so neither needs a reading across files. The in-run `bko` is read `--beside` the
  judged `bko` of its dataset (E1's judged run, E3's merged scores), to check it against them.
  [`docs/EVIDENCE.md`](../../EVIDENCE.md#b3-cllm-arms) (B3) holds the commands.
- **Egress.** The CLLM arms send training records whole. Every `Source` in
  `bko.datasets.sources` names the data use agreement it is held under
  (`bko.datasets.sources.Agreement(name, sending)`), None for open data, and every pinned source's
  is None: UCI Heart Disease and UCI Diabetes 130-US Hospitals are CC BY 4.0 with no other term.
  `bko.datasets.sources.permits_sending` reads a blank `sending` as no permission.
  `bko.evaluate.data.Data.agreement` carries it into `score` and the app, and
  `bko.evaluate.guided_run.guided_arms` refuses data whose agreement permits no sending, for every
  Messages arm, with an `EvaluationError` naming the agreement, before the call log or the
  candidates file exists, and the guard covers `PAID_ARMS`, every arm that sends a request,
  `bko-direct-perturbed`'s rewrite calls among them (ADR `research/identifier-perturbation`);
  `python -m bko.generate` refuses the same before its call log, since its
  bounds and percentiles are training values too. Importing `bko.generate.api`, the one module
  that imports the SDK, holds the SDK's `anthropic` logger at INFO (`hold_sdk_logger`), so
  `ANTHROPIC_LOG=debug` logs no request body from any entry point.
- **Departures from CLLM**, each on purpose:
  - each call shows its cell's records first and draws without replacement within a call, where
    CLLM draws 20 from the whole training set with replacement;
  - each call asks for one cell's records, its sex and label fixed by the structured-output
    schema, with the training-split bounds stated, where CLLM asks for rows of the whole table;
  - the generator is `SETTINGS`' Haiku 5.5 at the API's default sampling, where CLLM's results use
    GPT-4 at temperature 0.9;
  - the curator is scikit-learn's histogram boosting at XGBoost's default learning rate, depth and
    L2 regularization, with at least 5 rows per leaf in place of XGBoost's `min_child_weight` 1;
  - the curated rows join the real training split, as every arm's rows do, where CLLM's
    downstream models train on generated rows alone;
  - UCI's training split holds about 238 records, above CLLM's largest n of 200.

## Rationale

A Messages arm built on the shared flight keeps what an arm-built `Call` would lose: concurrent
waves, the ceiling's reservations and the idle rule (ADR
[`research/concurrent-call-waves`](concurrent-call-waves.md)), and replay and resume (ADR
[`research/call-log-replay`](call-log-replay.md)). The spec states what each arm does, so the
harness reads fields rather than branching on arm names.

Each request asks for one cell's records, so the shown records start with that cell's own, as
Smolyak et al. show a group's own records. The flagged cells hold fewer than 20 training records
by definition, so the nearest groups fill the rest: the same sex and label first, then the same
label, which the schema fixes. Twenty is CLLM's `ic_samples`, Smolyak et al.'s count and
`per_call`. Drawing without replacement keeps a record from repeating in one prompt.

Fixed thresholds make a row's fate depend on the fit and the row alone, so curating per call keeps
the same rows a whole-batch curation would, at any in-flight count and on replay. CLLM's adaptive
threshold, 0.75 times the range of the batch's aleatoric values, would not.

The rule is a conjunction, but it acts almost as a confidence threshold: a row whose mean
confidence is c has a mean p(1 - p) of at most c(1 - c), so every row with confidence at or below
0.183 also meets the aleatoric bound, and only rows with confidence between 0.183 and 0.2 can be
kept by their aleatoric uncertainty.

Curation sits before the quota so a dropped row costs budget, not a slot. An attempt needs about
1/s records asked per row kept, so `1.5 / s` leaves half again for top-ups. For an arm that does
not curate, the budget only caps an attempt's asks before it refuses; for `cllm` it caps each
cell's asks before the cell is given up. Either way it is a cap, not a target, so the UCI arm,
whose least share needs less than 3, asks no more under 22. The pilot curates BKO
records because no shown-record run exists before B3; B3's own `dropped-curation` counts measure
the arm.

XGBoost is not locked, and scikit-learn's histogram boosting is. XGBoost's `min_child_weight` bounds
a leaf's sum of hessians, which for log loss is at most a quarter per row, so it needs at least 4
rows; histogram boosting bounds rows instead, and 5 stands in for it.

The guard covers every Messages arm because the BKO arms' bounds and percentiles are training values
too, and RESEARCH section 6 binds them. The agreement sits on the `Source`, beside the archive's
size and sha256, so a dataset's terms have one home.

## Consequences

**Gained**: B3 reads the ontology against in-context records at the same model, settings and
holdouts, and curation alone against the uncurated twin. No Messages arm sends records held under
an agreement that grants no permission to send them.
**Accepted**:
- Twenty training records leave the machine on each CLLM call: UCI Heart Disease's and UCI
  Diabetes 130-US Hospitals' are de-identified records under CC BY 4.0 (Diabetes's without
  `encounter_id` or `patient_nbr`), Synthea's are generated
  ([`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#generation), What leaves the machine).
- A model shown records can copy them; the `training` lines count an arm's rows equal to a
  training row and the `copies` lines its rows equal to a test record, as for every arm.
- The curator is fit on the same training rows the shown records come from.
- The budget rests on B3's `cllm` keep shares on Synthea
  (`runs/b3-3/holdout-candidates.jsonl`; each share is the rows kept or dropped for the quota over
  those plus the rows dropped by curation): 0.5370 overall against the proxy pilot's 0.8856, and
  under half in five fills and cells, replica `<65/female/1` 0.0707 (1437 of 20314), replica
  `>=65/female/1` 0.0976 (366 of 3750), gap `>=65/female/1` 0.1276 (1348 of 10564), replica
  `>=65/male/0` 0.1543 (2666 of 17276) and replica `<65/male/1` 0.2074 (4504 of 21718). The
  budget rule on its least share gives `ceil(1.5 / 0.0707)` = 22, `CLLM_BUDGET`.
- B3's runs asked at the proxy pilot's budget of 9, and their curating calls asked each cell's
  shortfall, as a non-curating arm's still do (`tests/test_evaluate_cllm.py` pins that split). A
  call log replays only against the code that asks its calls, so B3's `cllm` call logs replay at
  their run commits, the replay-commit rule of ADR
  [`research/call-log-replay`](call-log-replay.md), and B3's rows in `docs/EVIDENCE.md` hold at
  those commits.
- The call log names `SETTINGS`' guidance level for calls whose prompt reads none.
- The curator's fits are not XGBoost's: histogram boosting bins each feature, and its leaf bound
  counts rows.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| `CLLM_BUDGET` 9, from the proxy pilot's least keep share | B3's `cllm` on Synthea kept 0.0707 of replica `<65/female/1`, under the pilot's least share, so 9 records per requested row cannot cover that cell; the rule on B3's share gives 22 (Consequences) | 2026-11-02 |
| A curating call that asks its cell's shortfall | A cell short by one asks one record per call, so a cell the curator mostly rejects asks one record at a time; a full call capped by the cell's limit asks fewer calls for the same rows (Decision, Budget) | 2026-11-02 |
| A curated arm's budget shared across its attempt's cells | One cell the curator rejects asks the rest of the attempt's budget one record per call (`b3-synthea`'s first launch, stopped at $1.64) | 2026-10-30 |
| A curated arm that refuses past its budget, as every other arm does | On Synthea the curator keeps no record of `>=65/female/1` on many holdouts, so `b3-synthea` would stop partway (its rehearsal refused on holdout 2) | 2026-10-30 |
| An arm-built `Call` outside the shared flight | Loses concurrent waves, the ceiling's reservations and the idle rule (Rationale) | 2026-10-29 |
| Thresholds relative to a batch, such as CLLM's adaptive 0.75 times the aleatoric range | A row's fate depends on its batch, so a replay or another in-flight count keeps other rows | 2026-10-29 |
| Ranking a batch's rows and keeping the most confident | Not CLLM's rule | 2026-10-29 |
| CLLM's 20 records drawn from the whole training set with replacement | Repeats records in one prompt and shows a request for one cell mostly other cells' records | 2026-10-29 |
| XGBoost as the curator | Not locked; scikit-learn's histogram boosting is (Rationale) | 2026-10-29 |
| CLLM's temperature 0.9 | No Messages arm sends a sampling parameter (ADR `research/record-generation-design`) | 2026-10-29 |
| Training on the curated rows alone | Every arm's rows join the real training split, so B3 reads beside E1's and E3's arms | 2026-10-29 |
| An optional gap definition per arm, with branches on arm names for the rest | Each new kind of arm adds a branch at every reader | 2026-10-29 |
| A map from dataset to its terms | A second home for a dataset's terms beside its source | 2026-10-29 |
| A guard on the CLLM arms alone | RESEARCH section 6 binds the BKO arms' bounds and percentiles too | 2026-10-29 |
| Adding `cllm` to E1's comparators | Changes RESEARCH section 4's comparison set first | 2026-10-29 |

## Enforcement

- Owner: `src/bko/evaluate/`
- Pinned by: `tests/test_evaluate_cllm.py`, `tests/test_evaluate_curation.py`,
  `tests/test_evaluate_egress.py`, `tests/test_generate_imports.py`,
  `tests/test_generate_skeleton.py` (curation's place among the drop reasons),
  `tests/test_evaluate_experiment.py`, `tests/test_generate_prompt.py`,
  `tests/test_generate_guidance.py`, `tests/test_generate_api.py` (the SDK logger held at INFO),
  `tests/test_generate_cli.py` (the generate command's refusal)

## Reversal Conditions

**The budget is re-measured if B3's `cllm` keeps under half of the rows that reach its curator in
any fill and cell**, where `1.5 / s` passes 3; the candidates file's `dropped-curation` lines
against its `kept` and `dropped-quota` lines show it. `CLLM_BUDGET` rests on B3's Synthea run (Consequences).

**The thresholds follow CLLM's code if `notebooks/compute_downstream.ipynb` changes
`curation_ythresh` or `curation_xthresh`** in a later commit of its repository.

**A dataset's `Agreement` gains a `sending` permission only when the agreement's text grants it**;
the permission names the clause, and the egress list in `docs/DEVELOPMENT.md` then names that
dataset.

## Related

- [Record generation](record-generation-design.md)
- [E1's evaluation](e1-evaluation-protocol.md)
- [The UCI evaluation frame](uci-evaluation-frame.md)
- [The Synthea downstream design](synthea-downstream-design.md)
- [Concurrent call waves](concurrent-call-waves.md)
- [Call-log replay](call-log-replay.md)
- [Direct generation from the fused BKB](bkb-direct-generation.md)
