---
status: accepted
topic: research
created: 2026-11-06
updated: 2026-11-08
---
# A perturbation arm rewrites the 19 unmapped Diabetes fields through the Messages API, beside a free PRAM arm, with per-call value quotas and a per-holdout reading of how far each moves the label-1 shapes

## Context

On Diabetes 130 with E6's target cell removed, `bko-direct` builds each record from blocks of real
training records' field values (ADR `research/bkb-direct-generation`). Every Diabetes field group
is a single field, and each field no field row maps to a concept is read from one record of the
cell or its label group, so a row is a mosaic of real records' values. The fields a patient could
be found by in this frame are the 19 that no field row maps to a concept:

- `race`;
- the eight count fields: `time_in_hospital`, `num_lab_procedures`, `num_procedures`,
  `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient` and
  `number_diagnoses`;
- the eight drug columns: `insulin`, `metformin`, `glipizide`, `glyburide`, `pioglitazone`,
  `rosiglitazone`, `glimepiride` and `repaglinide`;
- `change` and `diabetesMed`.

The question is whether rewriting those fields moves each record away from its source patients,
read as distance to closest record and membership inference (ADR
`research/membership-inference-reading`), without moving what the arm teaches a classifier. Post
randomisation (PRAM) changes each record's score on categorical variables by a known probability
mechanism, and an invariant matrix keeps the variable's distribution in expectation
([`gouweleeuw-1998`](../../literature.md#privacy)). Staab and colleagues report an LLM-based
anonymizer for text ([`staab-2025`](../../literature.md#privacy)); the work is not fetched beyond
its arXiv abstract, so each statement here about it is that work's report. `gouweleeuw-1998` is
read: its invariance condition is that the transition matrix leave the variable's distribution
unchanged. The free arm below redraws toward each field's label-group training marginal, which
equals that condition only when the rows it perturbs already follow that marginal.

Two arms rewrite the same rows. The reading, the quotas, the pilots, the re-pilot and its gate are
fixed here.

## Decision Drivers

* **Must leave every mapped concept, `age`, `sex` and the label unchanged by construction**, not
  by a check that refuses a row after the fact.
* **Must aim each column at the label-group distribution and report how close it lands**, since
  the arm exists to add rows a classifier can learn from.
* **Must give a free comparator**, so a gain over `bko-direct` is not credited to a language model
  that a redraw from marginals matches.
* **Must count every refusal and every unchanged row**, so a reading sees how much was rewritten.
* **Must stay inside the egress guard and the call log**, so a run replays.

## Decision

- **Fields.** The 19 unmapped Diabetes fields above, pinned as
  `bko.evaluate.perturb.DIABETES_UNMAPPED`. Where either arm is built, the fields no field row
  reads must equal that constant, or the run is refused; each arm refuses a field set holding
  `age`, `sex`, the label or a mapped field, and a rewrite call refuses to send `age`, `sex` or the
  label. No other field is rewritten.
- **LLM arm `bko-direct-perturbed`.**
  - It takes `bko-direct`'s rows for a holdout and sends them to the Messages API at `SETTINGS`'
    model in chunks of `settings.per_call` rows, cut by `bko.generate.calls.chunks`. A rewrite call
    carries a `row` index on each row and asks for the 19 fields per `row`.
  - The reply schema pins each coded field to its codebook and each count field to an integer
    `enum` over the range of the holdout's training split. A fractional field, which Diabetes
    lacks, is a plain `number`, bounded by the prompt and by the merge's range check.
  - The harness merges the reply by `row`. Mapped fields, `age`, `sex` and the label are not in
    the schema, so no reply can change them.
  - A call's reply keeps every original row when it decodes to nothing (`empty`), to another
    count of items than rows sent (`short`), to an item that is not an object with an integer
    `row` (`undecodable`), or to `row` indices other than those asked, in order (`order`); a
    permuted reply is refused. A row keeps its original when its item holds another key than
    `row` and the 19 fields, such as `age` or a mapped field (`keys`, the harness's key-set check),
    when the merged row fails `bko.validate` (`validate`, with the checks it failed), or when a
    count field leaves the training range (`bounds`). Each refusal is counted by reason, and so
    is each row the reply returned unchanged.
  - **Quotas.** Each call states the exact count of its rows that must hold each value category of
    each of the 19 fields. The categories come from the cell's label-group training rows: every
    code of a coded field, zero-count codes included, and for a measured field contiguous ranges
    in ascending order, each closed once it holds at least 1% of those rows, the remainder joining
    the last. Coded and integer fields' ranges tile the training range; a fractional field's
    ranges start at the next observed value, and `category_of` assigns a value between ranges to
    the next range. A holdout's rows are apportioned to
    the categories by largest remainder, ties to the earlier category, then spread across the
    holdout's calls: category k's T tokens sit at ((j + 1/2) / T + k / K) mod 1, sorted and cut
    into the calls' sizes, so a category's calls are evenly spaced and categories with one token
    land in different calls. The rule uses integers and fractions and no random draw, so a replay
    sends the same bytes (`bko.evaluate.quotas`).
  - The call opens with `bko.generate.rewrite.OPENING`: change each record so that its combination
    of values no longer matches the patient it was drawn from, a field may keep its value, and each
    field must hold exactly the counts listed. The counts follow `COUNTS_HEADER`, which names no
    outcome; a quota key outside the 19 fields is refused before the call.
  - The merge is unchanged. Each call logs `quota_off`, the sum over fields and categories of the
    quota count its final rows fall short of, and refuses nothing on it.
- **PRAM arm `bko-direct-pram`.** PRAM with theta = 0.5 on all 19 fields: each value is kept
  with probability 0.5 and otherwise redrawn from the holdout's label-group training marginal of
  that field, so each field redraws toward that marginal. Each field is redrawn on its own, so a
  redrawn row need not keep a coherent combination of its fields. It draws from
  `numpy.random.default_rng([seed, 1])` at the request seed, a stream apart from `bko-direct`'s
  draw at that seed, and makes no API call. It writes candidates lines as `bko-direct` does, so
  `StoredArm` rebuilds it.
- **Distribution reading, per holdout and per arm.** Let U be the unperturbed rows, P the arm's
  perturbed rows and R the label-1 training rows of the holdout, the cell's label group. Each
  holdout with a non-empty fill logs SDMetrics' column-shapes score over the 19 columns for P and
  for U against R, their difference, both column pair trends, each column's shape for P and for U,
  the share of rows breaking any of three Diabetes rules for P and for U, and the rows and values
  the arm changed (`docs/DEVELOPMENT.md`, Evaluation, holds the log formats). The three rules are:
  a drug not `No` with `diabetesMed=No`; `change=Ch` with `diabetesMed=No`; a drug `Up` or `Down`
  with `change=No`. Every real training row keeps all three. The reading never changes a row, and
  each arm keeps its own rows on every holdout.
- **Pilot.** Two named runs, `e10-pilot-thin` and `e10-pilot-empty` (`E10_PILOT_THIN` and
  `E10_PILOT_EMPTY`, built from `E9_THIN` and `E9_EMPTY`), each `E6_THIN` or `E6_EMPTY`
  with `arms=("none", REAL_ARM, DIRECT_ARM, "bko-direct-perturbed", "bko-direct-pram")`,
  `replica_arms=()`, `replica_rows=None`, `holdout_seeds=(0, 1)`,
  `candidate="bko-direct-perturbed"`, `comparators=(DIRECT_ARM, "bko-direct-pram", "none")` and
  `spend_ceiling=5.0`. Each run covers two holdouts and spends at most 5.0 USD. Full runs over
  E6's holdouts are not part of this decision.
- **Re-pilot.** `e10-pilot-thin` and `e10-pilot-empty` rerun at the commit that merges the reading
  and the quotas, under one tag. Its gate, every bar set before the run:
  - **G1, integrity:** both supervisor outcomes `complete`; the `versions` line names the tagged
    commit, clean; both replays byte-equal; both `rows` stores `matches_folder=true`; both
    `privacy` runs exit 0 against `bko-direct`.
  - **G2, schema:** refused rows at most 1% of the rows asked per fraction, and calls refused as
    `empty`, `short`, `order` or `undecodable` fewer than half of the fraction's calls.
  - **G3, untouched arms:** `none`, `real`, `bko-direct` and `bko-direct-pram` keep the first
    pilots' row sha256s on holdouts 0 and 1.
  - **G4, quotas took hold:** on every holdout, the rewrite's shapes and its `race` column shape
    each reach the midpoint between the first pilot's rewrite and the `real` cell rows on that
    holdout: shapes 0.913 and 0.910 thin, 0.918 and 0.916 empty; `race` 0.615 and 0.610 thin,
    0.659 and 0.656 empty. A planning measurement over the first pilots' rows stores, not a
    recorded run, sets each midpoint.
  - **G5, coherence kept:** the rewrite's share of rows breaking a rule is at most the first
    pilots' rewrite: 0.1596 thin and 0.1265 empty, a planning measurement over every reply row of
    `runs/e10-3/calls.jsonl` and `runs/e10-4/calls.jsonl`.
  - **G6, records moved:** on every holdout of both fractions, the rewrite's added-row `dcr-p5`
    exceeds `bko-direct`'s.
  - **Branches.** G1-G6 passing registers the full runs, their ceilings set from the re-pilot's
    measured USD per row asked. An operational G1 failure resumes under the paid-run protocol. G3
    failing is a regression to fix first. G2 failing meets the schema reversal below. G4, G5 or G6
    failing records the re-pilot alone and returns the prompt to design.
- **Egress.** The arm runs inside `guided_arms` over the shared `Flight` and `CallLog`, under the
  spend ceiling and the agreement refusal every Messages arm meets (ADR `research/cllm-arm-design`,
  Egress). A rewrite call sends its system prompt, each of the 19 fields' meaning with its codes
  or unit, each count field's training-split minimum and maximum, an accepted aggregate
  disclosure, the call's quotas, and each row's values of the 19 fields with its `row` index. The
  rows are the holdout's `bko-direct` fill, mosaics of rows of open Diabetes 130 data; no other
  BKO arm sends a record. Each quota is a count over a value or range of the cell's label-group training rows.
  A range's ends are real label-group training values, and each range holds at least 1% of the
  label group's rows, so a label group under 100 rows can expose a single row's value.
- **Privacy.** The `privacy` command reads `bko-direct-perturbed` and `bko-direct-pram` against
  `bko-direct` from their rows stores (ADR `research/membership-inference-reading`).

## Rationale

**Rewriting only unmapped fields leaves the concepts alone by construction.** A reply that cannot
name a mapped field, `age`, `sex` or the label cannot change one. The harness then has no
refusal list to maintain for them; a row can still trip the harness's key-set check, which
refuses any key beyond `row` and the 19 fields, `bko.validate`, which already reads range and
state, and the training-range check.

**The merge is by `row`, and a wrong row set voids the call.** A reply that skips, repeats or
invents a row index cannot be aligned to its input. Keeping all of that call's originals avoids
guessing which reply row belongs to which input.

**The reference is the label-1 training rows.** The arm adds rows for a label-1 cell, so the
distribution it should match is the label-1 rows of the training split, not the unperturbed
rows, which the rewrite is meant to move away from.

**A reading, not a gate.** A planning measurement, not a recorded run, over the first pilots'
rows stores and replies: the rewrite scored 0.846 to 0.861 shapes against R on all four holdouts,
where real data scores 0.974 to 0.990 (label-1 test rows 0.985, size-n resamples of R 0.986 to
0.990, `real` 0.974 to 0.976). Its `race` column scored 0.30 to 0.40, because the model changed
race on 99% of rows. The refusals came from the rewrite, not the check's reference. A per-holdout
gate also biases the paired contrast: a refused holdout adds an exact zero, and acceptance tracks
U's draw rather than P's (at thin, U swung 0.016 while P stayed at 0.846 to 0.850), so
`runs/e10-4-judged` reads `differing=0`. Reading every holdout keeps the contrast whole and
reports the cost beside it.

**Quotas from the label group.** A described marginal does not hold in 20-row calls: rounding
lifts rare values, and a planning measurement found `number_emergency>0` at 0.323 in the replies
against 0.105 in R. Quotas from each call's own rows keep, at thin, reusing the kept members'
values, the source of E8's cell leak. Exact counts from the label group make each call's rows sum
to R's marginals across the holdout. The quotas and the reading share R, so G4 reads whether the
quotas were followed, not fidelity; at thin, both arms repair `bko-direct`'s marginals toward R,
so thin contrasts mix repair with anonymization; at f=1.0, where U is already about R, the
contrast is the clean test of utility cost.

**Pair trends are reported, not gated.** A planning measurement, not a recorded run, found that
an ideal redraw of the 19 fields lowers SDMetrics' column pair trends further than it lowers
column shapes, and that shuffling the 19 columns on their own keeps shapes exact while lowering
pair trends. Any rewrite that breaks a source record's field combination lowers pair trends, so
pair trends cannot tell a useful rewrite from a damaging one; shapes can. The same measurement
found `bko-direct`'s 19-field combination equal to some training record's for fewer rows than
`real`'s.

**PRAM is the free comparator.** A keep-or-redraw step whose redraw follows the label-group
training marginal moves each field's marginal toward that one, and leaves it unchanged in
expectation when the rows already follow it, the invariance condition of
[`gouweleeuw-1998`](../../literature.md#privacy); it needs no model. A language model's rewrite
earns its cost only where it beats the redraw on distance or utility at the same shape. theta =
0.5 redraws about half the values of each row, and a redraw can equal the value it replaces, so
fewer than half change; with 19 fields a row's combination still rarely survives. Its own seed
stream keeps its keep and pick draws from replaying `bko-direct`'s draw at the same seed.

**The pilot is two holdouts per fraction.** It exists to read the arms' counts, refusals and
spend before a full run is planned. Two holdouts cannot carry an interval, so the pilot reports
and judges nothing.

## Consequences

**Gained**: a free comparator beside the language model's rewrite; every holdout carries the
arm's own rows, read and not gated; counts of refused and unchanged rows.
**Accepted**: a rewrite call sends rows that are mosaics of real encounters, and each count
field's training range; PRAM's redrawn rows need not keep a coherent combination of their 19
fields; a distorting rewrite enters the training set and is read, not refused; the quota block
discloses label-group value counts; thin contrasts mix repair with anonymization; the 19 fields
are the unmapped ones, so a
mapped field is never perturbed and a patient found by a mapped combination is not protected.

## Ruled Out

| Idea | Why ruled out | When |
|---|---|---|
| Refusing a rewritten row that changes a mapped concept, `age`, `sex` or the label | The schema omits those fields, so no reply carries them (Rationale) | 2026-11-06 |
| Gating on column pair trends | A redraw lowers them further than shapes, and a column shuffle that keeps shapes exact lowers them too, so they do not separate rewrites (Rationale) | 2026-11-06 |
| Comparing the perturbed rows with the unperturbed rows alone | The rewrite is meant to move away from them; the label-1 training rows are the target distribution (Rationale) | 2026-11-06 |
| Partial acceptance of a call whose `row` set differs | Alignment of reply to input is a guess (Rationale) | 2026-11-06 |
| A gate at shapes(U, R) minus a between-holdout `delta` | It biases the paired contrast with exact zeros, and refuses at f=1.0 by construction (Rationale) | 2026-11-08 |
| A gate at a real-data reference | It refuses the cell's own real rows at f=1.0, and its test-split variant leaks test data | 2026-11-08 |
| `delta` as a within or below label | `delta` is U's between-holdout noise, not the paired noise of P against U | 2026-11-08 |
| A described marginal in the prompt | 20-row calls round rare values up: `number_emergency>0` at 0.323 against 0.105 (Rationale) | 2026-11-08 |
| Quotas from each call's own rows | At thin they keep reusing the kept members' values, the source of E8's cell leak (Rationale) | 2026-11-08 |
| An unchanged prompt | The first pilots' rewrite scored 0.846 to 0.861 shapes against real data's 0.974 to 0.990 (Rationale) | 2026-11-08 |

## Enforcement

- Owner: `src/bko/evaluate/` and `src/bko/generate/`: `src/bko/evaluate/perturb.py` (the fields
  and the reading), `src/bko/evaluate/pram.py` (the PRAM arm), `src/bko/evaluate/perturbed.py`
  (the rewrite arm and its merge), `src/bko/evaluate/quotas.py` (the per-call value quotas) and
  `src/bko/generate/rewrite.py` (the rewrite call and its reply schema).
- Pinned by: `tests/test_evaluate_perturb.py` (the merge's refusals, the arm's calls and egress,
  the column lines, the three rules, and a reading below the unperturbed rows keeping the arm's
  rows), `tests/test_evaluate_pram.py` (the pinned fields, PRAM's stream, keep rate and marginals,
  and the reading of shapes, pair trends and columns against the label-1 rows),
  `tests/test_evaluate_perturb_replay.py` (a reading below the unperturbed rows replays byte for
  byte), `tests/test_datasets_diabetes_pinned.py` (every real training row keeps the three rules,
  under `--releases`), `tests/test_evaluate_perturb_runs.py` (the pilots and their ceilings),
  `tests/test_evaluate_quotas.py` (the categories, apportionment and spread of the quotas) and
  `tests/test_generate_rewrite.py` (the call, its quota block, its schema and its refusals).

## Reversal Conditions

Full runs in which `bko-direct-pram` matches `bko-direct-perturbed` on distance and utility at
the same column shapes would remove the language-model arm and keep the redraw. A re-pilot failing
G4, G5 or G6 returns the prompt to design. A pilot whose rewrite calls return a `row` set that
differs from the set asked on most calls would send the reply schema to a different format.

## Related

- [The CLLM arms and the egress guard](cllm-arm-design.md)
- [Membership inference and NNDR](membership-inference-reading.md)
- [Call log and replay](call-log-replay.md)
- [Direct generation from the fused BKB](bkb-direct-generation.md)
- [Diabetes subgroup removal](diabetes-subgroup-removal-design.md)
