---
status: accepted
topic: research
created: 2026-11-06
updated: 2026-11-06
---
# A perturbation arm rewrites the 19 unmapped Diabetes fields through the Messages API, beside a free PRAM arm, and a distribution check refuses a rewrite that distorts the label-1 shapes

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
read: its invariance condition, that the transition matrix leave the variable's distribution
unchanged, is the property the free arm below keeps for each field's label-group marginal.

Two arms rewrite the same rows. The distribution check, the pilot and its ceiling are fixed here
before any run.

## Decision Drivers

* **Must leave every mapped concept, `age`, `sex` and the label unchanged by construction**, not
  by a check that refuses a row after the fact.
* **Must keep each column's distribution close to the label-1 training rows'**, since the arm
  exists to add rows a classifier can learn from.
* **Must give a free comparator**, so a gain over `bko-direct` is not credited to a language model
  that a redraw from marginals matches.
* **Must count every refusal and every unchanged row**, so a reading sees how much was rewritten.
* **Must stay inside the egress guard and the call log**, so a run replays.

## Decision

- **Fields.** The 19 unmapped Diabetes fields above. No other field is rewritten.
- **LLM arm `bko-direct-perturbed`.**
  - It takes `bko-direct`'s rows for a holdout and sends them to the Messages API at `SETTINGS`'
    model in chunks of `settings.per_call` rows, cut by `bko.generate.calls.chunks`. A rewrite call
    carries a `row` index on each row and asks for the 19 fields per `row`.
  - The reply schema pins each coded field to its codebook and each count field to the range of
    the holdout's training split.
  - The harness merges the reply by `row`. Mapped fields, `age`, `sex` and the label are not in
    the schema, so no reply can change them.
  - A call whose set of `row` values differs from the set asked keeps every original row
    (`reason=rows`). A row that fails `bko.validate`, including a row with an extra key such as
    `age` or a mapped field, keeps its original (`reason=invalid`). Each refusal is counted by
    reason, and so is each row the reply returned unchanged.
- **PRAM arm `bko-direct-pram`.** Invariant PRAM with theta = 0.5 on all 19 fields: each value is
  kept with probability 0.5 and otherwise redrawn from the holdout's label-group training
  marginal of that field. It is seeded by the request seed and makes no API call. It writes
  candidates lines as `bko-direct` does, so `StoredArm` rebuilds it.
- **Distribution check, per holdout and per arm.** Let U be the unperturbed rows, P the arm's
  perturbed rows and R the label-1 training rows of the holdout. The check refuses the arm's
  rewrite for the holdout, and every original row is kept, when `shapes(P, R) <
  shapes(U, R) - delta`, `shapes` being SDMetrics' column-shapes score over the 19 columns.
  - `delta` is the 95th percentile, per removal fraction, of `|shapes(U_h, R_h) - shapes(U_{h+1},
    R_{h+1})|` over consecutive holdouts h and h+1 of E6's stored `bko-direct` rows.
  - `delta` is not yet fixed. The `perturb-threshold` command of `python -m bko.evaluate`
    computes it from the stage-1 rows stores and writes `threshold.csv` and `threshold.log`
    (`docs/DEVELOPMENT.md`, Evaluation); no run has applied a value.
  - Column pair trends are reported beside the check and gate nothing.
- **Pilot.** Two named runs, `e10-pilot-thin` and `e10-pilot-empty`, each `E6_THIN` or `E6_EMPTY`
  with `arms=("none", REAL_ARM, DIRECT_ARM, "bko-direct-perturbed", "bko-direct-pram")`,
  `replica_arms=()`, `replica_rows=None`, `holdout_seeds=(0, 1)`,
  `candidate="bko-direct-perturbed"`, `comparators=(DIRECT_ARM, "bko-direct-pram", "none")` and
  `spend_ceiling=5.0`. Each run covers two holdouts and spends at most 5.0 USD. Full runs over
  E6's holdouts are not part of this decision.
- **Egress.** The arm runs inside `guided_arms` over the shared `Flight` and `CallLog`, under the
  spend ceiling and the agreement refusal every Messages arm meets (ADR `research/cllm-arm-design`,
  Egress). A rewrite call sends rows of the holdout's `bko-direct` fill, which are mosaics of
  rows of open Diabetes 130 data; no other BKO arm sends a record.
- **Privacy.** The `privacy` command reads `bko-direct-perturbed` and `bko-direct-pram` against
  `bko-direct` from their rows stores (ADR `research/membership-inference-reading`).

## Rationale

**Rewriting only unmapped fields leaves the concepts alone by construction.** A reply that cannot
name a mapped field, `age`, `sex` or the label cannot change one. The harness then has no
refusal list to maintain for them, and the one a row can still trip is `bko.validate`, which
already reads range and state.

**The merge is by `row`, and a wrong row set voids the call.** A reply that skips, repeats or
invents a row index cannot be aligned to its input. Keeping all of that call's originals avoids
guessing which reply row belongs to which input.

**The reference is the label-1 training rows.** The arm adds rows for a label-1 cell, so the
distribution it should match is the label-1 rows of the training split, not the unperturbed
rows, which the rewrite is meant to move away from. `shapes(U, R)` is what the unperturbed rows
already score, and the rewrite is refused only when it scores lower by more than the change
between neighbouring holdouts' unperturbed scores, the noise a holdout's draw already carries.

**Pair trends are reported, not gated.** A planning measurement of 2026-11-05, which the pilot
supersedes, found that an ideal redraw of the 19 fields lowers SDMetrics' column pair trends
(0.9376 to 0.9450 against the unperturbed rows) further than it lowers column shapes (0.9869 to
0.9881), and that shuffling the 19 columns on their own scored 1.0000 on shapes and 0.9898 on
pair trends. Any rewrite that breaks a source record's field combination lowers pair trends, so
pair trends cannot tell a useful rewrite from a damaging one; shapes can. The same measurements
found `bko-direct`'s 19-field combination equal to some training record's for 0.39% of rows at
f = 1.0 and 0.56% at f = 0.985, against 1.10% for `real`. These are planning figures, not a
recorded run.

**PRAM is the free comparator.** A keep-or-redraw matrix whose redraw follows the marginal leaves that
marginal unchanged in expectation, the invariance condition of
[`gouweleeuw-1998`](../../literature.md#privacy), and needs no model. A language
model's rewrite earns its cost only where it beats the redraw on distance or utility at the same
shape. theta = 0.5 changes about half the values of each row, so a row's 19-field combination
rarely survives.

**The pilot is two holdouts per fraction.** It exists to read the arms' counts, refusals and
spend before a full run is planned. Two holdouts cannot carry an interval, so the pilot reports
and judges nothing.

## Consequences

**Gained**: a free comparator beside the language model's rewrite; a refusal that keeps every
original row, so a distorting rewrite never enters a training set; counts of refused and unchanged
rows.
**Accepted**: a rewrite call sends rows that are mosaics of real encounters; the check refuses a
whole holdout, not a row; `delta` is estimated from E6's `bko-direct` rows and holds for the
unperturbed arm's noise, not for a perturbed arm's; the 19 fields are the unmapped ones, so a
mapped field is never perturbed and a patient found by a mapped combination is not protected.

## Ruled Out

| Idea | Why ruled out | When |
|---|---|---|
| Refusing a rewritten row that changes a mapped concept, `age`, `sex` or the label | The schema omits those fields, so no reply carries them (Rationale) | 2026-11-06 |
| Gating on column pair trends | A redraw lowers them further than shapes, and a column shuffle that keeps shapes exact lowers them too, so they do not separate rewrites (Rationale) | 2026-11-06 |
| Comparing the perturbed rows with the unperturbed rows alone | The rewrite is meant to move away from them; the label-1 training rows are the target distribution (Rationale) | 2026-11-06 |
| Partial acceptance of a call whose `row` set differs | Alignment of reply to input is a guess (Rationale) | 2026-11-06 |

## Enforcement

- Owner: `src/bko/evaluate/` and `src/bko/generate/`. The modules `bko.evaluate.perturb` (the
  arms' runner and the distribution check) and `bko.generate.rewrite` (the rewrite call and its
  reply schema) are named here before they exist, and the `perturb-threshold` command and the
  pilot runs join `python -m bko.evaluate` and its named runs.
- Pinned by: nothing -- convention only. The tests for the arms, the check and the pilot runs
  land with those modules.

## Reversal Conditions

A pilot in which `bko-direct-pram` matches `bko-direct-perturbed` on distance and utility at the
same column shapes would remove the language-model arm and keep the redraw. A pilot in which the
check refuses every holdout of a fraction at the fixed `delta` would send the check to a different
reference. A pilot whose rewrite calls return a `row` set that differs from the set asked on most
calls would send the reply schema to a different format.

## Related

- [The CLLM arms and the egress guard](cllm-arm-design.md)
- [Membership inference and NNDR](membership-inference-reading.md)
- [Call log and replay](call-log-replay.md)
- [Direct generation from the fused BKB](bkb-direct-generation.md)
- [Diabetes subgroup removal](diabetes-subgroup-removal-design.md)
