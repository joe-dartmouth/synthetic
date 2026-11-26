---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-26
---
# The `bko-capped` arm reads a pattern with fewer cell donors than a floor from outside the cell and skips a donor at a reuse cap per unit, and `bko-micro` averages each numeric field but `age` over three such donors

## Context

`bko-direct` reads each unit of a record, `age`, each field group and each free field, from one
training record: the cell's records that show the drawn pattern, else the label group's (ADR
`research/bkb-direct-generation`). A thin cell's few members then donate many times each. On E6's
thinned cell a fill of 1,250 rows reads every unit it can from the 19 records the removal keeps,
so the added rows repeat those records' values, which `privacy`'s field-group match attack and donor
reuse read (ADR `research/membership-inference-reading`).

Hot-deck imputation meets the same shape: a sparse pool over-uses a single donor. Andridge and
Little report that some hot decks therefore limit the number of times any donor is used, and that
the limit trades a gain in precision against bias from poorer matches
([`andridge-2010`](../../literature.md#missing-data-and-imputation)); their concern is variance,
and this record borrows the mechanism for exposure. k-anonymity asks that each released record be
indistinguishable from at least k-1 others on its quasi-identifiers
([`sweeney-2002`](../../literature.md#privacy)); that work is not fetched, so the statement is
this record's, and nothing here claims k-anonymity.

The fused shares that decide which concepts a record shows are noised by the label group's prior,
but the values are copied. A noised share bounds nothing about a value once that value is copied
whole, so a limit on exposure has to act on who donates.

Under a floor and a cap a numeric value is still one record's value, copied verbatim. Averaging a
numeric field over several donors that show the same pattern yields a value no single record holds
when their values differ. Microaggregation, as Domingo-Ferrer and Mateo-Sanz name it, releases
the mean of a group of at least k similar records in place of each record's value
([`domingo-ferrer-2002`](../../literature.md#privacy)); that work is not fetched, so the statement
is this record's. E6's frame holds eight count fields (`bko.datasets.diabetes.COUNTS`) and maps no
clause onto a numeric field, so on that frame every averaged field is a free count field.

## Decision Drivers

* **Must bound how often one record donates one unit's values in a fill**, so a thin cell's
  members stop donating every row.
* **Must never copy a pattern that fewer cell records show than a floor from the cell**, so a
  singleton pattern is not copied from its one holder.
* **Must leave `bko-direct`'s draw byte for byte**, so its runs re-read under identity checks.
* **Must still fill E6's cell**, whose label group holds far fewer records than a fill makes
  picks.
* **Must keep an averaged value on its drawn concept's side of each cutoff**, so a mean never
  shows a pattern the draw did not choose.
* **Must leave `bko-capped`'s draw byte for byte**, so `bko-micro`'s contrast with it reads the
  averaging alone.

## Decision

1. **Spec fields.** `DrawSpec.min_donors`, an int at least 1, default 1, and
   `DrawSpec.reuse_cap`, an int at least 1 or None, default None. Either off its default needs
   `mapped=True` and `presence` shares. Each is logged only off its default, as every spec field
   is.
2. **Arm.** `bko-capped` draws at `min_donors=2` and `reuse_cap=1`.
3. **Floor.** A unit whose cell holds fewer records showing the drawn pattern than `min_donors`
   reads the label group's matching records less the cell's own records, so a pattern below the
   floor is never copied from the cell. For `age` the label group is its records in the cell's
   age band, as for `bko-direct`.
4. **Cap.** Within one fill attempt, a donor that has given a unit's values `reuse_cap` times is
   skipped for that unit. The count is kept per (unit, donor): one record may still give each
   unit once, so a record's total donations can reach `reuse_cap` times the frame's unit count,
   which `donor-max` reads. The count spans every cell of one `sample` call, which is one fill
   attempt, and the next call starts it at 0.
5. **Exhaustion.** When the floor holds and every matching cell donor is at the cap, the pick
   reads the label group's matching records less those at the cap. When the floor fails, the
   pool of Decision 3 less those at the cap is the only pool. When the pool a pick reaches is
   empty, the draw raises `SamplingError`, and the harness retries the fill at a derived seed, as
   it retries any refused attempt.
6. **Pick.** One uniform draw from the pool per unit, as `bko-direct`'s pick, so the arm differs
   from `bko-direct` only in which records the pool holds.
7. **Relation to E10's PRAM arms.** `bko-direct-pram` redraws unmapped values after the draw (ADR
   `research/identifier-perturbation`); `bko-capped` changes who donates and leaves each copied
   value as its donor held it. The two act at different steps and can be read against the same
   reference arm, `bko-direct`.
8. **Microaggregation spec field.** `DrawSpec.microaggregate`, an int at least 2 or None, default
   None, follows `donors` and joins the fields that need `mapped=True` and `presence` shares off
   their defaults. A value below 2 is refused. It is logged only off its default, as every spec
   field is.
9. **Micro arm.** `bko-micro` draws at `min_donors=2`, `reuse_cap=1` and `microaggregate=3`:
   `bko-capped`'s limits, with k = 3 donors per mean.
10. **Averaged units.** A unit holding a numeric field other than `age` is averaged; every other
    unit, `age` among them, is picked as `bko-capped` picks it. On E6's frame the averaged units
    are its eight count fields, each a free field and so a unit of its own.
11. **Averaged pool.** An averaged unit's floor is `max(min_donors, k)`, and its pool is formed as
    Decisions 3 to 5 form a pick's pool (`_pick_limited` in `bko.generate.donors`). Its records
    are keyed into classes by the unit's coded values, the truth of each `AtLeast` clause on its
    numeric fields, and each numeric field's missingness. A class with fewer than k distinct
    record ids under the cap counts as empty, so a pool with no class left falls back to the
    label group as an empty pool does; that fallback is logged on the `fallback` line. When the
    label group's pool also has no class left, the draw raises `SamplingError`, as Decision 5
    does. The primary donor is uniform over the remaining classes' records, and the k - 1 others
    come from its class, uniform without replacement over its other distinct ids, so a pool that
    repeats a record still averages k distinct records.
12. **Values.** Coded fields come from the primary. A numeric field is the k donors' mean,
    rounded half-even to a whole number for an integer field and to the frame's
    `fraction_digits` otherwise. Since the k donors share each clause's truth and each field's
    missingness, their mean lies on the drawn concept's side of each cutoff and is missing
    exactly when theirs are.
13. **Weighted cap.** For an averaged unit the counts of Decision 4 are kept in k-ths: a pick of
    weight `w` is admitted when the donor's count for the unit plus `w` is at most
    `reuse_cap * k`. The primary of a unit holding a coded field weighs k, a whole use; every other
    donor of a mean weighs 1. In a unit of numeric fields only, each of its k donors weighs 1. At
    `bko-micro`'s cap of 1, a donor enters at most three means of one count field, and a donor
    already in two means is refused as a primary of a unit holding a coded field.
14. **Readers.** `Draw.averaged`, default empty, maps each averaged field to its k donor ids;
    `trace` writes it as `averaged` only when non-empty, so every other arm's trace line keeps its
    bytes; `privacy`'s donor reuse counts every id it holds; and `attribution` gives an averaged
    field the class `donor-mean` (ADR `research/record-explainability`, Decision 7).
15. **Micro log line.** When the spec field is set, the sampler writes one
    `micro-field holdout=%d fill=%s key=%s field=%s values=%d equal=%d primary=%d sse=%s sst=%s`
    line per cell and averaged field: the values written, those equal to at least one of their
    own k donors' values, those equal to the primary's, and the donors' within-group and total
    sums of squares.
16. **Relation to SMOTE.** SMOTE, as this record reads it, places a minority record at a random
    point on the segment from a minority record to one of its nearest minority neighbours, over
    every feature ([`chawla-2002`](../../literature.md#augmentation-versus-resampling), not
    fetched). `bko-micro` differs in four ways: its donors are chosen by the drawn pattern, the
    clause sides and missingness, with no distance; each numeric value is the equal-weight mean of
    k = 3 donors, not a random point between two; coded fields and `age` are copied from one
    donor, never interpolated; and every donor counts against a reuse cap.

## Rationale

A cap per (unit, donor) is the one form that both bounds reuse and fills E6's cell. Each row
makes one pick per unit, so a cap of 1 on a record's total donations would need a donor per pick,
the fill's rows times the frame's units, against the 3,749 records of the cell's label group
(`runs/e9-2/scores.log`); a cap per unit asks only that a unit's pool hold as many records as the
fill has rows, which the cell's label group does.
A floor of 2 is the smallest that keeps a singleton pattern out of the cell. A cap of 1 is the
tightest a uniform pick can meet. Raising at exhaustion keeps the cap a bound: a fallback to the
least-used donor would let the cap break silently. The retry at a derived seed is the harness's
existing path for a refused attempt, so the arm needs no new control flow.

`bko-micro` sits on `bko-capped`'s limits so that its contrast with `bko-capped` reads the
averaging alone. The weighted cap is what lets it fill E6's cell: each count field is a unit of
numeric fields only, so each of a mean's three donors weighs one third of a use, and a unit's
pool holds three times as many slots as it has records. A fill then needs as many records per
unit as it has rows, as `bko-capped` does. Keying donors by clause truth and missingness keeps a
mean on its concept's side of the cutoff without clamping it, since a mean of values on one side
of a threshold is on that side. On E6's frame that guard is inert, since no clause there reads a
numeric field; only the fixture test
`test_a_microaggregated_value_stays_on_its_concepts_side_of_the_cutoff` in
`tests/test_generate_micro.py` exercises it, on a `trestbps` cutoff at 130. Treating a class
short of k free donors as empty reuses the existing fallback path, so a thin class reads the
label group and is logged rather than refusing the fill. `age` is picked as `bko-capped` picks
it, so on E6's frame the averaging touches the count fields alone.

A mean of three narrows a count field's spread. During planning, means of 20,000 label-group
triples per count field kept about one third of the donors' variance, as the mean of three
independent draws would; that is a planning measurement, not a run of record, and each run's
`micro-field` lines carry `sse` and `sst`, from which the share is read per cell and field.

The mean does not protect everything. Coded fields and `age` are still copied whole from one
donor each. A donor that knows its own value and the mean learns the sum of the other two
donors' values. When the donors' values agree, as they often do for small counts, the written
value equals each donor's, which the `micro-field` line's `equal` count reads. The shared key
discloses the donors' common clause sides and missingness. No noise is calibrated to any
record, so the arm makes no differential-privacy claim, and its means are formed per row and
unit rather than over a partition of the released file, so it makes no k-anonymity claim.

## Consequences

**Gained**: a free, seeded arm whose contrast with `bko-direct` on the same holdouts reads what a
donor floor and a reuse cap do to the match attack and to donor reuse, at the cost the variant
rule reads on removed-cell recall.
**Accepted**: no differential-privacy claim; a singleton pattern of the cell can still be copied
from a record of the label group that shows it; a record can donate once per unit, so `donor-max`
may exceed the cap; a pool exhausted on every attempt refuses the fill.

**Gained** (`bko-micro`): a free, seeded arm whose contrast with `bko-capped` on E20's holdouts
reads what averaging the count fields over three donors does to the match attack, at the cost
the variant rule reads on removed-cell recall, with each mean's donors named in the trace.
**Accepted** (`bko-micro`): a count field keeps about one third of its donors' variance, a
planning measurement; the cutoff-side guard is exercised on a fixture only, since E6's frame has
no clause on a numeric field; coded fields and `age` stay verbatim copies; a donor learns the sum
of its co-donors' values; and no k-anonymity or differential-privacy claim is made.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A cap on a record's total donations | Cannot fill E6's cell: each row makes one pick per unit, so a fill needs more donor slots than the label group holds | 2026-11-25 |
| A least-used fallback at exhaustion | Gives no bound: the cap breaks whenever the pool runs out | 2026-11-25 |
| Distance-based rejection of a drawn row | Out of the unit's scope; it bounds similarity after the draw, not who donates | 2026-11-25 |
| A differential-privacy claim | Values are copied from records, and no mechanism here adds calibrated noise to them | 2026-11-25 |
| Averaging the numeric fields of mapped groups only | E6's frame maps no clause onto a numeric field, so the arm would average nothing there and draw `bko-capped`'s rows | 2026-11-26 |
| A count of one per mean donor | Each row would take three whole uses per count field, 3,750 donor slots for a 1,250-row fill against the label group's 3,749 records before any class splits them | 2026-11-26 |
| Two donors under a strict cap | A donor who knows its own value recovers the other's exactly from a mean of two, and the mean keeps about half the donors' variance rather than one third | 2026-11-26 |
| A k-anonymity claim | Means are formed per row and unit, not over a partition of the released file, and coded fields and `age` are copied from one record; see Rationale | 2026-11-26 |

## Enforcement

- Owner: `src/bko/generate/donors.py` with the spec in `src/bko/generate/draw_spec.py`,
  `src/bko/generate/direct.py`, the micro draw in `src/bko/generate/micro.py`,
  `src/bko/evaluate/direct_arms.py`, `src/bko/evaluate/runs_direct.py`
- Pinned by: `tests/test_generate_direct.py` (`DRAW_DIGEST`, `bko-direct`'s draw unchanged),
  `tests/test_generate_capped.py` (the spec's refusals and logging, the cap per unit, the floor,
  the barred cell donors, exhaustion and determinism), `tests/test_evaluate_capped.py` (the arm's
  params, its trace and the refusal of a float cap, donor reuse per unit, E14's runs and their
  reference arm), `tests/test_generate_capped_pinned.py` (a fill of E6's cell completes at
  both fractions on the pinned data, run with `--releases`), `tests/test_generate_micro.py`
  (the micro draw: the cutoff side, distinct donors, the weighted cap, the fallback, the
  rounding and the `micro-field` line), `tests/test_evaluate_micro.py` (`bko-micro`'s spec, its
  trace's `averaged` ids, donor reuse over every averaged id and the `donor-mean` attribution),
  `tests/test_evaluate_micro_runs.py` (E20's runs, their slices and their reference arm)

## Reversal Conditions

**The match attack's AUC of `bko-capped` does not fall below `bko-direct`'s** on E14's runs; the
check is E14's Result in `docs/EVIDENCE.md`. **A fill of E6's cell raises at every attempt**; the
check is E14's `scores.log`. **The match attack's AUC of `bko-micro` does not fall below
`bko-capped`'s** on E20's runs; the check is E20's Result in `docs/EVIDENCE.md`.

## Related

- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [Membership inference, verbatim matches and NNDR read an arm's added rows against a null arm, outside the removed cell](membership-inference-reading.md)
- [A perturbation arm rewrites the 19 unmapped Diabetes fields through the Messages API, beside a free PRAM arm, with per-call value quotas and a per-holdout reading of how far each moves the label-1 shapes](identifier-perturbation.md)
