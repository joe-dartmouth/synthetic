---
status: accepted
topic: research
created: 2026-11-26
updated: 2026-11-26
---
# `bko-micro` averages each numeric field but `age` over three donors of one key class, on `bko-capped`'s floor and cap

## Context

`bko-capped` reads each unit of a record from one training record under a donor floor and a
reuse cap per unit (ADR [`research/sampler-donor-limits`](sampler-donor-limits.md)). Under those
limits a numeric value is still one record's value, copied verbatim, which `privacy`'s field-group
match attack and donor reuse read (ADR
[`research/membership-inference-reading`](membership-inference-reading.md)).

Averaging a numeric field over several donors that show the same pattern yields a value no single
record holds when their values differ. Microaggregation, as Domingo-Ferrer and Mateo-Sanz name
it, releases the mean of a group of at least k similar records in place of each record's value
([`domingo-ferrer-2002`](../../literature.md#privacy)); that work is not fetched, so the statement
is this record's. k-anonymity asks that each released record be indistinguishable from at least
k-1 others on its quasi-identifiers ([`sweeney-2002`](../../literature.md#privacy)); that work is
not fetched either.

E6's frame holds eight count fields (`bko.datasets.diabetes.COUNTS`) and maps no clause onto a
numeric field, so on that frame every averaged field is a free count field. No frame holds NaN in
a numeric field other than `age`: each loader drops, filters or bands a missing value.

## Decision Drivers

* **Must keep an averaged value on its drawn concept's side of each cutoff**, so a mean never
  shows a pattern the draw did not choose.
* **Must leave `bko-capped`'s draw byte for byte**, so `bko-micro`'s contrast with it reads the
  averaging alone.
* **Must leave `bko-direct`'s draw, every other arm's trace line and every other fill's
  attribution counts byte for byte**, so earlier runs re-read under identity checks.
* **Must still fill E6's cell**, whose label group holds 3,749 records against a 1,250-row fill
  (ADR `research/sampler-donor-limits`, Rationale).

## Decision

1. **Spec field.** `DrawSpec.microaggregate`, an int at least 2 or None, default None, follows
   `donors` and joins the fields that need `mapped=True` and `presence` shares off their defaults.
   A value that is not an int, a bool or a float among them, or is below 2 is refused. It is
   logged only off its default, as every spec field is. Set alone, it draws under a floor of 1 and
   no cap, so only Decision 4's effective floor and the classes act.
2. **Arm.** `bko-micro` draws at `min_donors=2`, `reuse_cap=1` and `microaggregate=3`:
   `bko-capped`'s limits, with k = 3 donors per mean.
3. **Averaged units.** A unit holding a numeric field other than `age` is averaged; every other
   unit, `age` among them, is picked as `bko-capped` picks it. On E6's frame the averaged units
   are its eight count fields, each a free field and so a unit of its own.
4. **Averaged pool.** An averaged unit's effective floor is `max(min_donors, k)`, and its pool is
   formed as ADR `research/sampler-donor-limits` forms a pick's pool (Decisions 3 to 5,
   `_pick_limited` in `bko.generate.donors`). Its records are keyed into classes by the unit's
   coded values and the truth of each `AtLeast` clause on its numeric fields; a record's class is
   computed once per (unit, record id) in a fill attempt, so a later pick only filters by the cap.
   A record holding NaN in an averaged field is refused with `SamplingError` naming the field,
   rather than averaged. A class with fewer than k distinct record ids under the cap counts as
   empty, so a pool with no class left falls back to the label group as an empty pool does. When
   the label group's pool also has no class left, the draw raises `SamplingError` naming the
   dataset, the frame, the holdout and the fill, and the harness retries the fill at a derived
   seed, as for any refused attempt. The primary donor is uniform over the remaining classes'
   records, and the k - 1 others come from its class, uniform without replacement over its other
   distinct ids, so a pool that repeats a record still averages k distinct records.
5. **Values.** Coded fields come from the primary. A numeric field is the k donors' mean, rounded
   half-even to a whole number for an integer field and to the frame's `fraction_digits`
   otherwise; a mean that rounds to zero is written 0, never -0. Since the k donors share each
   clause's truth, their mean lies on the drawn concept's side of each cutoff.
6. **Weighted cap.** For an averaged unit the counts of ADR `research/sampler-donor-limits`,
   Decision 4, are kept in k-ths: a pick of weight `w` is admitted when the donor's count for the
   unit plus `w` is at most `reuse_cap * k`. The primary of a unit holding a coded field weighs k,
   a whole use; every other donor of a mean weighs 1. In a unit of numeric fields only, each of
   its k donors weighs 1. At `bko-micro`'s cap of 1, a donor enters at most three means of one
   count field, and a donor already in any mean of a unit holding a coded field is refused as
   that unit's primary. A cap skip is counted at the pick's own weight, the primary's, as
   `_pick_limited` counts it.
7. **Readers.** `Draw.averaged`, default empty, maps each averaged field to its k donor ids, the
   primary's first; `trace` writes it as `averaged` only when non-empty, so every other arm's
   trace line keeps its bytes. `privacy`'s donor reuse counts every id it holds. `attribution`
   gives an averaged field the class `donor-mean` (ADR `research/record-explainability`,
   Decision 7) and writes a `donor-mean` row only for a fill whose arm's spec sets
   `microaggregate`, so every other fill's `attribution.csv` keeps its bytes.
8. **Log line.** When the spec field is set, the sampler writes one `micro-field` line per cell
   and numeric field but `age`, a field no draw averaged among them, after the `limits` line;
   its format and keys live in [`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#evaluation), and its
   `under` and `classless` counts split the `fallback` line's count by cause.
9. **Privacy reference arm.** E20's runs read `privacy` against `bko-capped`, so `bko-micro` minus
   `bko-capped`, the primary contrast, and `bko-direct` minus `bko-capped` are each read per
   holdout.
10. **Relation to SMOTE.** SMOTE, as this record reads it, places a minority record at a random
    point on the segment from a minority record to one of its nearest minority neighbours, over
    every feature ([`chawla-2002`](../../literature.md#augmentation-versus-resampling), not
    fetched). `bko-micro` differs in four ways: its donors are chosen by the drawn pattern and the
    clause sides, with no distance; each numeric value is the equal-weight mean of k = 3 donors,
    not a random point between two; coded fields and `age` are copied from one donor, never
    interpolated; and every donor counts against a reuse cap.

## Rationale

`bko-micro` sits on `bko-capped`'s limits so that its contrast with `bko-capped` reads the
averaging alone, which is why `bko-capped` is its privacy reference arm. The weighted cap is what
lets it fill E6's cell: each count field is a unit of numeric fields only, so each of a mean's
three donors weighs one third of a use, and a unit's pool holds three times as many slots as it
has records. A fill then needs as many records per unit as it has rows, as `bko-capped` does. A
unit holding a coded field fills tighter: each row uses k + k - 1 slots, the primary's k and one
for each other donor, against a record's k, so it needs (2k - 1) / k records per row, 5/3 at
k = 3.

The class key holds the cutoff guard: a mean of values on one side of a threshold is on that
side, so no clamp is needed. On E6's frame that guard is inert, since no clause there reads a
numeric field; it acts on a frame such as UCI's, whose `trestbps`, `chol` and `oldpeak` carry
cutoffs. Treating a class short of k free donors as empty reuses the existing fallback path, so a
thin class reads the label group and is logged rather than refusing the fill. Missingness is no
part of the key, since no frame holds NaN in an averaged field; a NaN that reaches a pick is a
defect upstream and is refused rather than averaged. `age` is picked as `bko-capped` picks it, so
on E6's frame the averaging touches the count fields alone.

A mean of three narrows a count field's spread: in expectation it keeps one third of the donors'
variance, as the mean of three independent draws does, and a draw of 20,000 label-group triples
per count field on E6's frame gives about that share. Each run's `micro-field` lines carry `sse`
and `sst`, from which the kept share is read per cell and field.

The mean does not protect everything. Coded fields and `age` are still copied whole from one
donor each. A donor that knows its own value and the written mean learns the sum of the other two
donors' values to within the mean's rounding: k times half the rounding step, 1.5 for a whole
field at k = 3. When the donors' values agree, as they often do for small counts, the written
value equals each donor's, which the `micro-field` line's `equal` count reads. The shared key
discloses the donors' common clause sides. One donor can enter up to three means of a count
field, so overlapping means share donors; `privacy`'s donor reuse counts each id each time it
enters a mean. The `averaged` ids live in the trace, an internal reading of a run folder that no
released fill carries. No noise is calibrated to any record, so the arm makes no
differential-privacy claim, and its means are formed per row and unit rather than over a
partition of the released file, so it makes no k-anonymity claim.

## Consequences

**Gained**: a free, seeded arm whose contrast with `bko-capped` on E20's holdouts reads what
averaging the count fields over three donors does to the match attack, at the cost the variant
rule reads on removed-cell recall, with each mean's donors named in the trace.
**Accepted**: a count field keeps about one third of its donors' variance; the cutoff-side guard
does not act on E6's frame, which has no clause on a numeric field; coded fields and `age` stay
verbatim copies; a donor learns the sum of its co-donors' values to within the rounding; a unit
holding a coded field needs 5/3 records per row; and no k-anonymity or differential-privacy claim
is made.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Averaging the numeric fields of mapped groups only | E6's frame maps no clause onto a numeric field, so the arm would average nothing there and draw `bko-capped`'s rows | 2026-11-26 |
| A count of one per mean donor | Each row would take three whole uses per count field, 3,750 donor slots for a 1,250-row fill against the label group's 3,749 records before any class splits them | 2026-11-26 |
| Two donors under a strict cap | A donor who knows its own value recovers the other's exactly from a mean of two, and the mean keeps about half the donors' variance rather than one third | 2026-11-26 |
| A k-anonymity claim | Means are formed per row and unit, not over a partition of the released file, and coded fields and `age` are copied from one record; see Rationale | 2026-11-26 |
| Missingness as a part of the class key | No frame holds NaN in an averaged field, so the part keyed nothing; a NaN is refused instead | 2026-11-26 |
| A `donor-mean` row in every fill's attribution counts | It would change the bytes of every existing fill's `attribution.csv`, though only a microaggregating arm can count one | 2026-11-26 |
| `bko-direct` as E20's privacy reference arm | `privacy` reads every arm against the reference, so the primary contrast, `bko-micro` minus `bko-capped`, would not be read per holdout | 2026-11-26 |

## Enforcement

- Owner: `src/bko/generate/micro.py` with the spec in `src/bko/generate/draw_spec.py`, the
  weighted count in `src/bko/generate/donors.py`, `src/bko/generate/direct.py`,
  `src/bko/evaluate/direct_arms.py`, `src/bko/evaluate/runs_direct.py`,
  `src/bko/evaluate/reference.py`, `src/bko/evaluate/attribution.py`
- Pinned by: `tests/test_generate_direct.py` (`DRAW_DIGEST`, `bko-direct`'s draw unchanged),
  `tests/test_generate_capped.py` (`CAPPED_DIGEST`, `bko-capped`'s draw unchanged),
  `tests/test_generate_micro.py` (the spec's refusals; the cutoff side through the class key,
  `test_a_means_donors_share_the_side_of_a_numeric_cutoff` and
  `test_a_microaggregated_value_stays_on_its_concepts_side_of_the_cutoff`; the coded-value key,
  distinct donors, the weighted cap and the refused primary, the fallback and its causes, the
  NaN refusal, the rounding and the `micro-field` line), `tests/test_evaluate_micro.py`
  (`bko-micro`'s spec, its trace's `averaged` ids, donor reuse over every averaged id, the
  `donor-mean` attribution and its row rule), `tests/test_evaluate_attribution.py` (`GOLDEN`,
  every other fill's `attribution.csv` bytes), `tests/test_evaluate_micro_runs.py` (E20's runs,
  their slices and their reference arm)

## Reversal Conditions

**The match attack's AUC of `bko-micro` does not fall below `bko-capped`'s** on E20's runs; the
check is E20's Result in `docs/EVIDENCE.md`. **A fill of E6's cell raises at every attempt under
`bko-micro`**; the check is E20's `scores.log`.

## Related

- [The `bko-capped` arm reads a pattern with fewer cell donors than a floor from outside the cell and skips a donor at a reuse cap per unit](sampler-donor-limits.md)
- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [Membership inference, verbatim matches and NNDR read an arm's added rows against a null arm, outside the removed cell](membership-inference-reading.md)
