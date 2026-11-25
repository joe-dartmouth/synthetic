---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# The `bko-capped` arm reads a pattern with fewer cell donors than a floor from outside the cell and skips a donor at a reuse cap per unit

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

## Decision Drivers

* **Must bound how often one record donates one unit's values in a fill**, so a thin cell's
  members stop donating every row.
* **Must never copy a pattern that fewer cell records show than a floor from the cell**, so a
  singleton pattern is not copied from its one holder.
* **Must leave `bko-direct`'s draw byte for byte**, so its runs re-read under identity checks.
* **Must still fill E6's cell**, whose label group holds far fewer records than a fill makes
  picks.

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
   which `donor-max` reads.
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

## Rationale

A cap per (unit, donor) is the one form that both bounds reuse and fills E6's cell. Each row
makes one pick per unit, so on E6's thinned cell 1,250 rows make 40,000 picks, and a cap of 1 on a
record's total donations would need as many donors against the 3,749 records of the cell's label
group (`runs/e9-2/scores.log`); a cap per unit asks only
that a unit's pool hold as many records as the fill has rows, which the cell's label group does.
A floor of 2 is the smallest that keeps a singleton pattern out of the cell. A cap of 1 is the
tightest a uniform pick can meet. Raising at exhaustion keeps the cap a bound: a fallback to the
least-used donor would let the cap break silently. The retry at a derived seed is the harness's
existing path for a refused attempt, so the arm needs no new control flow.

## Consequences

**Gained**: a free, seeded arm whose contrast with `bko-direct` on the same holdouts reads what a
donor floor and a reuse cap do to the match attack and to donor reuse, at the cost the variant
rule reads on removed-cell recall.
**Accepted**: no differential-privacy claim; a singleton pattern of the cell can still be copied
from a record of the label group that shows it; a record can donate once per unit, so `donor-max`
may exceed the cap; a pool exhausted on every attempt refuses the fill.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A cap on a record's total donations | Cannot fill E6's cell: each row makes one pick per unit, so a fill needs more donor slots than the label group holds | 2026-11-25 |
| A least-used fallback at exhaustion | Gives no bound: the cap breaks whenever the pool runs out | 2026-11-25 |
| Distance-based rejection of a drawn row | Out of the unit's scope; it bounds similarity after the draw, not who donates | 2026-11-25 |
| A differential-privacy claim | Values are copied from records, and no mechanism here adds calibrated noise to them | 2026-11-25 |

## Enforcement

- Owner: `src/bko/generate/donors.py` with the spec in `src/bko/generate/draw_spec.py`,
  `src/bko/generate/direct.py`, `src/bko/evaluate/direct_arms.py`,
  `src/bko/evaluate/runs_direct.py`
- Pinned by: `tests/test_generate_direct.py` (`DRAW_DIGEST`, `bko-direct`'s draw unchanged),
  `tests/test_generate_capped.py` (the spec's refusals and logging, the cap per unit, the floor,
  the barred cell donors, exhaustion and determinism), `tests/test_evaluate_capped.py` (the arm's
  params, its trace and the refusal of a float cap, donor reuse per unit, E14's runs and their
  reference arm), `tests/test_generate_capped_pinned.py` (a fill of E6's cell completes at
  both fractions on the pinned data, run with `--releases`)

## Reversal Conditions

**The match attack's AUC of `bko-capped` does not fall below `bko-direct`'s** on E14's runs; the
check is E14's Result in `docs/EVIDENCE.md`. **A fill of E6's cell raises at every attempt**; the
check is E14's `scores.log`.

## Related

- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [Membership inference, verbatim matches and NNDR read an arm's added rows against a null arm, outside the removed cell](membership-inference-reading.md)
- [A perturbation arm rewrites the 19 unmapped Diabetes fields through the Messages API, beside a free PRAM arm, with per-call value quotas and a per-holdout reading of how far each moves the label-1 shapes](identifier-perturbation.md)
