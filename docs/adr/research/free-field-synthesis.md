---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# The `bko-coherent` arm redraws `bko-direct`'s free fields by sequential CART conditioned on the cell, the label and the mapped fields

## Context

`bko-direct` copies each free field, a column no field row reads, from its own training record
(ADR `research/bkb-direct-generation`, Decision 6), so one added row's drug fields, `change` and
`diabetesMed` come from different patients. On E6's Diabetes 130 removal, E10's Result reads the
rows `bko-direct` draws under `bko-direct-pram` against three coherence rules: 0.279112 of them
break one at f = 0.985 and 0.298290 at f = 1.0
([E10](../../EVIDENCE.md#e10-identifier-perturbation)), where every real training row keeps all
three (`bko.evaluate.perturb.rule_breaks`). ADR `research/data-contradiction-handling`,
Decision 2, counts the same three rules in a fill's added rows through `contradictions --rows`.

Copying every free field from one record would keep them coherent, but it would copy every
unmapped field of one patient whole, which `privacy`'s field-group match attack reads (ADR
`research/membership-inference-reading`). A conditional model can keep the fields' joint
structure without a joint copy.

## Decision Drivers

* **Must leave `bko-direct`'s draw byte for byte**, so its runs re-read under identity checks.
* **Must keep `bko.generate` free of `bko.baselines`** (`tests/test_generate_imports.py`), so the
  model runs outside the sampler.
* **Must keep each value a training value with a named donor**, so the trace and donor reuse
  read the arm as they read `bko-direct`.
* **Must draw the free fields given the cell, the label and the mapped fields**, so a row's
  drugs agree with its own diagnoses and with each other.

## Decision

1. **Arm.** `bko-coherent` draws exactly as `bko-direct` does, at the default draw spec. Its
   free fields are then redrawn by sequential CART, a `Change` passed through
   `DirectArm.fill(change=...)` (`bko.evaluate.direct`), as `bko-direct-pram`'s PRAM is.
2. **Trees.** The trees are fit once per fill on the holdout's whole training split, with
   `bko.baselines.classical.cart`'s settings and predictors. The fixed predictors are `age`,
   `sex`, the label and the mapped fields, in frame order. The free fields are visited in frame
   order, and each one's tree adds the free fields visited before it.
3. **Values.** Each free field of each drawn row takes the value of a training row drawn
   uniformly from the training rows in the drawn row's leaf; the donor is that training row's
   record id, one per field.
4. **Stream.** Every leaf draw reads `numpy.random.default_rng([request.seed, COHERENT_STREAM])`
   with `COHERENT_STREAM` 2, a stream apart from `bko-direct`'s draw at that seed and from PRAM's,
   stream 1 (`bko.evaluate.pram.PRAM_STREAM`). The rows are read in quota order, as the draw
   returns them.
5. **Outcome.** Each row's candidates line's `perturb` is `changed` when a free field differs
   from its draw, else `unchanged`.
6. **Trace.** `trace` re-draws the fill, re-runs the change at the fill's seed, refuses unless the
   changed rows equal the stored ones, and names each free field's leaf donor as its source.
   Donor reuse reads the same leaf donors.

## Rationale

Sequential CART synthesizes each field from a tree fit on the fields before it, and copies a
donor's value from the new row's leaf, which keeps each field's joint structure with its
predictors ([`nowok-2016`](../../literature.md#tabular-generators)). Reiter uses CART to replace
chosen values of released records while the others stay as collected
([`reiter-2005`](../../literature.md#tabular-generators)); that work is not fetched, so the
statement is this record's. Here the mapped fields, the cell and the label are the values kept,
so the drug fields are drawn given the diagnoses the BKB drew. The repository's CART
(`bko.baselines.classical`) already holds the trees and the leaf draw, so the arm reuses them
rather than a second model. A `Change` keeps the model in `bko.evaluate`, where `bko.baselines`
may be imported, and leaves the sampler and its digests untouched.

## Consequences

**Gained**: a free, seeded arm whose contrast with `bko-direct` on the same holdouts reads what a
conditional draw of the free fields does to the coherence counts, at the cost the variant rule
reads on removed-cell recall; each free field still names one training donor.
**Accepted**: a leaf donor may hold the other label, since the trees are fit on every training row;
donors are per field, so a row is still a mosaic of records, and a field's leaf may hold few
donors, which donor reuse reads.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A joint copy of one record's free fields | Copies every unmapped field of one patient whole, which the match attack reads | 2026-11-25 |
| Rule-based rejection of an incoherent row | Encodes three rules by hand, keeps no other joint structure, and can starve a thin cell | 2026-11-25 |
| A `DrawSpec` field holding a callable | Would put the model inside `bko.generate`, which may not import `bko.baselines`, and a callable cannot be logged | 2026-11-25 |

## Enforcement

- Owner: `src/bko/evaluate/coherent.py` with the draw in `src/bko/baselines/conditional.py` and
  `src/bko/baselines/classical.py`, `src/bko/evaluate/direct.py`,
  `src/bko/evaluate/direct_arms.py`, `src/bko/evaluate/trace_perturb.py`,
  `src/bko/evaluate/donor_attacks.py`, `src/bko/evaluate/attribution.py`,
  `src/bko/evaluate/runs_direct.py`
- Pinned by: `tests/test_generate_direct.py` (`DRAW_DIGEST`, `bko-direct`'s draw unchanged),
  `tests/test_generate_imports.py` (`bko.generate` imports no `bko.baselines`),
  `tests/test_baselines_cart_digest.py` (`cart`'s fills unchanged by the shared leaf draw),
  `tests/test_baselines_conditional.py` (same-leaf donors, given columns kept, determinism and
  the refusals), `tests/test_evaluate_coherent.py` (the arm's params and pinned fill, the mapped
  fields and label equal to `bko-direct`'s, its lines and log lines, the trace's leaf donors and
  its refusal, the attribution class, donor reuse and E17's runs),
  `tests/test_evaluate_coherent_pinned.py` (a fill of E6's cell completes at both fractions on
  the pinned data, each coherence rule breaking in at most as many rows as under `bko-direct` at
  the same seed, run with `--releases`)

## Reversal Conditions

**The coherence counts of `bko-coherent` do not fall below `bko-direct`'s** on E17's runs; the check
is E17's Result in `docs/EVIDENCE.md`. **A fill of E6's cell cannot complete**; the check is E17's
`scores.log`.

## Related

- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [A perturbation arm rewrites the 19 unmapped Diabetes fields through the Messages API, beside a free PRAM arm, with per-call value quotas and a per-holdout reading of how far each moves the label-1 shapes](identifier-perturbation.md)
- [Each stage keeps a data contradiction and reports it; only an ablation arm forces the ontology onto the data](data-contradiction-handling.md)
- [A record's explanation is the stored or re-derivable facts that determine it](record-explainability.md)
- [Membership inference, verbatim matches and NNDR read an arm's added rows against a null arm, outside the removed cell](membership-inference-reading.md)
