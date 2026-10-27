---
status: accepted
topic: research
created: 2026-10-27
updated: 2026-10-27
---
# Direct generation draws each mapped concept at its fused share and reads fields from matching records

## Context

Every BKO arm of E1 generates through the Messages API, so its result mixes what the ontology
contributes with what the model's prior contributes; C1's Limits note that `cell`-level
candidates, whose prompt names no gap, closed the same per-subgroup gaps as `ontology`-level ones
([EVIDENCE.md](../../EVIDENCE.md#c1-gap-closure-after-fusion)). A sampler that draws records from
the fused BKB alone is free, seeded and sends nothing off the machine.

What the fused BKB holds fixes what such a sampler can draw. Every class component has one state,
the individual `a` (ADR `research/bko-fragment-encoding`); each dataset fragment is one prior per
mapped concept its records show, at that concept's share (ADR `research/dataset-fragment-encoding`);
and every `is_a` S-node has probability 1. There is no absent state, so a sampler cannot walk the
BKB choosing states; it can draw each concept's presence at a share read from the dataset priors,
and close the drawn set upward under `is_a`. The ontology fragments carry no prior, and the mapping
table says when a record shows a concept, never what values a concept implies (ADR
`research/ontology-mapping-design`). The arm therefore isolates the mapping table and the data's
concept shares, not the ontology's structure: closure is recorded and never read into a field.

## Decision Drivers

* **Must draw only from what the fused BKB and the mapping table hold**, so the arm measures them
  and no model prior.
* **Must let a cell draw a concept its own records never show**, or the arm cannot close a
  per-subgroup gap.
* **Must write a candidates file `bko.validate`, the drop rule and `bko.fusion` read unchanged.**
* **Must be reproducible from a seed**, with `bko.bkb` kept standard-library only (ADR
  `research/bkb-layer-source`).

## Decision

1. **Share.** A concept's share in a cell is `s(c) = (n_c p_c + 20 p_g) / (n_c + 20)`: `p_c` is
   the cell's dataset-fragment prior, `p_g` its label group's (every training record of the
   cell's label), each 0 where the fragment holds no S-node for the concept, and `n_c` the cell's
   record count. 20 is `UNDERREPRESENTED_BELOW` (`bko.generate.direct.GROUP_WEIGHT`).
2. **Concepts are drawn independently, except `restecg`'s.** Each concept but the label-only ones
   is drawn present when a uniform draw falls below its share, in sorted order. `HP:0001712`
   (`restecg` 2) is drawn at its share; `HP:0003115` (`restecg` 1 or 2) is present when it is, and
   otherwise drawn at `(s_EKG - s_LVH) / (1 - s_LVH)`, present outright when `s_LVH = 1`. The
   nesting is read from the rows: one concept whose one row's codes lie strictly inside another's
   on the same field (`bko.generate.direct.nested`).
3. **Closure upward** (`bko.ontology.terms.ancestors`, within each concept's release) is recorded
   in the draw log and never forces an ancestor's row: an ancestor with a row of its own, drawn
   absent, keeps its fields' absent side.
4. **Label-only rows** (`MONDO:0006715`, `HP:0005145`, every row of which reads only the label)
   follow the cell's label: present in a target 1 cell, absent in a target 0 cell, drawing nothing.
5. **Field groups.** The rows' fields join when one concept's rows read both, giving `cp`+`exang`,
   `trestbps` (its side of 130), `chol` (its side of 240), `fbs`, `restecg`, `oldpeak`+`slope`,
   `ca` and `thal` (`bko.generate.direct.field_groups`). Each group's fields are copied together
   from one record drawn uniformly from the cell's records that show exactly the group's drawn
   concepts, else from the label group's such records, else the attempt refuses
   (`SamplingError`). A concept drawn absent reads as the contrapositive of its rows: a matching
   record satisfies none of them. A group read from the label group is a fallback, recorded per
   field on the draw and logged per cell and field.
6. **The other fields.** `thalach`, which no row reads, comes from a cell record, else a label-group
   record; `age` from a cell record, else a label-group record in the cell's age band, so the record
   stays in its cell; each is a fallback when it comes from the label group. `sex` and the label
   come from the cell.
7. **Randomness.** One `numpy.random.default_rng(seed)`, held by `bko.generate.direct`, consumed in
   quota order, then record order, then per record the concepts in their drawing order and the
   fields in column order. A draw consumes a number only where chance decides it.
8. **The file.** A direct candidates file's call plan is `chunks(quota, per_call)` at
   `SETTINGS.per_call`, 20, the plan a Messages run of the same request writes, so fusion's line
   checks read it unchanged. Its header is schema 2, naming `generator` `direct` and the `seed`,
   with `definition` and `guidance` null (ADR `research/gap-closure-design`). `--log` names the
   draw log: one `draw` line per record with its call, cell, position, drawn concepts, closure and
   fallback fields.

## Rationale

A weighted share is the one way a cell can draw a concept it never shows without discarding what
it does show: the cell's prior alone could never close its own gap, and the label group's alone
would erase the cell. Weighting the group as 20 records reads the fused BKB's two priors at the
count the gap request already calls enough, so a cell at the threshold weighs its own records and
its group's equally, and an empty cell draws its group's shares. The label group, not every
record, keeps the label's share: the population pools both labels, so a target 0 cell would draw
target 1 findings at the pooled rate.

Drawing concepts independently is all the BKB supports, since it represents no co-occurrence (ADR
`research/dataset-fragment-encoding`, Consequences). `restecg` is the one exception the table
forces: its two rows nest, so independent draws would produce LVH without an abnormal EKG, a
pattern no record shows, and the conditional keeps both shares as marginals.

Copying a group's fields from one matching record, instead of drawing each value inside its side,
keeps every value a training value and every within-group combination one a record holds, so a
record passes validation by construction and the numeric fallback the unit asked for is the same
record draw at the label group. Closure is recorded rather than read into fields because the
ontology fragments carry no prior: forcing an ancestor's row would assert values no share
supports.

Run `g2-2` over E1's 100 holdouts
([EVIDENCE.md](../../EVIDENCE.md#g2-direct-generation-from-the-fused-bkb), G2) measured validity
1.0000 with 0 drops and no attempt refused; 0 test copies; replica quality, column shapes and
column pair trends 0.9295, 0.9681 and 0.8910; and a privacy margin of +0.0375 [0.0154, 0.0597],
added rows sitting closer to the training rows than `bko`'s.

## Consequences

**Gained**: a free, seeded arm that sends nothing off the machine and isolates what the mapping
table and the data's concept shares contribute; candidates `bko.validate` and `bko.fusion` read
unchanged. The arm `bko-direct` of the named run `g2` (`bko.evaluate.direct`) runs the sampler on
each of E1's requests, seeded by the request's seed, through the attempt bookkeeping a BKO arm uses,
writing each record to the per-holdout candidates file with a null call.
**Accepted**: the arm reads nothing the ontology fragments add beyond the closure it records, so it
cannot test the ontology's structure; within-group values are copies of training records, so the
arm's privacy and test-copy readings matter; a cell whose label group shows no record matching a
draw refuses the attempt.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Walking the fused BKB's states | Every class component has one state, so there is no absent state to choose | 2026-10-26 |
| The cell's prior alone | A concept the cell never shows has share 0, so no per-subgroup gap could close | 2026-10-26 |
| The population's prior as the second fragment | Pools both labels, so a target 0 cell draws target 1 findings at the pooled rate | 2026-10-26 |
| Each numeric value drawn inside its cutoff side from the field's own distribution | Breaks the within-group combinations records hold (`oldpeak` with `slope`, `cp` with `exang`) and needs a second fallback rule | 2026-10-26 |
| Forcing every closed ancestor's row | The ontology fragments carry no prior, so it asserts values no share supports; see Rationale | 2026-10-26 |
| Independent `restecg` concepts | Draws LVH without an abnormal EKG, a pattern no record shows | 2026-10-26 |
| A random generator inside `bko.bkb` | `bko.bkb` stays standard-library only (ADR `research/bkb-layer-source`) | 2026-10-26 |
| An `age` read from any label-group record when the cell holds none | The record would name another cell, and the join drops it as off-cell | 2026-10-27 |

## Enforcement

- Owner: `src/bko/generate/direct.py`
- Pinned by: `tests/test_generate_direct.py`, `tests/test_generate_candidates.py`,
  `tests/test_generate_cli.py`, `tests/test_fusion_pipeline.py`, `tests/test_evaluate_direct.py`

## Reversal Conditions

**The share and the concept draw change if ADR `research/dataset-fragment-encoding` comes to
represent co-occurrence**; the check is that record's Decision. **Closure is read into fields if
the ontology fragments gain priors** (ADR `research/bko-fragment-encoding`, Decision).

## Related

- [A dataset becomes a fragment of one prior per mapped concept its records show](dataset-fragment-encoding.md)
- [Ontology classes become single-state BKB components joined by is_a and exactMatch S-nodes](bko-fragment-encoding.md)
- [Validated records join the training records of their asked cell, and closure is read per gap definition](gap-closure-design.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
