---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# The no-mapping ablation draws bko-direct's records with an empty mapping table

## Context

`bko-direct` draws each mapped concept present at its share and copies each field group from a
training record that shows exactly the group's drawn concepts (ADR
`research/bkb-direct-generation`). Its records so carry two things at once: the mapping table,
which joins fields into groups and says which records show a concept, and the pooling of a cell's
records with its label group's, which every draw reads whatever the table says. The control
reading rule of [RESEARCH section 11](../../RESEARCH.md#11-direct-method-study) contrasts
`bko-direct` with an arm that keeps the pooling and drops the table, so the contrast shows what
the table carries.

## Decision Drivers

* **Must differ from `bko-direct` only in the mapping table**, so the contrast reads the table and
  nothing else.
* **Must keep `bko-direct`'s draw byte for byte**, since its runs of record are re-read under
  identity checks and digests check its draw.
* **Must keep each drawn record inside its cell.**
* **Must be reproducible from a seed and traceable**, as every direct arm is (ADR
  `research/record-explainability`).

## Decision

1. **The draw spec.** `DrawSpec` is a frozen dataclass in `bko.generate` with
   `presence: Presence = Presence.SHARES` and `mapped: bool = True`. `sample`, `DirectArm` and
   `redraw` take a `spec`. `bko-direct` and `bko-direct-pram` draw at the default spec,
   `bko-logic` at `presence=logic`, and `bko-nomap` at `mapped=False`.
2. **No table.** At `mapped=False` the sampler reads an empty mapping table: no concept, share,
   prior or field group. Every column but `age`, `sex` and the label is a free field, picked
   uniformly from the cell's records, else from the label group's.
3. **Cell columns.** `age` is picked from the cell's records, else from the label group's records
   in the cell's age band, as in `bko-direct`; `sex` and the label follow the cell.
4. **Randomness.** One `integers` call per column but `sex` and the label, per record, in
   `frame.columns` order, on the sampler's `default_rng(seed)`; no `random` call, since no concept
   is drawn.
5. **Everything else is `bko-direct`'s.** The donor fallback, the seed, the attempt budget, the
   candidates file and the trace are the direct arm's. The trace names no concept: a record's
   `concepts` and `closed` are empty, and it keeps a source id for every copied column.
6. **Logging.** A `DrawSpec` field is logged only when it differs from its default: in the `arm`
   line's params, as a ` name=value` token before `presence=` on the sampler's `direct` and
   `fallback` lines, and as `fallback_draws=` on the `direct arm=` line. The default spec logs
   nothing, so existing arms' lines keep their bytes.
7. **No run here.** This record adds the arm and no `RUNS` entry; the control runs are registered
   by their own unit.

## Rationale

An empty table is the one change that removes the mapping and keeps the pooling: with no
concept, every field becomes a free field, which `bko-direct` already draws from the cell, else
from the label group. The draw then reaches the same donor pools `bko-direct` reaches, field by
field instead of group by group, so any difference between the arms is the table's grouping and
matching. `age` keeps `bko-direct`'s banded fallback because a record whose age leaves the cell's
band names another cell, and the harness drops it as off-cell. A spec that logs only its
non-default fields keeps every existing line, params object and digest as its runs wrote it.

## Consequences

**Gained**: a free, seeded arm whose contrast with `bko-direct` on the same holdouts measures what
the mapping table adds to pooling; a spec type a later sampler option joins without a new
parameter on every caller.
**Accepted**: free fields drawn one by one break the within-record combinations a field group
keeps, which is the point of the ablation; the arm's concepts are empty, so its trace explains a
record by its source ids alone.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A `Presence` member for "no table" | Presence says how a concept is drawn; with no table there is no concept, and the logic arm would need a third combination | 2026-11-24 |
| Drawing concepts and discarding them | Spends draws that change the stream without changing a field, so the arm would differ in randomness as well as the table | 2026-11-24 |
| `age` from the whole label group | The row would leave its cell, and the harness drops an off-cell row | 2026-11-24 |
| A flag on `score` | An arm is the unit a run names and a trace reads; a flag would make one arm name two draws | 2026-11-24 |
| Logging every spec field on every line | Moves the bytes of every existing direct line and params object | 2026-11-24 |
| A separate sampler function | Duplicates the cell, pool, fallback and trace code the spec reuses | 2026-11-24 |

## Enforcement

- Owner: `src/bko/generate/direct.py` with the spec in `src/bko/generate/draw_spec.py`,
  `src/bko/evaluate/direct.py`, `src/bko/evaluate/trace_direct.py`
- Pinned by: `tests/test_generate_direct.py` (`DRAW_DIGEST`), `tests/test_evaluate_direct.py`,
  `tests/test_generate_no_mapping.py` (the uniform pick per column, the label-group fallback, the
  empty concepts and the logged spec)

## Reversal Conditions

**A spec field's default changes the order of `bko-direct`'s draws**; the check is
`DRAW_DIGEST` in `tests/test_generate_direct.py`. **The control reading rule stops naming
`bko-nomap`**; the check is the Contrasts line of
[RESEARCH section 11](../../RESEARCH.md#11-direct-method-study).

## Related

- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [The logic-only ablation draws each component of mapped concepts uniformly from the assignments the ontology allows](logic-only-ablation.md)
- [A record's explanation is the stored or re-derivable facts that determine it](record-explainability.md)
