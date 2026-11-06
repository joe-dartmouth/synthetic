---
status: accepted
topic: research
created: 2026-11-05
updated: 2026-11-05
---
# The logic-only ablation draws each component of mapped concepts uniformly from the assignments the ontology allows

## Context

`bko-direct` draws each mapped concept present at its share, read from the fused BKB's dataset
priors, and records the `is_a` closure without reading it into a field (ADR
`research/bkb-direct-generation`). On E6's frame it was the strongest arm with the cell wholly
removed ([EVIDENCE.md](../../EVIDENCE.md#e6-diabetes-130-subgroup-removal)). How much of that
comes from the logical structure and how much from the probabilities has not been measured, and
the unit `compare-elk-logic-only` asks for a configuration in which presence comes from logic
alone, its draw fixed before any run.

Logic alone states constraints, not rates. Among a dataset's mapped concepts the table and the
releases give three:

- an `is_a` link between two mapped concepts: a record that shows the child shows the parent;
- a nested pair (`bko.gaps.shares.nested`): one concept's row codes lie strictly inside another's
  on the same field, so a record that shows the inner shows the outer;
- a label-only concept (`bko.gaps.shares.label_only`), every row of which reads only the label.

Any rate of presence beyond these must come from a choice. The three BKO gap definitions read the
dataset fragments differently. Per-subgroup and population-level grounding ask only whether some
record shows a concept, since instantiation reads structure and ignores probabilities (ADR
`research/dataset-fragment-encoding`, Rationale). The frequency-aware score compares a cell's
share of a concept with its label group's (ADR `research/frequency-aware-gap-score`).

## Decision Drivers

* **Must read no data share or prior**, so the contrast with `bko-direct` measures what the shares
  add.
* **Must add no rate the constraints do not imply**, so a drawn rate is not mistaken for knowledge.
* **Must keep `bko-direct`'s field selection, randomness and file**, so the two arms differ only in
  presence and closure.
* **Must be reproducible from a seed.**

## Decision

1. **Presence is a parameter of the direct sampler** (`Presence` in `bko.generate.logic`). At
   shares it is `bko-direct` (ADR `research/bkb-direct-generation`); at logic it is `bko-logic`,
   below. Field groups, record picks, the other fields and the candidates file are the direct
   sampler's for both.
2. **Consistent assignments.** Over the mapped concepts other than the label-only ones, an
   assignment of present or absent is consistent when every present concept's mapped `is_a`
   ancestors in its release (`bko.ontology.terms.ancestors`) are present and every present inner
   concept of a nested pair has its outer present.
3. **The draw is maximum entropy under those constraints.** Concepts joined by a constraint form a
   connected component (`bko.generate.logic.components`), and each component draws one
   assignment uniformly from its enumerated consistent assignments, with one `integers` draw on
   the sampler's seeded generator per component, in component order. A lone concept is present at
   1/2; a child and its parent draw each of their three consistent assignments at 1/3; a chain of
   three draws each of its four at 1/4. A component above `bko.generate.logic.MAX_COMPONENT`, 12
   concepts, is refused before any draw rather than enumerated.
4. **Closure is read into fields for `bko-logic` only.** The drawn set is closed upward under
   `is_a`, and every closed concept with a row of its own reads its present side. `bko-direct`
   keeps its rule: closure recorded, never read.
5. **Label-only concepts follow the label**, present in a target 1 cell and absent in a target 0
   cell, as in `bko-direct`; the team chose this so the arms differ in no other concept.
6. **E9's runs.** The named runs `e9-thin` and `e9-empty` take E6's frame, removal and holdouts at
   each fraction, with the arms `none`, `real`, `bko-direct` and `bko-logic`, no replica, the
   candidate `bko-logic` and the comparators `none` and `bko-direct`; their holdout slices are
   `e9-thina` to `e9-thinj` and `e9-emptya` to `e9-emptyj`. `bko-logic` is read against
   `bko-direct` on removed-cell recall, AUC-PR and distance to closest record, per holdout.
7. **E2 gets no logic-only run.** Per-subgroup and population-level grounding read no prior, so a
   logic-only prior leaves their gap lists as they are. The frequency-aware score has no
   logic-only analogue: a logic-only share is the same in every cell, so no cell reads below its
   label group. If E9's hierarchy comparison lists a differing link, E2's readings also run on the
   ELK-classified hierarchy, as the unit's Scope asks.

## Rationale

Uniform over the consistent assignments is the distribution of maximum entropy among those the
constraints allow, so it adds no information the ontology does not state. Components are
independent because no constraint joins them, and the product of uniform draws is the
maximum-entropy joint.

A fair coin per concept followed by closure looks simpler and is not neutral. With a child and its
parent, the coin pair gives both present at 1/2, the parent alone at 1/4 and neither at 1/4, so
the parent is present at 3/4: a rate the order of drawing then closing sets, which no source
states. The uniform draw gives each of the three assignments 1/3.

Closure is read into fields here, and not in `bko-direct`, because the two arms read different
evidence. `bko-direct`'s presence comes from data shares, and forcing an ancestor's row would
assert values no share supports. `bko-logic`'s presence comes from the ontology, which entails
every ancestor of a present concept, so a record that showed the child without the parent's row
would contradict the only source the arm reads. Where the data contradicts an ancestor, as
Diabetes 130 shows `MONDO:0005016` without `MONDO:0005300` in most records (ADR
`research/data-contradiction-handling`), fewer records match a forced pattern; field selection
then falls back to the label group, or refuses, and records either as `bko-direct` does.

Enumeration is exact, and 12 concepts bound it at 4,096 assignments per component; a larger
component would need a sampler whose own choices would enter the ablation.

## Consequences

**Gained**: a free, seeded arm whose contrast with `bko-direct`, on the same holdouts, removal and
field selection, measures what the data shares add to the logical structure; an E2 reading that
needs no run.
**Accepted**: rates such as 1/2 and 1/3 are not clinical, so `bko-logic`'s records show
combinations at frequencies no data supports, and its utility can fall below `none`; a component
above the bound stops the run before any draw, and a forced pattern no record matches refuses an
attempt, counted, not hidden; label-only concepts keep the label's rule in both arms, so neither
arm tests them.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A 0.5 coin per concept, then closure upward | Not uniform over the consistent assignments: a parent with one child is present at 3/4, a rate the closure order sets; see Rationale | 2026-11-05 |
| The closed world: a concept absent unless the label or a constraint forces it | Reads a missing statement as false, which neither OWL nor a BKO does, and draws every finding absent | 2026-11-05 |
| Label-only concepts drawn by the logic rule | The arms would then differ in the label's concepts too, and the label already fixes them | 2026-11-05 |
| Closure recorded and never read, as in `bko-direct` | The arm would then show a child without its parent's row, which the ontology it reads forbids | 2026-11-05 |
| A frequency-aware score under logic-only shares | Every cell reads the same share, so the score flags nothing | 2026-11-05 |

## Enforcement

- Owner: `src/bko/generate/direct.py` and the module `bko.generate.logic`, with the runs in
  `src/bko/evaluate/named_runs.py`
- Pinned by: `tests/test_generate_direct.py`, `tests/test_evaluate_direct.py`, and the modules
  `tests.test_generate_logic_only` (the draw reads no share, the 1/2, 1/3 and 1/4 rates, closure
  forced into fields, the refusal above 12 concepts), `tests.test_evaluate_logic_only` (both direct
  arms in one run) and `tests.test_gaps_logic_only` (every dataset prior at 0.5 leaves the
  per-subgroup and population listings on the Synthea fixture unchanged)

## Reversal Conditions

**The uniform draw gives way to a stated rate when the ontology fragments gain priors** (ADR
`research/bko-fragment-encoding`, Decision); the check is that record's Decision. **The bound of 12
concepts is revisited when a run stops on a component above it**; the check is the error naming
that component in the run's `scores.log`.

## Related

- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [A dataset becomes a fragment of one prior per mapped concept its records show](dataset-fragment-encoding.md)
- [A frequency-aware gap: a concept credibly under four-fifths of its share in the rest of its label group](frequency-aware-gap-score.md)
- [Each stage keeps a data contradiction and reports it; none forces the ontology onto the data](data-contradiction-handling.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
