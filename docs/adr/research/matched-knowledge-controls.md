---
status: accepted
topic: research
created: 2026-11-26
updated: 2026-11-26
---
# The matched knowledge controls change one input of bko-direct's draw at a time

## Context

`bko-direct` reads two things at once: K, the knowledge the study isolates, and the training
records. K is a dataset's mapping table (its `FieldMapping` rows, `bko.evaluate.data.field_rows`)
plus the pinned ontology releases those rows name. E9, E12, E13 and E15 already hold related
controls: `bko-logic` drops the data shares, `bko-nomap` drops the table, `label-bootstrap` copies
whole label-group records into the cell, `prior-shift` adds no rows, and `bko-tree` reads the
hierarchy into the shares. None of them separates K's choice of which fields to pool from the
pooling itself, and a change in donor selection or field dependence can also change predictive
utility.

Two facts about `bko-direct` on E6's frame bound what a control can show:

- At full removal `bko-direct` draws the same distribution as `bko-nomap`: every share is the
  label group's and every unit falls back to the label group (ADR
  [`research/no-mapping-ablation`](no-mapping-ablation.md), Consequences). Measured on holdout 0,
  every mapped field's share was within 0.004 of the label group's.
- `bko-direct` pools at `GROUP_WEIGHT` 20 only on the 12 mapped fields, through the concept shares
  (`bko.gaps.shares.SharePart.share`); `age` and the 19 free fields are copied own-first, from the
  cell's records, else the label group's (`_Plan.draw` and `_pick` in
  `src/bko/generate/direct.py` and `src/bko/generate/donors.py`).

So a contrast with K removed reads K only where the cell holds records, and a control that pooled
every field would differ from `bko-direct` in which fields pool as well as in K.

## Decision Drivers

* **Each control must differ from its neighbour in one input**, so a contrast reads that input.
* **Must keep `bko-direct`'s draw and every existing run byte for byte**, since its runs of record
  are re-read under identity checks and digests check its draw.
* **Must state, per arm, which inputs it reads**, so an arm's reading is not credited to an input
  it never reads.
* **Must include a draw from K alone**, with no training value, so K's own floor is measured.
* **Must fix every model choice before an E19 holdout is scored.**

## Decision

1. **K.** K is the mapping table plus the pinned releases its rows name. Every arm below is a
   matched control of `bko-direct` on E6's split, quota, seed and classifiers.
2. **The arms.**

   | Arm | Home | What it draws | Single change |
   |---|---|---|---|
   | `bko-nomap` | `DIRECT_ARMS` | every unit own-first | (exists) |
   | `bko-donor` | `DrawSpec(presence=Presence.POOLED)` | `bko-nomap`, each mapping group copied whole by the pooled rule | from `bko-direct`: a share draw becomes a donor copy |
   | `bko-pool-random` | `DrawSpec(presence=Presence.POOLED, pool=Pool.RANDOM)` | `bko-nomap`, the pooled rule on as many fields as the mapping's groups hold, chosen at random | from `bko-donor`: K's field choice becomes a random one |
   | `bko-pool-all` | `DrawSpec(presence=Presence.POOLED, pool=Pool.ALL)` | `bko-nomap`, the pooled rule on every feature but `age` and `sex` | from `bko-donor`: K's subset becomes every field |
   | `row-resample` | `Method` | a whole row from one pooled donor in the cell's band | from `label-bootstrap`: the cell's weight |
   | `marginals` | `Method` | each column from its own pooled donor in the cell's band | from `row-resample`: rows become columns |
   | `eb-marginals` | `Method` | `marginals` at a fitted weight | from `marginals`: the weight |
   | `hb-marginals` | `Method` | each column from a multilevel model | a competitor, not an ablation |
   | `bko-kb` | `DrawSpec(presence=Presence.LOGIC, donors=False)` | codebook values consistent with a drawn pattern; no record | from `bko-logic`: the donors |

3. **The pooled rule, direct arms.** A unit's donor is a uniform cell record with probability
   n/(n+`GROUP_WEIGHT`), n the cell's records, else a uniform record of the whole label group.
   `Pool.RANDOM` draws its fields without replacement from the features less `age` and `sex`, by
   `np.random.default_rng([seed, POOL_RANDOM_STREAM])`, once per fill, and logs
   `pool-random holdout=%d fill=%s fields=%s`. Under `RANDOM` and `ALL` the draw reads the mapping
   only for the count of grouped fields. Every unit outside the pooled set is picked as
   `bko-nomap` picks it, and a pooled field is attributed `donor`.
4. **The pooled rule, methods.** As above with the label group restricted to the cell's band, so a
   whole row stays in its cell. One line per cell:
   `pool holdout=%d fill=%s key=%s own=%d banded=%d`, `key` naming the stratum as the direct arms'
   lines do.
5. **Categorisation** (`eb-marginals`, `hb-marginals`). A categorical column's codes; a numeric
   column's values when the label group holds at most 16 distinct, else 10 label-group decile
   bins. An ordered column merges its upper tail into one category so each holds at least 5
   label-group records; an unordered one folds codes under 5 into `other`.
6. **`eb-marginals`.** One `kappa` for all columns: the value of `KAPPA_GRID` (101 log-spaced
   values in [0.1, 10000]) that maximises the summed Dirichlet-multinomial log-likelihood of every
   column over the label's cells, each cell's prior the banded label group less that cell; ties go
   to the smaller. It logs `eb holdout=%d fill=%s kappa=%s`.
7. **`hb-marginals`.** Every feature but `age` and `sex`; `age` by the pooled rule, `sex` and the
   label from the cell. Per column a baseline-category multinomial logit, its reference the
   label-group modal category:
   `eta_v(a,s,y) = mu_v + A_v[a] + S_v[s] + Y_v[y] + AS_v[a,s] + AY_v[a,y] + SY_v[s,y] + E_v[a,s,y]`.
   Priors: `mu`, `A`, `S` and `Y` normal at scale 10; the two-way terms at `tau2`; `E` at `tau3`,
   0 at an emptied cell. `TAU_GRID = (0, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2)` for each of
   `tau2` and `tau3`; a scale of 0 drops its terms from the model, the two-way terms at `tau2` 0
   and every `E` at `tau3` 0, so evidence can choose main effects alone for a column. On a real
   Diabetes 130 fill a grid floored at 0.125 bound most columns at that floor, with evidence gaps to
   the extended grid's best pair of up to 35 nats. The fit is the posterior mode by L-BFGS-B from
   zeros with the analytic gradient (`gtol` 1e-8, `maxiter` 1000), then at most 20 Newton steps on
   the analytic Hessian, each halved at most 30 times while it lowers neither the objective nor
   the largest gradient entry, until every gradient entry is below 1e-8: L-BFGS-B alone stops on
   its relative-reduction test with entries up to about 0.07 left, enough to move a rounded
   evidence and the chosen pair. The pair with the largest Laplace log evidence, rounded to 6
   decimals, wins, ties going to the smaller `tau2`, then the smaller `tau3`. A value is its
   category for a coded column, and for a numeric one a uniform banded label-group record of that
   category, else a label-group record. Non-convergence, a gradient still at or above 1e-8 after
   the Newton steps, or a failed Cholesky raises `BaselineError`. One line per
   column:
   `hb holdout=%d fill=%s key=%s column=%s tau2=%s tau3=%s evidence=%.6f iterations=%d`. Its
   specification is fixed before any E19 holdout is scored, and a change to it is a new arm name.
8. **Refusals.** Every pooled method and `hb-marginals` refuse a request by label
   (`Strata.LABEL`) with `BaselineError`.
9. **`bko-kb`.** `kb_sample` takes no training frame. Presence per logic component is uniform over
   the assignments `bko.generate.logic.components` allows, from
   `np.random.default_rng([seed, cell_ordinal, record, component_ordinal])`; each group's coded
   fields are uniform over the codebook tuples showing the drawn pattern, from
   `np.random.default_rng([seed, cell_ordinal, record, 1000 + column_index])`; `age` is uniform
   over the band's integers inside `age_range`, the training split's observed range. A frame with
   a free field or a numeric clause is refused before any arm fits. It is K's structure alone,
   registered as a floor expected below `bko-logic`, and only on the supported-fields dataset,
   since E6's frame has free fields.
10. **Supported fields.** `supported_frame(frame, rows)` keeps `age`, `sex`, the label and every
    field of a `Codes` clause of a non-label row: 15 of E6's 34 columns.
    `Dataset.DIABETES_SUPPORTED` (`diabetes-supported`) is `diabetes` projected onto it, a variant
    under ADR [`research/dataset-variants`](dataset-variants.md).
11. **Shared constants.** A change to `GROUP_WEIGHT` moves the pooled arms with `bko-direct`,
    since all of them read the one constant.
12. **The runs.** E19's four families, each sliced in ten, `REFERENCE` `real`, candidate
    `bko-direct`, at E12's split, quota and seed:

    | Family | Dataset | Arms |
    |---|---|---|
    | `e19-thin` | `diabetes` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-donor`, `bko-pool-random`, `bko-pool-all`, `row-resample`, `marginals`, `eb-marginals`, `hb-marginals`, `label-bootstrap`, `prior-shift` |
    | `e19-empty` | `diabetes` | `none`, `real`, `bko-direct`, `bko-nomap`, `marginals`, `hb-marginals`, `label-bootstrap`, `prior-shift` |
    | `e19-supported-thin`, `e19-supported-empty` | `diabetes-supported` | `none`, `real`, `bko-direct`, `bko-nomap`, `bko-logic`, `bko-kb`, `hb-marginals`, `label-bootstrap` |

    `e19-empty` holds only the arms that differ in distribution at full removal: with no cell
    record, n/(n+w) is 0 for every w, so `bko-donor`, `bko-pool-random` and `bko-pool-all` draw
    `bko-nomap`'s distribution (E6's mapping groups are single fields), `row-resample` draws
    `label-bootstrap`'s, and `eb-marginals` draws `marginals`'.
13. **The reading** is RESEARCH section 11's
    [matched-control reading](../../RESEARCH.md#matched-control-reading).
14. **Declared inputs.** Each arm of E9, E12, E13, E15 and E19 reads the inputs marked `yes`, and
    a change to any other leaves its fill's bytes as they were. `cell` is the cell's own training
    rows; `label-group` the rest of its label's rows; `other-label` the other label's rows;
    `mapping` the `FieldMapping` rows; `hierarchy` the releases' `is_a` links; `codebook` the
    frame's codes per coded field; `age-range` the training split's observed age range, read as
    such and not through a copied record.

    | Arm | cell | label-group | other-label | mapping | hierarchy | codebook | age-range |
    |---|---|---|---|---|---|---|---|
    | `none` | no | no | no | no | no | no | no |
    | `class-weight` | no | no | no | no | no | no | no |
    | `prior-shift` | no | no | no | no | no | no | no |
    | `real` | yes | no | no | no | no | no | no |
    | `smote` | yes | yes | yes | no | no | no | no |
    | `adasyn` | yes | yes | yes | no | no | yes | no |
    | `ctgan` | yes | yes | yes | no | no | no | no |
    | `tvae` | yes | yes | yes | no | no | no | no |
    | `bayes-net` | yes | yes | yes | no | no | yes | no |
    | `label-bootstrap` | yes | yes | no | no | no | no | no |
    | `bko-direct` | yes | yes | no | yes | no | no | no |
    | `bko-nomap` | yes | yes | no | no | no | no | no |
    | `bko-logic` | yes | yes | no | yes | yes | no | no |
    | `bko-tree` | yes | yes | no | yes | yes | no | no |
    | `bko-donor` | yes | yes | no | yes | no | no | no |
    | `bko-pool-random` | yes | yes | no | yes | no | no | no |
    | `bko-pool-all` | yes | yes | no | no | no | no | no |
    | `row-resample` | yes | yes | no | no | no | no | no |
    | `marginals` | yes | yes | no | no | no | no | no |
    | `eb-marginals` | yes | yes | no | no | no | no | no |
    | `hb-marginals` | yes | yes | yes | no | no | no | no |
    | `bko-kb` | no | no | no | yes | yes | yes | yes |

    `none`, `class-weight` and `prior-shift` add no row: class weighting and the prior shift read
    label counts, never a feature value. `smote` takes its neighbours inside the cell and reads the
    other rows only through the z-scoring of measured fields; `adasyn` weighs each cell record by
    its neighbours among all rows. `bko-direct` records the `is_a` closure in `Draw.closed` and
    never reads it into a field, so a hierarchy change leaves its records equal. `bko-pool-random`
    reads the mapping for the count of grouped fields alone, and `bko-pool-all` not at all. The
    banded methods read the label group only inside the cell's band, and `eb-marginals`' `kappa`
    reads every band of it.
15. **Audit.**

    | Arm | Presence weights | Donor pool | Field grouping | Fitted model |
    |---|---|---|---|---|
    | `none`, `class-weight` | class weights n/(k n_c) from training label counts (`class-weight`) | none | none | none |
    | `prior-shift` | none | none | none | a logit offset to each requested subgroup's prevalence after the gap fill, from out-of-fold training predictions |
    | `real` | none | the cell's removed training rows | whole rows | none |
    | `smote`, `adasyn` | none | the cell's records, interpolated between neighbours | whole rows | k nearest neighbours, k = min(5, n-1), on the z-scored training split |
    | `ctgan`, `tvae`, `bayes-net` | none | none: rows sampled from the model, kept by stratum | whole rows | the named model on the whole training split |
    | `label-bootstrap` | none | the label's records in the cell's band, uniform, sex set to the cell's | whole rows | none |
    | `bko-direct` | each mapped concept at (n p_cell + 20 p_group)/(n + 20), training counts | each unit: cell records showing its pattern, else the label group's | mapping groups whole; `age` and each free field alone, own-first | none |
    | `bko-nomap` | none | each field: the cell's records, else the label group's; `age` else the banded label group | every field alone | none |
    | `bko-logic` | uniform over each component's consistent assignments | as `bko-direct`, closure read into fields | as `bko-direct` | none |
    | `bko-tree` | tree fields at ancestor-shrunk shares at `GROUP_WEIGHT`, training counts; others as `bko-direct` | a tree field: a donor of that column alone showing the leaf | tree fields alone; others as `bko-direct` | none |
    | `bko-donor` | none | each mapping group: a cell record with probability n/(n + 20), else a label-group record; others as `bko-nomap` | mapping groups whole | none |
    | `bko-pool-random` | none | the pooled rule on its drawn fields; others as `bko-nomap` | as many fields as the mapping's groups hold, drawn once per fill | none |
    | `bko-pool-all` | none | the pooled rule on every feature but `age` and `sex`; `age` as `bko-nomap` | every field alone | none |
    | `row-resample` | none | a cell record with probability n/(n + 20), else a banded label-group record | whole rows | none |
    | `marginals` | none | per column, the `row-resample` pool | every column alone | none |
    | `eb-marginals` | none | per column, the pool at n/(n + `kappa`) | every column alone | `kappa` by Dirichlet-multinomial likelihood over `KAPPA_GRID` |
    | `hb-marginals` | none | a numeric category's banded label-group record; `age` by the pooled rule | every column alone | per column the multinomial logit of Decision 7 |
    | `bko-kb` | uniform over each component's consistent assignments | none | mapping groups, uniform over codebook tuples | none |

    The weights read from training counts are `class-weight`'s, `bko-direct`'s and `bko-tree`'s
    shares, and the pooled rule's n.

## Rationale

The study asks whether K's choice of which fields to pool adds recall. `bko-donor` alone cannot
answer it: it shows only that a share draw and a donor copy agree. `bko-pool-random` keeps
everything `bko-donor` does and replaces K's field choice with a random choice of as many fields,
so `bko-direct` minus `bko-pool-random` is the contrast that carries K's choice. `bko-pool-all`
bounds it from the other side.

The pooled rule's n/(n+w) is the adjustment-cell and shrinkage weight
([`little-1986`](../../literature.md#missing-data-and-imputation),
[`efron-1975`](../../literature.md#partial-pooling),
[`fay-1979`](../../literature.md#partial-pooling)): a cell's own records count in proportion to
how many it holds. The direct arms pool against the whole label group because that is the
population `bko-direct`'s shares read; a banded pool would be a second difference. The methods pool
inside the band because a whole copied row must stay in its cell.

`eb-marginals` fits one weight for all columns because eight label cells give noisy per-column
fits, and a one-category column would pick the grid's smallest value. `hb-marginals` is the
data-only model of an emptied cell a reviewer would name: a multilevel model with every two-way
interaction reads the band-by-label and sex-by-label signal a main-effects model discards
([`gelman-2006`](../../literature.md#partial-pooling),
[`bishop-1975`](../../literature.md#partial-pooling)). A posterior mode with a fixed start, an
analytic gradient, Newton steps to a gradient tolerance, a fixed grid and evidence rounded to 6 decimals gives the same fit on every
machine.

`bko-kb` reads no training record, so it measures what K's structure alone gives. Uniform presence
adds no rate the releases do not state (ADR
[`research/logic-only-ablation`](logic-only-ablation.md), Rationale); a frame with a free field
would need data for that field, so the arm runs on the supported-fields projection. Seeding per
component and per column keeps a change to one component's rows from shifting the others through
one stream.

The declared-inputs table states each arm's reach, so a contrast is read against the inputs that
differ between its two arms and no others.

## Consequences

**Gained**: a contrast that isolates K's field choice from pooling; data-only competitors for an
emptied cell, from a fixed weight to a fitted multilevel model; a floor from K alone; one table
saying what every control of E9, E12, E13, E15 and E19 reads.
**Accepted**: four new direct arms and four new methods, each a name a run carries; at full removal
the pooled direct arms coincide with `bko-nomap`, so no K contrast is read there; `bko-kb` runs only
on a projection of 15 of E6's 34 columns, so its readings do not carry to E6's full frame; the
pooled methods refuse a label-keyed request.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A pooled rule on every unit | On E6's frame it equals `marginals` | 2026-11-26 |
| The banded label group as the direct arms' pool | A second difference from `bko-direct`, whose shares read the whole label group | 2026-11-26 |
| Duplicates of `label-bootstrap` and `prior-shift` | E19 registers both as they are | 2026-11-26 |
| A probabilistic-programming dependency for `hb-marginals` | A posterior mode by L-BFGS-B needs only scipy, which the lock holds; see Rationale | 2026-11-26 |
| A main-effects-only model for every column | Discards the label-by-subgroup signal an emptied cell is predicted from; the grid's 0 lets evidence drop the terms per column instead | 2026-11-26 |
| A grid floored at 0.125 | On a real Diabetes 130 fill the floor bound most columns, and the extended grid's best pair beat it by up to 35 nats of evidence | 2026-11-26 |
| One-vs-rest fits | Not a likelihood model of a column's categories | 2026-11-26 |
| One `kappa` per column | Eight label cells give noisy fits, and a one-category column picks the grid's smallest value | 2026-11-26 |
| An `Experiment.features` field | Placeholder values in place of a dataset name; ADR `research/dataset-variants` holds the projection | 2026-11-26 |
| An external prevalence table for `bko-kb` | A new knowledge source beside K, so the arm would not read K alone | 2026-11-26 |
| A causal reading of subgroup reassignment of positive donors with the other fields fixed | The matched design registers no arm for it, and fixing the other fields does not make the reassignment an intervention | 2026-11-26 |

## Enforcement

- Owner: `src/bko/evaluate/runs_direct.py`, with the draw spec in
  `src/bko/generate/draw_spec.py` and the methods in `src/bko/baselines/methods.py`
- Pinned by: nothing -- convention only

## Reversal Conditions

**E19's primary contrast reads `equivalent` or `below`**: `bko-direct` minus `bko-pool-random` on
removed-cell recall at the thin fraction on `diabetes`, read in
[EVIDENCE.md](../../EVIDENCE.md#e19-matched-knowledge-controls-on-the-diabetes-130-removal) under
the [matched-control reading](../../RESEARCH.md#matched-control-reading).

## Related

- [The no-mapping ablation draws bko-direct's records with an empty mapping table](no-mapping-ablation.md)
- [The logic-only ablation draws each component of mapped concepts uniformly from the assignments the ontology allows](logic-only-ablation.md)
- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [A record's explanation is the stored or re-derivable facts that determine it](record-explainability.md)
