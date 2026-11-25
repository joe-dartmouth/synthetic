---
status: accepted
topic: research
created: 2026-11-05
updated: 2026-11-25
---
# Each stage keeps a data contradiction and reports it; only an ablation arm forces the ontology onto the data

## Context

A mapped dataset can disagree with the ontology it is mapped to. A record can show a narrow
concept without its broader one, because the mapping table says when a record shows a concept and
nothing more (ADR `research/ontology-mapping-design`). On Diabetes 130, most records coded for
diabetes with renal manifestations (`MONDO:0005016`) do not show chronic kidney disease
(`MONDO:0005300`), and almost no record coded for chronic bronchitis (`MONDO:0005607`) shows
chronic obstructive pulmonary disease (`MONDO:0005002`); the Rationale gives the counts. Each
stage then meets a contradiction: fusion holds an `is_a` S-node of probability 1 beside a dataset
prior that contradicts it, gap detection reads a held child, a share can leave its unit interval
or its sample, and the direct sampler draws a child without its parent.

## Decision Drivers

* **Must name each kind of contradiction once**, so a stage's handling of it is stated against a
  fixed list.
* **Must keep every stage's output a valid fused BKB**, so the fused BKB a stage reads is one
  `bko.bkb.fusion.check_fused` accepts.
* **Must keep every recorded run reproducible**, so `bko-direct`'s draw and every non-empty
  cell's share are the ones its recorded runs read.
* **Must measure each kind on the data**, every cell examined, not on a sample.

## Decision

1. **Kinds.** Five, each a property of a mapped frame:
   - `narrow-without-broad`: for each `is_a` pair among the mapped concepts, the records that show
     the narrow concept and not the broad one.
   - `child-share-above-parent`: a concept whose share in a cell exceeds its `is_a` parent's.
   - `empty-cell`: a cell with no record, whose share is its label group's `p_g`.
   - `all-show`: a concept every record of a cell shows, of the concepts that do not follow the
     label (`bko.gaps.shares.label_only`), which every record of their label shows by their rows.
     A cell of few records shows a concept in every record far more readily than a large one, so
     the kind fires most in the smallest cells; it is counted there, not guarded.
   - `code-overlap`: two concepts whose codes on one field intersect, unless `nested` resolves
     the pair, as Decision 7 draws it. A concept's codes on a field are taken per row: the codes
     every one of the row's `Codes` clauses on that field admits, united over the concept's rows.
     A row's clauses on other fields, and its clauses that are not `Codes`, are ignored for this
     kind; UCI's `HP:0012250` row reads a threshold beside its codes, and only its codes count.
2. **Bases.** The `contradictions` command reads one of two bases.
   - **The census.** With no run, it examines, for each frame, every cell of the frame's
     training split at holdout seed 0 with no removal, and lists each finding. The frames are
     `CENSUS`'s, UCI Heart Disease, Synthea and Diabetes 130, unless a repeatable `--dataset`
     names others: one of `CENSUS`'s or a catalogued dataset (`bko.ontology.catalog`). The
     default set stays `CENSUS`, so a recorded census keeps its bytes and the bare command fetches
     no catalogued archive. `--dataset` and `--rows` exclude each other.
   - **The added rows.** `contradictions <run> --rows <store>...` reads each holdout and fill of
     a run's `rows` stores, gap and replica fills alike, and counts in the fill's added rows A:
     `narrow-without-broad`, the rows of A showing a narrow concept without its broad one,
     summed over the `is_a` pairs; `child-share-above-parent` and `all-show`, the survey's
     findings over the cells A adds rows to; `empty-cell`, the cells of the fill's request that
     hold no row of A; `code-overlap`, the rows of A whose code lies in an unresolved overlap;
     and the three coherence rules of `bko.evaluate.perturb.rule_breaks`, `drug-without-med`,
     `change-without-med` and `dose-without-change`, each as the rows of A breaking it, blank
     where the frame lacks their columns. The request is the one the fill answered, rebuilt from
     the holdout's training split: the gap request for a gap fill, the replica request for a
     replica fill. `empty-cell` is blank for a fill that answers no request: `none`, `real`,
     whose rows are the removed rows, the weighting arms, which add no rows, and `prior-shift`.
     Every other fill answers one, so a fill of it with no rows counts every cell of its request
     as empty and 0 on every other kind, its rules still blank off the Diabetes columns.
3. **Valid fused BKB.** One for which `bko.bkb.fusion.check_fused` passes. It requires:
   - every input S-node is present, with its source tail and an unchanged probability;
   - every fused S-node other than a source prior maps to an input; a dataset conditional (ADR
     `research/dataset-fragment-encoding`) maps to its input like any non-prior S-node;
   - each source component's priors sum to 1 (under the default `FRAGMENTS` normalization, to
     at most 1, and to 1 where every fragment holds an S-node on the head component), a source
     component a fused S-node's tail names with no prior on it summing to 0;
   - every prior's state names only sources;
   - every probability is in [0, 1], which `BKB.add_snode` enforces as each S-node is added, so
     `check_fused` does not repeat it.

   Mutual exclusion is not required (ADR `research/bko-fragment-encoding`).
4. **Fusion** keeps both S-nodes: the `is_a` S-node of probability 1 and the dataset prior that
   contradicts it. Neither is dropped or rescaled. Where a dataset conditional is fused, fusion
   keeps the ontology's `is_a` S-node and the conditional both, at their own probabilities.
5. **Gap detection**: a held child grounds its parent, as the `is_a` S-node says, whatever the
   dataset shows of the parent. It reads no dataset conditional: its fragments come from
   `dataset_fragment`, which holds none.
6. **Shares** are finite and in [0, 1]. An empty cell has `n_c` of 0, so its share is exactly
   `p_g` (ADR `research/bkb-direct-generation`, Decision 1): `shares` returns `p_g` itself there,
   since multiplying it by `GROUP_WEIGHT` and dividing it back can move it by one ulp; a
   non-empty cell's share is the formula.
7. **`bko-direct`**, and every direct arm that shares its concept draw, never forces an `is_a`
   ancestor into a field. A `nested` field pair keeps the outer concept whenever the inner is
   drawn. A draw no record matches falls back to the label group's records, then raises
   `SamplingError`. A label group with no record raises `GapError`, once. `bko-hier-0` and
   `bko-hier-w` draw a pair's broad concept given its narrow one at the data's conditional shrunk
   toward the `is_a` S-node (ADR `research/bkb-direct-generation`, Decision 12), forcing none.
   `bko-hier-closed` forces each pair's broad concept whenever its narrow one is drawn, and
   `bko-logic` reads the closure into fields (ADR `research/logic-only-ablation`); both are
   ablation arms only.
8. **Reported, not corrected.** The `contradictions` command reports every kind; no stage
   rewrites the data, and no draw but an ablation arm's removes one (Decision 7).

## Rationale

Forcing an ancestor into its fields would assert values no share supports: the ontology fragments
carry no prior, and the mapping table says when a record shows a concept, never what it implies
(ADR `research/bkb-direct-generation`, Decision 3). It would also rewrite most of the records
that show such a child. On the census's basis, the Diabetes 130 training split at holdout seed 0,
`MONDO:0005016` shows without `MONDO:0005300` in 757 of its 838 records, and `MONDO:0005607`
without `MONDO:0005002` in 2,092 of its 2,095; the `diabetes` frame's `census pair` lines in the
D2 census run's log derive them. The data contradicts the ontology far more often
than it agrees, and a forced ancestor would replace the data's reading with the ontology's.

Keeping both S-nodes in fusion is what a source-weighted fusion does: each source stays a
source (ADR `research/bkb-layer-source`), and a BKB need not be mutually exclusive to be valid
(ADR `research/bko-fragment-encoding`, Consequences). A check on the fused BKB is therefore a
check on what fusion promises, that it adds sources and priors and changes no input probability,
and not on whether the sources agree.

A nested pair is the one contradiction the sampler resolves, because the mapping table fixes it:
two rows on one field whose codes lie strictly inside each other cannot show the inner without
the outer. Every other kind is a property of the data, so the command reports it and the stages
read the data as it is.

The census runs from `bko.evaluate` because it loads a dataset through `bko.evaluate.data.load`
(records, digest, agreement) and logs `versions` through `bko.evaluate.provenance`, and `bko.gaps`
sits below `bko.evaluate`; the frames it reads live in `bko.ontology.frames`, which `bko.gaps` may
import.

## Consequences

**Gained**: each stage's handling of each kind is one statement, checkable by `check_fused` and
the `contradictions` command, and an empty cell's share is `p_g` itself.
**Accepted**: a fused BKB can hold an `is_a` S-node and a dataset prior that disagree, so
inference over it reads both; `bko-direct` can draw a child without its parent when the mapping
table does not nest them, a pattern the data shows.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Forcing ancestors into `bko-direct`'s fields | Asserts values no share supports, and the data shows the child without the ancestor far more often than with it (Rationale) | 2026-11-05 |
| Dropping the dataset prior a contradiction touches | Changes an input probability, which `check_fused` refuses | 2026-11-05 |
| Requiring mutual exclusion of a fused BKB | A class with two scoped children is already non-mutex (ADR `research/bko-fragment-encoding`) | 2026-11-05 |
| The census in `bko.gaps` | It would move the evaluate loader, `Context` and `log_versions` below `bko.evaluate` for a command nothing outside evaluate runs | 2026-11-09 |
| The training split as the only basis | It cannot show whether an arm's added rows contradict the ontology or break the Diabetes coherence rules, the measure the generation units target | 2026-11-24 |
| One share of rows breaking any coherence rule, as `incoherent` reads it | It hides which rule a fill breaks; each rule is counted on its own | 2026-11-24 |
| Appending each catalogued dataset to `CENSUS` | It would change the bytes of a recorded census and make the bare command fetch every catalogued archive; `--dataset` names them instead | 2026-11-25 |
| Forcing ancestors in every direct arm | It would replace the data's reading with the ontology's in every fill; `bko-hier-closed` forces them as one ablation arm beside the shrunk conditional, so the forcing is measured against `bko-direct` | 2026-11-25 |

## Enforcement

- Owner: `src/bko/bkb/fusion.py` (`check_fused`), `src/bko/generate/direct.py`,
  `src/bko/generate/donors.py` (the label-group fallback in `_pick`),
  `src/bko/gaps/shares.py`, `src/bko/gaps/contradictions.py`, `src/bko/evaluate/contradictions.py`,
  `src/bko/evaluate/added_contradictions.py`, `src/bko/ontology/frames.py`
- Pinned by: `tests/test_contradictions_census.py`, `tests/test_contradictions_dataset.py`,
  `tests/test_contradictions_added.py`,
  `tests/test_contradictions_added_diabetes.py`, `tests/test_contradictions_kinds.py`,
  `tests/test_contradictions_fusion.py`, `tests/test_contradictions_gaps.py`,
  `tests/test_contradictions_shares.py`, `tests/test_contradictions_direct.py`,
  `tests/test_ontology_frames.py`, `tests/test_gaps_conditionals.py`; for the `bko-hier` arms
  `tests/test_generate_hierarchy.py`, `tests/test_evaluate_hierarchy.py` and
  `tests/test_generate_hierarchy_pinned.py`; the digest pins
  in `tests/test_generate_direct.py` and `tests/test_evaluate_direct.py` hold `bko-direct`'s draw
  and every non-empty cell's share

## Reversal Conditions

**`bko-direct` and the stages other than the ablations force ancestors when the mapping table
gains a row that implies its parent's values**;
the check is a search of `bko.ontology.mapping.FIELD_MAPPINGS` and the Diabetes field rows for a
row that reads a concept's parent. **The valid-BKB check requires mutual exclusion if ADR
`research/bko-fragment-encoding` comes to encode a state for absence.** **The census moves to
`bko.gaps` when a module outside `bko.evaluate` other than `bko.app.steps` imports
`bko.evaluate.data`**; the check is
`grep -rn "^from bko.evaluate.data import" src/bko | grep -v "^src/bko/evaluate/"`, which lists
only `src/bko/app/steps.py` until then.

## Related

- [Ontology classes become single-state BKB components joined by is_a and exactMatch S-nodes](bko-fragment-encoding.md)
- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [The BKB layer is pybkb's core, ported under GPL-3.0 and checked against pybkb's own results](bkb-layer-source.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
