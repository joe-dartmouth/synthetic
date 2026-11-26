---
id: paper-controls
title: Isolate the contribution of external knowledge
status: ready
depends_on: --
blocks: ontology-auto-mapping, paper-control-results
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: research/matched-knowledge-controls, research/dataset-variants
---

# Isolate the contribution of external knowledge

The set's graph is [direct-index.md](direct-index.md).

## Why this unit exists

E9, E12, E13 and E15 supply related controls. A change in donor selection or field dependence can
also change predictive utility.

## Scope

**In.**

- K, the knowledge the study isolates, is the dataset's mapping table (its `FieldMapping` rows,
  `bko.evaluate.data.field_rows`) plus the pinned ontology releases those rows name. Every arm
  below is a matched control of `bko-direct` on E6's split, quota, seed and classifiers.
- An audit of the draws of E9's, E12's, E13's and E15's arms against the matched design, held as
  two tables of a new ADR under `docs/adr/research/`: a declared-inputs table (one row per arm of
  E9, E12, E13, E15 and E19; columns `cell`, `label-group`, `other-label`, `mapping`,
  `hierarchy`, `codebook`, `age-range`) and an audit table (presence weights, donor pool, field
  grouping and fitted model per arm), which records the weights that already come from training
  counts. The unit creates `tests/_arm_inputs.py`, whose `ARM_INPUTS` maps each arm to the
  frozenset of the seven declared-input names it reads, and `tests/test_arm_inputs.py`. The ADR
  points at ADR [`research/no-mapping-ablation`](../docs/adr/research/no-mapping-ablation.md),
  Consequences, for why `bko-direct` and `bko-nomap` draw one distribution at full removal on
  E6's frame, and states that `bko-direct` pools at `GROUP_WEIGHT` 20 only on the 12 mapped
  fields while `age` and the 19 free fields are copied own-first
  (`src/bko/generate/direct.py:365-388`).
- Three direct arms, each `bko-nomap`'s record with a set of fields drawn by the pooled rule (a
  unit's donor is a uniform cell record with probability n/(n+`GROUP_WEIGHT`), else a uniform
  record of the whole label group), each pooled field attributed `donor`:
  - `bko-donor`, `DrawSpec(presence=Presence.POOLED)`: each mapping group copied whole by the
    pooled rule; from `bko-direct`, a share draw becomes a donor copy.
  - `bko-pool-random`, `DrawSpec(presence=Presence.POOLED, pool=Pool.RANDOM)`: the pooled rule on
    as many fields as the mapping's groups hold, drawn without replacement from the features less
    `age` and `sex` once per fill, logged as `pool-random holdout=%d fill=%s fields=%s`; from
    `bko-donor`, K's field choice becomes a random one.
  - `bko-pool-all`, `DrawSpec(presence=Presence.POOLED, pool=Pool.ALL)`: the pooled rule on every
    feature but `age` and `sex`.
- Four retained-data methods, each refusing a request by label with `BaselineError`:
  - `row-resample`: a whole row from one pooled donor of the cell's band;
  - `marginals`: each column from its own pooled donor of the cell's band;
  - `eb-marginals`: `marginals` at one empirical-Bayes weight `kappa` for all columns, the
    Dirichlet-multinomial maximum over a fixed log-spaced grid, logged as
    `eb holdout=%d fill=%s kappa=%s`;
  - `hb-marginals`: a hierarchical partial-pooling competitor, per column a baseline-category
    multinomial logit with main effects, every two-way interaction and a cell effect, its scales
    chosen on a fixed grid by Laplace evidence, logged per column as
    `hb holdout=%d fill=%s key=%s column=%s tau2=%s tau3=%s evidence=%.6f iterations=%d`, `key`
    naming the stratum as the direct arms' lines do. Its
    specification is fixed before any E19 holdout is scored; a change is a new arm name.
- `bko-kb`, `DrawSpec(presence=Presence.LOGIC, donors=False)`: the ontology-only draw. It takes
  no training frame: presence per logic component is uniform over what K allows, coded fields
  uniform over the codebook tuples showing the drawn pattern, and `age` uniform inside its band
  and the training split's age range. A frame with a free field or a numeric clause is refused
  before any arm fits. It is registered only in `e19-supported-thin` and `e19-supported-empty`,
  since E6's frame has free fields, which it refuses; `bko-logic` sits beside it there, and
  `bko-kb` is a floor expected below it.
- The fields K cannot generate marked by a projected dataset, `diabetes-supported`
  (`Dataset.DIABETES_SUPPORTED`): `diabetes` on `age`, `sex`, the label and every field of a
  `Codes` clause of a non-label row, 15 of E6's 34 columns, held in `VARIANTS` over its base, so
  a mapping change or a projection is a new dataset name (a new ADR under `docs/adr/research/`).
- E19's four families, each sliced in ten, `REFERENCE` `real`, candidate `bko-direct`, at E12's
  split, quota and seed: `e19-thin` and `e19-empty` on `diabetes`, `e19-supported-thin` and
  `e19-supported-empty` on `diabetes-supported`. `e19-empty` holds only the arms that differ in
  distribution at full removal.
- The primary contrast: `bko-direct` minus `bko-pool-random` on removed-cell recall at the thin
  fraction on `diabetes`, claimed only when the regression's and the boosting model's 0.95
  intervals both lie above 0; `equivalent` uses the margin of the
  [control reading rule](../docs/RESEARCH.md#control-reading-rule). Every other E19 contrast is
  secondary and descriptive, a secondary claim Holm-adjusted within its fraction. No K contrast
  is read at full removal, where `bko-direct` and `bko-nomap` draw one distribution (ADR
  [`research/no-mapping-ablation`](../docs/adr/research/no-mapping-ablation.md), Consequences).
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry E19's Configuration with status
  `not yet run`; `docs/RESEARCH.md` section 11's ablation list, register row and matched-control
  reading; the cited works registered in [`docs/literature.md`](../docs/literature.md).

**Out.**

- A pooled rule on every unit: on E6's frame it equals `marginals`.
- The banded label group as the direct arms' pool: a second difference from `bko-direct`.
- Duplicates of `label-bootstrap` and `prior-shift`, which E19 reads as registered.
- A probabilistic-programming dependency, a main-effects-only model, one-vs-rest fits and one
  `kappa` per column.
- An `Experiment.features` field in place of a dataset name.
- An external prevalence table for `bko-kb`.
- Subgroup reassignment of positive donors with the other fields fixed, as a diagnostic: the
  matched design registers no arm for it, and a causal reading of it is ruled out.
- Any edit to an existing `RUNS` entry or to `bko-direct`'s draw: `DRAW_DIGEST`
  (`tests/test_generate_direct.py`) and `ADDED_DIGEST` (`tests/test_evaluate_direct.py`) stay
  byte-identical.
- Running E19: unit [`paper-control-results`](paper-control-results.md).

## Acceptance

- Register a development run with an information-source table: `RUNS` holds E19's four families
  (`tests/test_evaluate_control_runs.py`, and `tests/test_named_runs_slices.py`, whose `KEYS`
  gains the new names immediately before `"p1-id"` and whose slice parametrize list gains `e19`
  and `e19-supported`), and the ADR's
  declared-inputs table mirrors `tests/_arm_inputs.py`
  (`tests/test_arm_inputs.py::test_the_adr_table_mirrors_arm_inputs`).
- Include an ontology-only draw using no empirical donors or fitted marginals:
  `tests/test_generate_knowledge_draw.py::test_the_draw_takes_no_training_frame` and
  `test_two_training_frames_of_one_age_range_give_one_fill`.
- Mark fields K cannot generate and compare on common supported fields:
  `tests/test_evaluate_variants.py::test_the_supported_frame_keeps_the_axes_the_label_and_the_mapped_fields`
  and `test_the_supported_split_is_the_base_split`, and
  `tests/test_generate_knowledge_draw.py::test_a_frame_with_a_free_field_is_refused`.
- Test that each arm reads only its declared inputs and that swapping K changes only the
  intended draw component:
  `tests/test_arm_inputs.py::test_an_arm_changes_only_when_a_declared_input_changes` and
  `test_a_hierarchy_swap_leaves_bko_direct_records_equal`, and
  `tests/test_generate_knowledge_draw.py::test_swapping_one_components_rows_leaves_other_components_equal`.
- Each E19 run shares E12's split, quota and seed, and `e19-empty` holds no arm that coincides
  there: `tests/test_evaluate_control_runs.py::test_the_matched_runs_share_split_quota_and_seed`
  and `test_the_empty_run_holds_no_arm_that_coincides_there`.
- The pooled methods and the multilevel baseline give their hand-computed fills:
  `tests/test_baselines_pooling.py` and `tests/test_baselines_multilevel.py` pass, the latter
  with `test_the_fit_is_deterministic` and `test_a_fit_that_does_not_converge_is_refused`.
- The pooled direct arms score and trace end to end:
  `tests/test_generate_pooled.py` and `tests/test_evaluate_pooled_arms.py` pass, with the digest
  literals of `tests/test_generate_direct.py` and `tests/test_evaluate_direct.py` untouched.
- One real slice of each dataset fills for free, as evidence recorded in the PR body rather than
  a gate check: `python -m bko.evaluate score e19-thina` and `score e19-supported-thina` into
  `scratchpad/` exit 0, logging one `arm` line per arm and holdout, a `pool-random holdout=` line
  per holdout in the first, and no `refused` or `contract` line.
- `make check` passes.
