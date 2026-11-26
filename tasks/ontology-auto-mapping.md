---
id: ontology-auto-mapping
title: Build mapping rows automatically from a codebook and a pinned release, with no paid model
status: blocked
depends_on: paper-controls
blocks: ontology-auto-mapping-results, paper-scarcity
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/automatic-mapping
---

# Build mapping rows automatically from a codebook and a pinned release, with no paid model

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The mapping table is hand-written. NVIDIA auto-ontology does not replace it. Read at its source
on 2026-11-18, it makes one self-named term per table, its term prompt rules out `is_a`
relations, and it links to no standard ontology.

## Scope

**In.**

- Two builders of `FieldMapping` rows (`bko.ontology.schema`) from a frame's codebook and a
  pinned release, in a new `bko.ontology.automatic` that imports nothing from `bko.generate` or
  `bko.evaluate`. Neither calls a paid API. An automatic mapping is the builder's rows alone,
  each row's `source` `research/automatic-mapping`.
  - `auto-lexical` uses no model. A coded value with a live cross-reference maps through it to
    its code system. A flag or level maps to the term whose name or EXACT synonym best matches
    its codebook meaning (the field description when the meaning is generic; negative states
    skipped) by token-set Jaccard at or above `THRESHOLD` 4/5, after NFKC, casefolding and
    non-alphanumerics to spaces; ties go to the sorted order of the dataset's release prefixes,
    a name over a synonym, then the smallest curie. Rows are sorted by (field, code, concept).
    Numeric columns stay unmapped. `Term.synonyms` carries each term's (scope, text) synonyms.
  - `auto-local` judges the same candidates with the pinned on-device model
    `gemma-4-26b-a4b-q8_0` through `bko.generate.local`: `local_send` takes a `build` payload
    function whose default keeps every existing request's bytes, and `bko.generate.judge` sends
    temperature 0, seed 0 and a strict `{"concept": enum}` schema over the top ten lexical
    candidates and `none`, one call per candidate state. Its rows are a pure function of a
    committed verdict table, `VERDICTS`, keyed `"<field>=<code>"`.
- The leakage rule: no record-level statistic; a frame's codebook may list the cohort's distinct
  values, as the hand table's generator reads them. `diabetes-codes`' codebook is read through
  `bko.datasets.diabetes.load_diabetes` and `bko.ontology.code_frames`; its cross-reference
  route reproduces the hand rows, and the codes MONDO links only by `relatedTo` or an unreviewed
  source xref (ADR
  [`research/diabetes-diagnosis-codes`](../docs/adr/research/diabetes-diagnosis-codes.md)) are
  the candidate set judged on top, reported as an acceptance count.
- Entry points `python -m bko.ontology.automatic lexical <dataset>` and
  `python -m bko.generate.judge <dataset> --model gemma-4-26b-a4b-q8_0`, each written through
  `scratchpad/` then `mv`, into four committed modules under `src/bko/ontology/`:
  `auto_lexical_diabetes_rows.py`, `auto_lexical_codes_rows.py`, `auto_local_diabetes_rows.py`
  and `auto_local_codes_rows.py`, each split in two by field if over 500 lines. Each module's
  rows sha256 is stated in the ADR, and its rows pass the structural check
  `bko.ontology.check.check` against the pinned releases.
- Precision and recall over (field, code, concept) atoms against the hand table, exact curie,
  each `None` on an empty denominator.
- The permuted control: the hand rows with their concept labels permuted at a seed
  (`permuted(rows, seed=...)`), every field state kept.
- Six dataset variants in `VARIANTS`, `diabetes-auto-lexical`, `diabetes-auto-local`,
  `diabetes-permuted`, `diabetes-codes-auto-lexical`, `diabetes-codes-auto-local` and
  `diabetes-codes-permuted`, each logging `mapping dataset=%s rows=%d sha256=%s`; the ADR on
  dataset variants that unit `paper-controls` writes, revised in place for
  row-swap variants. `run_direct`'s pre-run check
  also builds the hierarchy for a conditional spec and the column trees for a tree spec,
  refusing with `EvaluationError` before any arm fits.
- E22's `RUNS` entries, eight families each thin and empty, sliced, `REFERENCE` `bko-direct`,
  candidate `bko-direct`, comparators `bko-nomap` and `none`: on E6's frame, `e22-hand-*`,
  `e22-auto-lexical-*`, `e22-auto-local-*` and `e22-permuted-*` with `none`, `real`,
  `bko-direct`, `bko-nomap` and `bko-hier-w`; on `diabetes-codes`, `e22-codes-hand-*`,
  `e22-codes-auto-lexical-*`, `e22-codes-auto-local-*` and `e22-codes-permuted-*` with `none`,
  `real`, `bko-direct`, `bko-nomap` and `bko-tree` (ADR
  [`research/ancestor-shrunk-shares`](../docs/adr/research/ancestor-shrunk-shares.md)). The
  permuted control is scored on `bko-hier-w` (E6's frame) and `bko-tree` (`diabetes-codes`). A
  variant whose rows cannot build a hierarchy or trees is registered without `bko-hier-w` or
  `bko-tree`, named with its reason in the ADR's Consequences and E22's Configuration.
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry E22's Configuration with status
  `not yet run`, with the margin for automatic minus hand.
- A new ADR under `docs/adr/research/` that records the route, the NVIDIA finding and the leakage
  rule; one sentence in ADR
  [`research/ontology-mapping-design`](../docs/adr/research/ontology-mapping-design.md) admitting
  the ADR id to the `source` vocabulary; BERTMap and the NVIDIA source in
  [`docs/literature.md`](../docs/literature.md).

**Out.**

- Paid models.
- A model-proposed cutoff row for a numeric column.
- A mapping built from records, and an SSSOM table.
- Running NVIDIA auto-ontology.
- Confirmatory datasets.
- Running E22: unit [`ontology-auto-mapping-results`](ontology-auto-mapping-results.md).

## Acceptance

- On UCI Heart Disease, `auto-lexical`'s rows are byte-equal across two runs:
  `uv run --no-sync python -m bko.ontology.automatic lexical uci > scratchpad/auto_uci_<n>.py </dev/null`
  twice, then `cmp scratchpad/auto_uci_1.py scratchpad/auto_uci_2.py` is silent, and
  `tests/test_ontology_automatic.py::test_render_is_byte_equal_under_two_hash_seeds` passes.
- A test computes each builder's precision and recall against the hand table on a fixture
  codebook: `tests/test_ontology_automatic.py::test_precision_and_recall_on_the_fixture_codebook`
  (3 rows built, 2 matching, 4 hand rows: `Fraction(2, 3)` and `Fraction(1, 2)`) and
  `test_precision_and_recall_are_none_on_empty_denominators`.
- The builder takes no records: `tests/test_ontology_automatic.py::test_the_builder_takes_no_records`.
- `auto-local`'s rows are a pure function of its verdicts, and existing local payloads keep their
  bytes: `tests/test_generate_judge.py::test_rows_are_a_pure_function_of_the_verdicts` and
  `test_the_judge_sends_temperature_0_and_a_seed`, and
  `test_existing_local_payloads_keep_their_bytes`.
- The permuted rows keep every field state and move concepts:
  `tests/test_ontology_automatic.py::test_permuted_rows_keep_every_field_state_and_move_concepts`,
  and a permuted variant changes a hierarchy-reading arm's fill:
  `tests/test_evaluate_variants.py::test_a_permuted_variant_changes_a_hierarchy_reading_arms_fill`.
- Each committed module equals a rebuild, its rows sha256 equals its stated constant, and
  `bko.ontology.check.check` accepts it:
  `uv run --no-sync pytest tests/test_ontology_auto_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null`
  passes.
- Each variant's rows build the hierarchy or trees its family's arms read, or the family is
  registered without that arm:
  `uv run --no-sync pytest tests/test_evaluate_auto_draws_pinned.py --releases -q -p no:cacheprovider --no-cov </dev/null`
  passes.
- `RUNS` holds E22: `tests/test_evaluate_auto_runs.py` and `tests/test_named_runs_slices.py`
  pass.
- One real slice scores for free: `python -m bko.evaluate score e22-auto-lexical-thina` into
  `scratchpad/` exits 0 with a `mapping dataset=diabetes-auto-lexical` line and no `refused`
  line.
- `make check` passes.
