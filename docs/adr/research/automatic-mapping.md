---
status: accepted
topic: research
created: 2026-11-26
updated: 2026-11-26
---
# An automatic mapping: codebook meanings matched to pinned releases, with no paid model

## Context

Every field row of the mapping table is written by hand from a frame's codebook and a pinned
release (ADR [`research/ontology-mapping-design`](ontology-mapping-design.md)). Whether the BKO
needs that work is an open question: if rows built by a rule or a small local model serve
`bko-direct` as well as the hand rows do, the method carries to a dataset no one has mapped.
[E22](../../EVIDENCE.md#e22-automatic-mappings-on-the-diabetes-130-removal) asks it on E6's
Diabetes 130 frame and on the `diabetes-codes` frame.

The nearest tool, NVIDIA auto-ontology (`autoontology-nd` in
[`docs/literature.md`](../../literature.md)), does not build such rows. Read at its source on
2026-11-18, it names one term of its own per table, its term prompt rules out `is_a` relations,
and it links no term to a standard ontology. BERTMap (`he-2022`) is, by its title, a BERT-based
system that aligns one ontology with another, not a dataset's codebook with one. Both are named in the registry; neither is run here.

## Decision Drivers

* **Must produce `FieldMapping` rows the existing readers take**, so every arm draws from them as
  from the hand rows.
* **Must call no paid model**, so the mapping is reproducible for free.
* **Must read no record-level statistic**, so a mapping cannot encode the outcome it is scored on.
* **Must be byte-reproducible**, so a committed module equals a rebuild.
* **Must keep the hand rows, every existing dataset and every local request's bytes unchanged.**

## Decision

1. **Two builders.** `auto-lexical` uses no model; `auto-local` judges the same candidates with
   the pinned on-device model `gemma-4-26b-a4b-q8_0`. An automatic mapping is the builder's rows
   alone. Each row's `source` is this record's id, `research/automatic-mapping`.
2. **The cross-reference route.** A field whose codebook meanings are ICD-9 diagnosis tokens maps
   each code by `bko.ontology.codes`' rule with no curated row: the one live MONDO holder of its
   `ICD9:` xref under `source="MONDO:equivalentTo"` that every other live exact holder descends
   from by `is_a` (ADR [`research/diabetes-diagnosis-codes`](diabetes-diagnosis-codes.md)). On
   `diabetes-codes` it gives the hand code rows. A code MONDO links only by `relatedTo` or an
   unreviewed source is a candidate state, its live holders the options; no other state of such a
   field is matched by label.
3. **The label route.** Every other coded state, the label's included, is matched by its codebook
   meaning. A state is skipped when a token of its meaning is a negative word (`absent`, `no`,
   `none`, `not`, `norm`, `normal`, `false`, `negative`, `without`) or it has no token. A meaning
   whose tokens are all digits or generic words (`present`, `yes`, `true`, `positive`, `up`,
   `down`, `steady`, `ch`, `changed`, `high`, `low`) is replaced by the field's description.
   Text is NFKC-normalized, casefolded, its non-alphanumeric characters made spaces and split
   into a token set. Targets are each live term's name and EXACT synonyms, among the terms of each
   release whose curie carries the release's prefix. The score is token-set Jaccard. Options are
   ranked by score, then the sorted order of the dataset's release prefixes, a name over a
   synonym, then the smaller curie, one per term. The first option maps the state when its score
   is at or above `THRESHOLD`, 4/5. Numeric columns stay unmapped. Rows are sorted by
   (field, code, concept).
4. **The local judge.** `bko.generate.judge` sends one call per candidate state: the top ten
   options and `none` as a strict `{"concept": enum}` schema, temperature 0 and seed 0, through
   `bko.generate.local.local_send` with its payload builder. A reply outside the enum is refused.
   An `auto-local` module holds `VERDICTS`, each state's concept or `none` keyed
   `"<field>=<code>"`, the cross-reference states included, and `LABELS`, each verdict concept's
   name; its rows are `rows_from_verdicts(frame, VERDICTS, LABELS)`, a pure function of that
   table.
5. **The leakage rule.** No builder reads a record-level statistic. A frame's codebook may list
   the cohort's distinct values, as `diabetes-codes`' does and as the hand code rows' generator
   reads them; its codebook is read through `bko.datasets.diabetes.load_diabetes` and
   `bko.ontology.code_frames`.
6. **Committed modules.** `python -m bko.ontology.automatic lexical <dataset>` and
   `python -m bko.generate.judge <dataset> --model gemma-4-26b-a4b-q8_0` write
   `auto_lexical_diabetes_rows.py`, `auto_lexical_codes_rows.py` and `auto_local_diabetes_rows.py`
   under `src/bko/ontology/`, each through `scratchpad/` then `mv`. `diabetes-codes`' verdict
   table is over the size cap in one module, so `--part low` and `--part high` write it as
   `auto_local_codes_rows_low.py` and `auto_local_codes_rows_high.py`, split at the field
   `diag_3`; `bko.ontology.auto_tables` merges them.
7. **Measures.** Precision and recall over (field, code, concept) atoms against the hand rows,
   exact curie, each None on an empty denominator. On `diabetes-codes`, the judged candidates are
   reported as the count the judge accepts.
8. **The permuted control.** `permuted(rows, seed=...)` keeps every row's clauses and permutes its
   concepts, with their labels, among the distinct concepts of each release prefix.
9. **Datasets and runs.** Each automatic or permuted table is a row-swap variant of `diabetes` or
   `diabetes-codes` (ADR [`research/dataset-variants`](dataset-variants.md)); E22 holds the runs.

## Rationale

A rule over the codebook and the release is the floor any learned matcher has to beat, and it
needs no model at all. The cross-reference route is the release's own statement of equivalence,
so it is the hand code rows' rule minus the one curated choice, which on the pinned release
changes no code's concept. Token-set Jaccard at 4/5 accepts a reordered or lightly extended name
and refuses a partial one; a generic meaning says nothing about its concept, while the field's
description names it. A local model at temperature 0 and seed 0, restricted to the rule's own
candidates, adds judgement without a paid call and without inventing a curie, and a committed
verdict table makes its output a fixed input to every later run. The permuted control keeps the
rows' shape and counts and breaks only what the concepts mean, so a hierarchy-reading arm's
result can be read against knowledge that is wrong.

## Consequences

**Gained**: a mapping for any frame with a codebook, at no cost, and two measures of how far it is
from the hand table.
**Accepted**: a meaning whose words differ from the release's names maps nothing at 4/5, so
`auto-lexical` trades recall for precision; a field description holds more words than a concept
name, so a generic state rarely reaches the threshold; the local judge's verdicts depend on the
pinned weights and the server's build, which its run logs name.
**Refused**: `diabetes-auto-lexical` holds no row. Every Diabetes 130 indicator's state reads
`present`, a generic meaning, so it is matched by its field's description, and no state's query
reaches a term at 4/5. With no concept drawn there is no `is_a`
pair, so `plan_hierarchy` refuses `bko-hier-w` on it, and `e22-auto-lexical-*` is registered
without `bko-hier-w`; its `bko-direct` reads no concept, as `bko-nomap` reads none. Every other
`diabetes` variant builds the hierarchy, and every `diabetes-codes` variant holds a tree field
for `bko-tree`.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A paid model as the judge | The mapping would cost money to rebuild and depend on a model that changes | 2026-11-26 |
| A model-proposed cutoff row for a numeric column | A cutoff needs a cited source; a model's number has none | 2026-11-26 |
| A mapping built from records | Breaks the leakage rule: the rows would carry the outcome they are scored on | 2026-11-26 |
| An SSSOM table as the output | The readers take `FieldMapping` rows; a second format adds a converter and no reading | 2026-11-26 |
| Running NVIDIA auto-ontology | It names its own terms and links none to a standard ontology (Context) | 2026-11-26 |

## Enforcement

- Owner: `src/bko/ontology/automatic.py`
- Pinned by: `tests/test_ontology_automatic.py` (the cross-reference route and its candidates,
  the threshold, negative and generic states, numeric columns, ties, no records taken, no import
  of a later stage, the render under two hash seeds, the structural check, precision and recall,
  the permuted control and the verdict table's rows), `tests/test_ontology_obo.py`
  (`Term.synonyms`), `tests/test_generate_judge.py` (temperature 0 and a seed, the top ten options
  and `none`, one call per state, a refused verdict, rows a pure function of the verdict table)
  and `tests/test_generate_local.py` (every existing local request's bytes); with `--releases`,
  `tests/test_ontology_auto_pinned.py` (each lexical module a
  rebuild, each `auto-local` table's rows `rows_from_verdicts` of its verdicts on the
  records-built frame, the check accepting every table's rows, and each table's
  `bko.ontology.schema.rows_digest`: `auto-lexical` `diabetes`
  `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`, no row; `auto-lexical`
  `diabetes-codes` `bf119a4d05428defc250c070b551d232f6a6d5f8a2421d84d01490fcd606b5c2`;
  `auto-local` `diabetes` `3b71bbe6e3f5cd8cf15965252a6b3afb60a5dc81022fb5855fa0e57369115e7a`;
  `auto-local` `diabetes-codes`
  `45caeb10a710164738fb6f28eccbd811dbcd78e991c9fcb0ac8f5da3293a3a63`) and
  `tests/test_evaluate_auto_draws_pinned.py` (which variants build `bko-hier-w`'s hierarchy and
  `bko-tree`'s trees, the Consequences' refusal included); `tests/test_evaluate_direct.py`
  (a hierarchy or a tree the rows cannot build, refused before any arm fits);
  `tests/test_evaluate_auto_runs.py` (E22's sixteen runs and their slices, each family's dataset,
  the families of one frame differing only in dataset and the gated `bko-hier-w`, and the
  reference arm `bko-direct`) and `tests/test_named_runs_slices.py` (E22's names in `RUNS`'
  order)

## Reversal Conditions

**The rule maps a hand-mapped state to another concept on the pinned releases**: a row of
`auto_lexical_codes_rows.py` whose (field, code) a hand code row holds names another concept.

## Related

- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [A mapping change or a field projection is a new dataset name over its base](dataset-variants.md)
- [On-device generators: a llama.cpp server on the loopback interface, weights fixed by digest](local-generator-backend.md)
