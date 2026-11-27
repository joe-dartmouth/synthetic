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
system that aligns one ontology with another, not a dataset's codebook with one. Both are named
in the registry; neither is run here.

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
   unreviewed source is a candidate state, its live holders the options, the first ten by curie.
   V and E codes are skipped, as the hand generator skips them, and a code whose live exact
   holders do not nest maps nothing; `lexical` logs each such count by reason
   (`bko.ontology.codes.REASONS`). No other state of such a field is matched by label.
3. **The label route.** Every other coded state, the label's included, is matched by its codebook
   meaning (`bko.ontology.lexical`). A state is skipped when its meaning has no word or its first
   word is a negative word (`absent`, `no`, `none`, `not`, `norm`, `normal`, `false`, `negative`,
   `without`); a negative word later in a meaning, as in "diabetes without complications",
   qualifies a concept and does not skip it. A meaning whose words are all digits or generic words
   (`present`, `yes`, `true`, `positive`, `up`, `down`, `steady`, `ch`, `changed`, `high`, `low`)
   is matched by its field's description instead, with the description's parenthetical legend and
   every generic, negative and stop word dropped (`a`, `among`, `an`, `and`, `any`, `as`, `at`,
   `by`, `during`, `for`, `from`, `in`, `its`, `of`, `on`, `or`, `s`, `the`, `to`, `with`,
   `within`). Text is NFKC-normalized, casefolded and NFKC-normalized again, then split into
   words: runs of alphanumeric characters and combining marks. Targets are each live term's name
   and EXACT synonyms, among the terms of each
   release whose curie carries the release's prefix. The score is token-set Jaccard. Options are
   ranked by score, then the sorted order of the dataset's release prefixes, a name over a
   synonym, then the smaller curie, one per term. The first option maps the state when its score
   is at or above `THRESHOLD`, 4/5. Numeric columns stay unmapped. Rows are sorted by
   (field, code, concept).
4. **The local judge.** `bko.generate.judge` sends one call per candidate state: the top ten
   options and `none` as a strict `{"concept": enum}` schema, temperature 0 and seed 0, through
   `bko.generate.local.local_send` with its payload builder, after checking the server. A reply
   outside the enum is refused, and so is a run with no candidate state. Its model is
   `gemma-4-26b-a4b-q8_0` by default and any local model served with thinking off, since a
   verdict's output limit leaves no room for reasoning.
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
   `diag_3`; `--part` is refused for any other dataset, and `bko.ontology.auto_tables` merges the
   two parts, refusing parts that share a state, name one concept two ways, or name another
   dataset or part. Each module is written as UTF-8 bytes, and the command's closing line logs
   their sha256; a lexical module writes each row's `source`.
7. **Measures.** `bko.ontology.auto_measures`: precision and recall over (field, code, concept)
   atoms against the hand rows, exact curie, each None on an empty denominator; a row that is not
   one `Codes` clause is one atom keyed on each clause's type, field and sorted codes. Apart from
   the exact matches, `hierarchy_hits` counts the built atoms that miss the hand rows but name a
   strict `is_a` ancestor or descendant, in the shared prefix's release, of a hand concept of the
   same (field, code). For each `auto-local` table, the states the judge was asked about and the
   count it mapped. `python -m bko.ontology.auto_measures` writes all of them per table.
8. **The permuted control.** `permuted(rows, seed=...)` keeps every row's clauses and moves its
   concepts, with their labels, by a derangement of the distinct concepts of each release prefix:
   a permutation is redrawn until no concept keeps its place. A prefix with one concept cannot
   move; `unmovable` counts the rows it holds, 0 for both hand tables, so no permuted row keeps
   its hand concept.
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

**Gained**: a mapping for any frame with a codebook, at no cost, and measures of how far it is
from the hand table.
**Accepted**: a meaning whose words differ from the release's names maps nothing at 4/5, so
`auto-lexical` trades recall for precision; a field description holds more words than a concept
name, so a generic state rarely reaches the threshold; the local judge's verdicts depend on the
pinned weights and the server's build, which its run logs name. `diabetes-auto-lexical` holds no
row: every Diabetes 130 indicator's state reads `present`, a generic meaning, so it is matched by
its field's description, and with the legend and stop words dropped each query still carries
`three diagnoses`, so none reaches a term at 4/5. With no concept drawn there is no `is_a` pair,
so `plan_hierarchy` refuses `bko-hier-w` on it, and `e22-auto-lexical-*` is registered without
`bko-hier-w`; its `bko-direct` reads no concept, as `bko-nomap` reads none. Every other
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
- Pinned by: `tests/test_ontology_automatic.py` (the cross-reference route on each kind of code
  and its candidates, the threshold, negative and generic states, numeric columns, ties, no
  records taken, no import of a later stage, the render under two hash seeds, the structural
  check, the `lexical` command's bytes, digest and exit codes, the permuted control's derangement
  and the verdict table's rows), `tests/test_ontology_lexical.py` (the word rules, the 4/5
  boundary, the tie orders, the label field, the prefix filter, NFKC folding and the option cap),
  `tests/test_ontology_auto_measures.py` (atom keys, hierarchy hits, judged counts and the
  `measures` command), `tests/test_ontology_obo.py` (`Term.synonyms`),
  `tests/test_generate_judge.py` (temperature 0 and a seed, the request bytes and system turn by
  sha256, the top ten options and `none`, one call per state, a refused verdict, the eleventh
  option refused, a failed call named, the server checked first, the command's log lines and
  exit codes, rows a pure function of the verdict table) and `tests/test_generate_local.py`
  (every existing local request's bytes); `tests/test_ontology_auto_tables.py` (each table's
  `bko.ontology.schema.rows_digest`: `auto-lexical` `diabetes`
  `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`, no row; `auto-lexical`
  `diabetes-codes` `bf119a4d05428defc250c070b551d232f6a6d5f8a2421d84d01490fcd606b5c2`;
  `auto-local` `diabetes` `7a52217e6d908c456cfa13d60811182fb047f16caf00e2eb1b295b596b0b16f0`;
  `auto-local` `diabetes-codes`
  `ab632685d5bf2995b580bd09e071dbd863e79319a862d500ee95e0ab935b9109`; permuted `diabetes`
  `b1a0cc61c9daaee98d7507d47ad41536ffd31f5678f8efa5c429937c5fde783b`; permuted
  `diabetes-codes` `8f90fe5a70fe83467f7bad88d61754a74cdc4313cdfdc17b02328ca88eb6775c`; each
  permuted table moving every row, the merge of a split table's parts, and the reversal
  condition); with `--releases`, `tests/test_ontology_auto_pinned.py` (each lexical module a
  rebuild, each `auto-local` table's rows `rows_from_verdicts` of its verdicts on the
  records-built frame, each verdict an offered option or `none` of a surveyed state with the
  cross-referenced states' verdicts their rows' concepts, and the check accepting every table's
  rows) and `tests/test_evaluate_auto_draws_pinned.py` (which variants build `bko-hier-w`'s
  hierarchy and `bko-tree`'s trees, the Consequences' refusal included);
  `tests/test_evaluate_direct.py` (a hierarchy or a tree the rows cannot build, refused before
  any arm fits); `tests/test_evaluate_auto_runs.py` (E22's sixteen runs and their slices, each
  family's dataset, the families of one frame differing only in dataset and the gated
  `bko-hier-w`, and the reference arm `bko-direct`) and `tests/test_named_runs_slices.py`
  (E22's names in `RUNS`' order)

## Reversal Conditions

**The rule maps a hand-mapped state to another concept on the pinned releases**: a row of
`auto_lexical_codes_rows.py` whose (field, code) a hand code row holds names another concept.

## Related

- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [A mapping change or a field projection is a new dataset name over its base](dataset-variants.md)
- [On-device generators: a llama.cpp server on the loopback interface, weights fixed by digest](local-generator-backend.md)
