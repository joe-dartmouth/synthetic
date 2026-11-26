---
status: accepted
topic: research
created: 2026-11-05
updated: 2026-11-26
---
# A record's explanation is the stored or re-derivable facts that determine it

## Context

A reader who sees a generated row asks why it looks as it does. Each arm kind determines its rows
differently. `bko-direct` draws them from the fused BKB's shares and training records with a seeded
generator, so every fact behind a record can be recomputed; `bko-logic` draws the same way with
presence from the ontology's constraints alone. `bko-direct-pram` and `bko-direct-perturbed` change
a direct draw's unmapped fields, by PRAM or by a rewrite call, and their candidate lines hold the
changed rows. A Messages arm sends a prompt and keeps
the model's reply; the prompt and the call are stored, and the model's choice of value within them
is not. A baseline fits a method at a seed. A run folder keeps the call log, the draw log and the
`arm` sha256 of each stored fill (ADR `research/rebuilt-rows-store`), but no per-record account.

## Decision Drivers

* **Must state only facts the run folder stores or a re-draw reproduces**, never a guess at what a
  model did.
* **Must refuse a re-draw that does not reproduce the stored fill.**
* **Must add no random draw to a direct arm**, so a seed's records stay the same.
* **Must keep source record ids out of anything a user downloads.**

## Decision

1. **Definition.** A record's explanation is the stored or re-derivable facts that determine it.
2. **Per arm kind.**
   * `bko-direct`: the cell and quota of the count-based request; per concept, the share, the cell
     and label-group priors with their counts and `GROUP_WEIGHT`, whether it was drawn present, and
     the S-node names (head `(concept, "a")`, the class component in its one state
     `bko.ontology.fragments.INDIVIDUAL`; tail `(__Source__[concept], <source>)`); per field, the
     source training record id; the fallback fields; the closure; and the validation outcome.
   * `bko-logic`: as `bko-direct`, drawn with presence `LOGIC`; its concepts are the closed draw
     (`bko.generate.draws.Draw.closed`), not share priors.
   * `bko-nomap`: as `bko-direct`, drawn with an empty mapping table (ADR
     `research/no-mapping-ablation`); its `concepts` and `closed` are empty, and each copied
     column names its source training record id.
   * `bko-donor`, `bko-pool-random` and `bko-pool-all`: as `bko-nomap`, with a pooled set of
     fields copied by the pooled rule (ADR `research/matched-knowledge-controls`); their `concepts`
     and `closed` are empty, each copied column names its source training record id, and a pooled
     field read from the label group's branch of the rule is a fallback field.
   * `bko-kb`: as `bko-logic`, its concepts the closed draw, with no donor (ADR
     `research/matched-knowledge-controls`): each mapping group's codes are drawn uniformly over
     the codebook tuples showing the drawn pattern and `age` uniformly inside the cell's band and
     the training split's age range, so its `sources` and fallback fields are empty.
   * `bko-capped`: as `bko-direct`, drawn under a donor floor and a reuse cap (ADR
     `research/sampler-donor-limits`); a field read from outside the cell below the floor is a
     fallback field.
   * `bko-micro`: as `bko-capped`, and each averaged field names, under `averaged`, the k
     training record ids its mean was taken over, the primary's first; its source is the
     primary's id (ADR `research/sampler-donor-limits`, Decisions 11 to 14). `averaged` is
     written only when non-empty, so every other arm's trace line keeps its bytes.
   * `bko-tree`: as `bko-direct` for every concept no tree field reads, then, per tree field, its
     drawn leaf with each level's counts, weight, conditional and share (ADR
     `research/ancestor-shrunk-shares`); the field names its source training record id.
   * `bko-hier-0`, `bko-hier-w` and `bko-hier-closed`: as `bko-direct`, and each broad concept of
     an `is_a` pair adds its pair parts: the narrow concept it is drawn given, the weighted counts
     of records showing both and showing the narrow one, the pseudo-count, the conditional, the
     complement and whether its clamp fired (ADR `research/bkb-direct-generation`, Decision 12).
   * `bko-coherent`: as `bko-direct-pram` below, and each free field names as its source the
     training record id of its leaf donor, the training row whose value sequential CART copied
     (ADR `research/free-field-synthesis`).
   * `bko-direct-pram` and `bko-direct-perturbed`: the record as its candidate line holds it, the
     facts of the direct draw underneath, and a `perturb` object: the line's `perturb` result
     (decision 6; on a schema-3 line, `perturbed` or `unchanged` by comparison with the draw when
     the line names a call, `kept` when it names none, and PRAM's `changed` or `unchanged`), the
     fields that differ from the draw, and the draw's values for them. That `kept` is a result
     only the trace writes, for a rewrite row that kept its draw for a reason a schema-3 line
     does not hold; it is none of the arm's stored results.
   * Messages arms: the call index, the guidance level, the prompt's sections as `prompt.sections`
     returns them (the gap statements with their labels as rendered), and the validation outcome.
   * Baselines: the method and the seed.
   * Not traceable: a model's choice of value within its prompt.
3. **Mechanism.** `bko.evaluate.trace` re-draws a fill of an arm of `DIRECT_ARMS` at the logged
   attempt's seed with that arm's spec and no extra random draw, and rebuilds a perturbation
   arm's fill from its candidate lines over that re-draw. `bko-coherent` then re-runs its change
   at the fill's seed, whose leaf draws read their own stream, and refuses unless the changed
   rows are the candidate lines' records. It refuses (exit 3) a holdout whose
   `split` line the data does not reproduce, a re-drawn fill whose digest is not the sha256 of the
   folder's `arm` line, re-drawn kept records that are not the candidates file's kept lines, a
   direct fill whose `arm` line's params are not a JSON object or whose logged spec fields are
   not its arm's, and a perturbation fill whose rebuilt rows miss its `arm` sha256, whose returned
   attempt is not one line per draw, or whose unperturbed columns differ from the re-draw. It
   writes `trace.jsonl` with sorted keys, one line per record, one per fill for a baseline.
   `Draw.sources` carries the source ids out of the draw (ADR `research/bkb-direct-generation`).
4. **Readers.** The app reads only `trace.jsonl` and a Demo fill's own run folder
   (`bko.app.provenance_frames` builds the frames); a Demo fill of a Messages arm is traced from
   its own call log and candidates file, with no read of `data/` (ADR
   `research/streamlit-app-design`). It renders a trace only as frames, JSON and plain code blocks.
5. **Source ids stay private.** They live in the `trace.jsonl` file, kept in its run folder, and
   never in the app's download.
6. **Perturbation outcomes are stored.** The candidates file (schema 4) carries each perturbation
   row's outcome in its line's `perturb` key: `perturbed`, `unchanged` or `refused_<reason>` for
   `bko-direct-perturbed`, `changed` or `unchanged` for `bko-direct-pram` and `bko-coherent`.
7. **Attribution.** A reading over `trace.jsonl` files counts, per arm, what each record field
   can be attributed to. An arm's kind is read from one trace line: a line whose `record` is null
   is a baseline's; one holding a `perturb` key is a perturbation arm's; one whose `request` holds
   `quota` is a direct arm's; any other is a Messages arm's. Each field gets one of five classes,
   in this precedence: knowledge fact, donor mean, training donor id, prompt section only, method
   and seed only.
   * Direct: a field a mapping row's `when` clause reads (the label aside), under an arm whose
     `DrawSpec` reads the mapping table at a presence other than `POOLED`, is a knowledge fact;
     else a field the line's `averaged` names is a donor mean (`donor-mean`), its value the mean
     of its donors'; else a field with a source id is a donor, a pooled field among them, since
     the mapping chose only which fields pool and the value is a donor's; else (`sex`, the label, and `bko-kb`'s
     `age`, drawn from its band and range by the seed) it is method and seed. A `bko-kb` mapped
     field is a knowledge fact, its value drawn from the mapping table and the codebook alone.
   * Perturbation: a field the line's `perturb` names as changed is prompt section only under the
     rewrite arm, method and seed only under PRAM, and a training donor id under `bko-coherent`,
     whose leaf donor it names, a mapped field among them, since its value is no longer the drawn
     fact; this class comes before every other. Every other field is read as a direct one.
   * Messages: every field is prompt section only; every field of a record whose call the log
     lacks (`untraced`) is method and seed only, and the reading counts such records.
   * Baseline: one count per fill, method and seed only, in the unit `fill`, since the trace holds
     no baseline record.

## Rationale

A re-draw needs the seed, the request and the data, all of which a run folder pins, and the digest
check turns "the same code and data" into a test instead of an assumption. Keeping the source ids
on `Draw` costs no draw, so the stored fills stay valid. A Messages arm's record is the model's
reply to a stored prompt; naming the prompt's sections and the call states what the run kept and
claims nothing about why the model chose a value. A perturbation row's outcome depends on a reply
and on the merge code that read it; storing it on the row's line keeps it readable after either
changes, and a schema version may grow when its runs are repeated. The contrast the attribution
reading claims is attribution to knowledge facts, which only a direct draw that reads the mapping
table makes; donor attribution is not unique to the direct arms, since a nomap draw names a donor
for every copied field, so knowledge takes precedence over donor. A source id names a training
record, which can be a real patient's row; the explanation is useful with it and the download does
not need it.

## Consequences

**Gained**: a per-record account of every direct and perturbation arm's fill that a digest check
keeps honest, the prompt and call behind a Messages record, and a count per arm of each field's
attribution class, with no API call.
**Accepted**: a Messages record's explanation stops at its prompt, and a rewrite row's at its
result and changed fields; a direct or perturbation trace needs the data the run used and refuses
on any drift; `trace.jsonl` holds record ids and stays in its run
folder.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Source ids in the app's download | They name training records; the download is the one artifact that leaves the folder | 2026-11-05 |
| A guess at a model's choice of value | Nothing stored or re-derivable determines it | 2026-11-05 |
| Keep schema 3 and re-derive perturbation outcomes in `trace` | It needs the reply and the merge code to stay unchanged, and a schema version may grow with its runs repeated | 2026-11-08 |
| Donor over knowledge as the attribution precedence | Every mapped field of a direct record also names a donor, so the knowledge contrast would vanish | 2026-11-24 |
| Per-record attribution counts for a baseline | The trace holds no baseline record, only the fill's method and seed | 2026-11-24 |
| A fifth attribution class for untraced Messages records | The reading names four classes; an untraced record's fields are method and seed only, and the log counts the records | 2026-11-24 |
| A new trace format for the attribution reading | Every class is read from the lines `trace` writes now | 2026-11-24 |

## Enforcement

- Owner: `src/bko/evaluate/trace.py`, with the trace lines in `src/bko/evaluate/trace_lines.py`,
  the direct arms' in `src/bko/evaluate/trace_direct.py`, the perturbation arms' in
  `src/bko/evaluate/trace_perturb.py`, the call log's reading in
  `src/bko/evaluate/call_prompts.py`, `Draw.sources` in `src/bko/generate/draws.py`,
  `sections` in `src/bko/generate/prompt.py` and the app's frames in
  `src/bko/app/provenance_frames.py`; decision 6's `perturb` key and each arm's results in
  `src/bko/evaluate/holdout_candidates.py`; decision 7's reading in
  `src/bko/evaluate/attribution.py`
- Pinned by: `tests/test_evaluate_trace.py`, `tests/test_evaluate_trace_direct.py`,
  `tests/test_evaluate_trace_perturb.py`, `tests/test_generate_direct.py`,
  `tests/test_generate_prompt.py`, `tests/test_run_folders.py`, `tests/test_app_trace.py`,
  `tests/test_app_imports.py`; decision 6 by `tests/test_evaluate_holdout_candidates.py`,
  `tests/test_evaluate_perturb.py` and `tests/test_evaluate_pram.py`; decision 7 by
  `tests/test_evaluate_attribution.py`

## Reversal Conditions

The candidates file gains a per-record explanation field for `bko-direct` or the Messages arms
(then the re-draw is redundant; decision 6's `perturb` key covers the perturbation arms alone);
`bko-direct` draws through a generator the logged seed does not reproduce; or the app's download
includes `trace.jsonl`.

## Related

- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [The Streamlit app: one split of E1's harness, one paid fill per confirmed click](streamlit-app-design.md)
- [Run folders](../tooling/run-folders.md)
- [Record generation design](record-generation-design.md)
