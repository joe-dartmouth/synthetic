---
status: accepted
topic: research
created: 2026-11-05
updated: 2026-11-08
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
     (`bko.generate.direct.Draw.closed`), not share priors.
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
   attempt's seed with that arm's presence and no extra random draw, and rebuilds a perturbation
   arm's fill from its candidate lines over that re-draw. It refuses (exit 3) a holdout whose
   `split` line the data does not reproduce, a re-drawn fill whose digest is not the sha256 of the
   folder's `arm` line, re-drawn kept records that are not the candidates file's kept lines, a
   direct fill whose `arm` line's params are not a JSON object or whose `presence` is not its
   arm's, and a perturbation fill whose rebuilt rows miss its `arm` sha256, whose returned
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
   `bko-direct-perturbed`, `changed` or `unchanged` for `bko-direct-pram`.

## Rationale

A re-draw needs the seed, the request and the data, all of which a run folder pins, and the digest
check turns "the same code and data" into a test instead of an assumption. Keeping the source ids
on `Draw` costs no draw, so the stored fills stay valid. A Messages arm's record is the model's
reply to a stored prompt; naming the prompt's sections and the call states what the run kept and
claims nothing about why the model chose a value. A perturbation row's outcome depends on a reply
and on the merge code that read it; storing it on the row's line keeps it readable after either
changes, and a schema version may grow when its runs are repeated. A source id names a training
record, which can be a real patient's row; the explanation is useful with it and the download does
not need it.

## Consequences

**Gained**: a per-record account of every direct and perturbation arm's fill that a digest check
keeps honest, and the prompt and call behind a Messages record, with no API call.
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

## Enforcement

- Owner: `src/bko/evaluate/trace.py`, with the trace lines in `src/bko/evaluate/trace_lines.py`,
  the direct arms' in `src/bko/evaluate/trace_direct.py`, the perturbation arms' in
  `src/bko/evaluate/trace_perturb.py`, the call log's reading in
  `src/bko/evaluate/call_prompts.py`, `Draw.sources` in `src/bko/generate/direct.py`,
  `sections` in `src/bko/generate/prompt.py` and the app's frames in
  `src/bko/app/provenance_frames.py`; decision 6's `perturb` key and each arm's results in
  `src/bko/evaluate/holdout_candidates.py`
- Pinned by: `tests/test_evaluate_trace.py`, `tests/test_evaluate_trace_direct.py`,
  `tests/test_evaluate_trace_perturb.py`, `tests/test_generate_direct.py`,
  `tests/test_generate_prompt.py`, `tests/test_run_folders.py`, `tests/test_app_trace.py`,
  `tests/test_app_imports.py`; decision 6 by `tests/test_evaluate_holdout_candidates.py`,
  `tests/test_evaluate_perturb.py` and `tests/test_evaluate_pram.py`

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
