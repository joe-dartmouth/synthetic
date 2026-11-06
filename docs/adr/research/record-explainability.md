---
status: accepted
topic: research
created: 2026-11-05
updated: 2026-11-05
---
# A record's explanation is the stored or re-derivable facts that determine it

## Context

A reader who sees a generated row asks why it looks as it does. Each arm kind determines its rows
differently. `bko-direct` draws them from the fused BKB's shares and training records with a seeded
generator, so every fact behind a record can be recomputed. A Messages arm sends a prompt and keeps
the model's reply; the prompt and the call are stored, and the model's choice of value within them
is not. A baseline fits a method at a seed. A run folder keeps the call log, the draw log and the
`arm` sha256 of each stored fill (ADR `research/rebuilt-rows-store`), but no per-record account.

## Decision Drivers

* **Must state only facts the run folder stores or a re-draw reproduces**, never a guess at what a
  model did.
* **Must refuse a re-draw that does not reproduce the stored fill.**
* **Must add no random draw to `bko-direct`**, so a seed's records stay the same.
* **Must keep source record ids out of anything a user downloads.**

## Decision

1. **Definition.** A record's explanation is the stored or re-derivable facts that determine it.
2. **Per arm kind.**
   * `bko-direct`: the cell and quota of the count-based request; per concept, the share, the cell
     and label-group priors with their counts and `GROUP_WEIGHT`, and the S-node names (head
     `(concept, present)`, tail `(__Source__[concept], <source>)`); per field, the source training
     record id; the fallback fields; the closure; and the validation outcome.
   * Messages arms: the call index, the guidance level, the prompt's sections as `prompt.sections`
     returns them (the gap statements with their labels as rendered), and the validation outcome.
   * Baselines: the method and the seed.
   * Not traceable: a model's choice of value within its prompt.
3. **Mechanism.** `bko.evaluate.trace` re-draws a `bko-direct` fill at the logged attempt's seed
   with no extra random draw, and refuses unless the re-drawn fill's digest equals the sha256 of
   the folder's `arm` line. It writes `trace.jsonl` with sorted keys, one record per line.
   `Draw.sources` carries the source ids out of the draw (ADR `research/bkb-direct-generation`).
4. **Readers.** The app reads only `trace.jsonl` and the Demo's in-memory fill
   (`bko.app.provenance_frames` builds the frames); a Demo fill of a Messages arm is traced from
   its own call, with no read of `data/` (ADR `research/streamlit-app-design`).
5. **Source ids stay private.** They live in the `trace.jsonl` file, written at mode 0600, and in
   the run folder, and never in the app's download.

## Rationale

A re-draw needs the seed, the request and the data, all of which a run folder pins, and the digest
check turns "the same code and data" into a test instead of an assumption. Keeping the source ids
on `Draw` costs no draw, so the stored fills stay valid. A Messages arm's record is the model's
reply to a stored prompt; naming the prompt's sections and the call states what the run kept and
claims nothing about why the model chose a value. A source id names a training record, which can be
a real patient's row; the explanation is useful with it and the download does not need it.

## Consequences

**Gained**: a per-record account of a `bko-direct` fill that a digest check keeps honest, and the
prompt and call behind a Messages record, with no API call.
**Accepted**: a Messages record's explanation stops at its prompt; a `bko-direct` trace needs the
data the run used and refuses on any drift; `trace.jsonl` holds record ids and stays in its run
folder.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| New fields in the candidates file (a schema 4) | The user chose to keep the file's schema; a re-draw holds the same facts | 2026-11-05 |
| Source ids in the app's download | They name training records; the download is the one artifact that leaves the folder | 2026-11-05 |
| A guess at a model's choice of value | Nothing stored or re-derivable determines it | 2026-11-05 |

## Enforcement

- Owner: `src/bko/evaluate/`, with the draw in `src/bko/generate/direct.py` and the readers in
  `src/bko/app/`
- Pinned by: `tests/test_run_folders.py`, `tests/test_generate_direct.py`

## Reversal Conditions

The candidates file gains a per-record explanation field (then the re-draw is redundant); `bko-direct`
draws through a generator the logged seed does not reproduce; or the app's download includes
`trace.jsonl`.

## Related

- [Direct generation draws each mapped concept at its fused share and reads fields from matching records](bkb-direct-generation.md)
- [The Streamlit app: one split of E1's harness, one paid fill per confirmed click](streamlit-app-design.md)
- [Run folders](../tooling/run-folders.md)
- [Record generation design](record-generation-design.md)
