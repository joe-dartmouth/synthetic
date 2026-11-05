---
id: explain-record-provenance
title: Define explainability for BKO, trace each record to its sources, and show the trace in the app
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: research/record-explainability
---

# Define explainability for BKO, trace each record to its sources, and show the trace in the app

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

The instructor's feedback on the 2026-11-04 status update asked the team to:
- define what explainability means in a BKO system;
- map each generated record to the BKB probabilities and the ontology statements that produced
  it, so every record can be traced back to them.

The pieces exist but nothing joins them:
- `bko-direct`'s `Draw` records each record's drawn concepts, their closure and the fields that
  fell back (`bko.generate.direct`), but not the shares it drew at or the records it copied from.
- A Messages arm's candidates file holds each record with its call, and the call log holds the
  prompt's gap statements and summary (`bko.generate.guidance`). Nothing reads the two together.
- The Streamlit app (`bko.app`) shows gaps, fills and scores, but no record's origin.

## Scope

**In.**

- **Phase 1, the definition.** A new ADR under `docs/adr/research/` defining explainability for a
  BKO record as the set of facts that determine it. It names, for each arm kind (`bko-direct`, the
  Messages arms, the baselines), which facts can be traced and which cannot. A language model's
  values, for example, trace to the prompt, not to a draw.
- **Phase 2, the trace.** A `Trace` per record, built from stored run files with no API call:
  - its cell and the gap statements that requested it, each with its antecedent and consequent
    labels from the pinned MONDO and HPO releases;
  - for each mapped concept: the share drawn at, the cell's and label group's priors behind it,
    and the S-nodes of the fused BKB they come from;
  - for `bko-direct`: each field group's source training record id, and the fields that fell
    back to the label group;
  - for a Messages arm: the call that returned it, the guidance level, and the summary and
    profile the prompt carried;
  - its validation outcome.

  `bko-direct` writes the source record ids and shares into its candidates lines as optional
  fields, bumping the candidates schema. Older files trace what they hold.
  A command, `python -m bko.evaluate trace <run-folder> --holdout <h> --arm <arm> [--record <i>]`,
  prints the traces as JSON lines.
- **Phase 3, the app.**
  - **Earlier runs:** a run folder with a candidates file gets a record picker (arm, holdout,
    record). It shows the selected record's fields beside its trace: the gap statements with
    their ontology labels, a table of concepts with their shares and priors, the source records
    for `bko-direct`, and the validation outcome.
  - **Demo, fill and score:** a BKO fill shows the same trace for each added row.
  - Both views read stored files and make no API call.
  - The gate drives both through `streamlit.testing.v1.AppTest`, as the app's other views are
    tested.
- **Phase 4, records.** `docs/DEVELOPMENT.md` (App, and the `trace` command), `src/bko/app/README.md`
  and `src/bko/evaluate/README.md`.

**Out.**
- Explaining a classifier's predictions (feature attribution); this unit explains where the
  training records came from.
- Explaining a baseline's rows beyond naming the method and its seed.

## Acceptance

- A test builds a `bko-direct` fill in `tmp_path` and checks each record's trace:
  - the drawn concepts equal the record's concept fields;
  - each share equals `bko.gaps.shares.shares` at that cell;
  - each source record id holds the copied values.
- A Messages arm's trace, rebuilt from a fixture call log and candidates file, names the call and
  the gap statements its prompt carried.
- The app's Earlier runs and Demo views show a trace through `AppTest`, with no API call.
- `make check`.
