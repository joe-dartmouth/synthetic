# `bko.app`: Streamlit interface

The Streamlit front end over the pipeline: pick a dataset, analyze its gaps, fill them, score the
result and download the augmented training set.

The app calls each stage's public functions and holds no pipeline logic. It fills E1's gap request
on the default split with one arm of `bko.evaluate.experiment.E1` that adds rows, scores it beside
every method of `bko.evaluate.experiment.METHOD_ARMS` with no verdict on E1's criteria, and keeps
each fill in a run folder (`bko.app.runs`). A second view shows the run folders on the machine, and a picker over a folder's `trace.jsonl` shows the facts behind each record (ADR [`research/record-explainability`](../../../docs/adr/research/record-explainability.md)); a demo fill of a Messages arm is traced from its own run folder (`bko.app.provenance_frames`).
[`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#app) holds the command, what a fill spends
and what the app writes; ADR
[`research/streamlit-app-design`](../../../docs/adr/research/streamlit-app-design.md) holds the
design.

- **Dependencies:** The `app` extra (`pyproject.toml`), and the stages it calls with their extras:
  `bko.datasets`, `bko.gaps`, `bko.fusion`, `bko.baselines`, `bko.evaluate` and `bko.generate`.
- **Owner:** Krish Sapru
- **Status:** built.
