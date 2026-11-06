---
status: accepted
topic: research
created: 2026-10-21
updated: 2026-11-05
---
# The Streamlit app: one split of E1's harness, one paid fill per confirmed click

## Context

The Working Prototype milestone is a Streamlit page over the pipeline that calls the stages and
adds no pipeline logic of its own. E1's harness already fills and scores one holdout
in one call, `bko.evaluate.run.score_holdout`, given each arm's runner. Streamlit reruns the whole
script on every interaction, stops a running script only at its next streamlit call, binds every
interface unless `server.address` is set, and sends usage statistics unless
`browser.gatherUsageStats` is false. A BKO arm's fill calls the Anthropic Messages API, and every
call costs money.

## Decision Drivers

* **Must add no pipeline logic**: the app composes the stages' public functions.
* **Must never pay twice or lose a paid result to a rerun.**
* **Must bound the spend of one click.**
* **Must keep the API seam in `bko.generate.api`**, so the paid-request guard covers the app.
* **Should run every path that spends nothing in the gate.**

## Decision

We will build the page as follows.

* **Evaluate shape**: one "Fill and score" action calls `score_holdout` once, with the chosen arm
  wrapped in a recorder that keeps its fill, and no replica fill.
* **Replay**: a BKO fill answers live or from the call log an earlier app fill of that arm wrote.
* **Register**: an interactive session is not an EVIDENCE run; the app's `live` test runs once as
  a G1 run.
* **Orchestration**: `bko.app.steps` composes the stages' public APIs; the BKO wiring is
  `bko.evaluate.guided_run.guided_arms`, which the evaluate command and `bko.app.steps` both call; the
  app loads UCI once through `bko.evaluate.data.load`, as `score` does, and splits, observes and
  names its knowledge from that one `Data` value.
* **Fill path**: E1's runner interface over E1's gap request on the default split, E1's first
  holdout.
* **Methods**: the arms of E1 that add rows, row-adding methods first, then the three BKO arms.
* **Comparators**: every arm of `METHOD_ARMS` beside the fill; a BKO arm fills first, so a failed
  call costs no comparator's fit.
* **Datasets**: UCI Heart Disease only.
* **Paid guard**: a confirmation box keyed to a per-session run counter and re-checked after the
  click; `SPEND_CEILING` per fill; the send built inside the action; the loopback bind; one action
  per process under a lock.
* **Config home**: the flags live in the `make app` recipe, and nowhere else.
* **Verdicts**: a demo fill shows one split's measures and no verdict; the Earlier runs view shows
  E1's stored criteria table and the validity verdict `bko.evaluate.validity.verdict` gives on
  the stored scores.
* **`SPEND_CEILING`**: 0.25 USD per fill.
* **Run folders**: each fill keeps its run folder (ADR `tooling/run-folders`).
* **Earlier runs**: the page's Earlier runs view reads run folders' stored artifacts through the
  stages' public readers (`bko.evaluate.scores.parse`, `bko.evaluate.validity` and
  `bko.generate.candidates.read`) and pandas for the CSVs, and recomputes nothing from `data/`
  (`docs/DEVELOPMENT.md`, App, lists what it shows).
* **Provenance**: the Earlier runs view gets a picker over a folder's `trace.jsonl` and reads
  nothing else for it, so it still recomputes nothing from `data/`; the Demo traces a Messages
  fill from the fill's own call log and candidates file, and the download carries no source
  record id (ADR `research/record-explainability`).

## Rationale

One action keeps the fill's private steps private: a second step would need the harness's fill
public and a runner that reuses it. A replica fill would add many paid calls and minutes per BKO
click, and no criterion reads its fidelity. Replay lets the demo run free and repeatably from a
recorded log, with the refusals the evaluate command already applies. A row per click would flood
the register with readings no criterion uses. One loader keeps the app's split, cells and
releases the ones the gaps, fusion and generate commands read.
The completeness request is another request, and scoring it would pay twice. Controls add no rows,
so they cannot fill. A confirmation keyed to a run counter cannot carry over to a second click; a
credential pre-check would itself call the API, and a process-wide spend ledger guards a server
this app never runs. A config file is a second home for the same flags. E1's 25 USD ceiling bounds
a whole experiment; one fill makes a few calls.

## Consequences

**Gained**: every path that spends nothing runs in the gate; one paid transaction per click; a
free, repeatable demo from a recorded log.
**Accepted**: no live rejected-row counter; no
cancel of an in-flight call, and a closed tab or Ctrl-C waits for the running action, CTGAN and TVAE
fits included; the page's numbers never equal E1's mean over its holdouts; a closed or reloaded tab
loses the shown result, which the run folder keeps and a replay answers again for free; clicks queue
during an action; an AppTest that reaches the SDK fails on what AppTest's `run` raises, with the
guard's message on the script thread's exception; DNS rebinding to `127.0.0.1:8501` can still drive
fills, each bounded by the ceiling and a confirmation.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Two steps, fill then evaluate | Needs the harness's fill public and a runner that reuses it | 2026-10-21 |
| A replica fill per BKO click | Many extra paid calls and minutes per click; no criterion reads fidelity | 2026-10-21 |
| Live answers only | A demo would spend on every showing | 2026-10-21 |
| A register row per click | Interactive readings answer no criterion | 2026-10-21 |
| Copying the commands' UCI load sequence into the app | Two homes drift; one loader serves the app and the commands that split once | 2026-10-22 |
| Copying the evaluate command's BKO wiring into the app | Two homes drift; one public function serves both | 2026-10-22 |
| The completeness request as the fill path | Another request; scoring it would pay twice | 2026-10-21 |
| Controls as fill methods | They add no rows | 2026-10-21 |
| `none` as the only comparator | Hides how the fill compares with the baselines | 2026-10-21 |
| Synthea or an uploaded dataset | No fill path for either | 2026-10-21 |
| A credential pre-check | Costs a call; a missing credential already stops the fill before any spend | 2026-10-21 |
| A process-wide spend ledger | The app serves loopback only, to one user | 2026-10-21 |
| `.streamlit/config.toml` | A second home for the flags the recipe sets | 2026-10-21 |
| A headline improvement | One split is not E1's verdict | 2026-10-21 |
| E1's 25 USD ceiling per fill | Bounds a whole experiment, not one click | 2026-10-21 |
| `st.cache_data` | Cannot pickle `Ontology`, `Cells` or `Request` | 2026-10-21 |
| Recomputing readings from `data/` in the Earlier runs view | Copies the gaps and fusion commands' private steps; the stored artifacts already hold what the register reads | 2026-10-21 |

## Enforcement
- Owner: `src/bko/app/`
- Pinned by: `tests/test_repo_gate.py`, `tests/test_app_page.py`, `tests/test_app_imports.py`,
  `tests/test_app_steps.py`, `tests/test_app_views.py`, `tests/test_app_browse.py`

## Reversal Conditions
The app is served beyond loopback (then authentication and a spend ledger); `GuidedArm` gains a
per-batch hook; a criterion reads replica fidelity; a second dataset gains a fill path.

## Related
- [E1 evaluation protocol](e1-evaluation-protocol.md)
- [Call-log replay](call-log-replay.md)
- [Record generation design](record-generation-design.md)
- [A record's explanation is the stored or re-derivable facts that determine it](record-explainability.md)
- [Run folders](../tooling/run-folders.md)
- [uv, ruff and ty](../tooling/uv-ruff-ty.md)
- [Paid API request guard](../tooling/paid-api-request-guard.md)
