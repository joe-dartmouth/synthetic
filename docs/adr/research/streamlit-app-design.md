---
status: accepted
topic: research
created: 2026-10-21
updated: 2026-10-21
---
# The Streamlit app: one split of E1's harness, one paid fill per confirmed click

## Context

The Working Prototype milestone is a Streamlit page over the pipeline, and the unit's scope says the
app calls the stages and adds no pipeline logic. E1's harness already fills and scores one holdout
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
* **Orchestration**: `bko.app.steps` composes the stages' public APIs; the copies of the UCI load
  sequence and of the evaluate command's BKO wiring are filed in `tasks/CLEANUP.md`.
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
* **Verdicts**: the page shows one split's measures and no verdict.
* **`SPEND_CEILING`**: 0.25 USD per fill.
* **Run folders**: each fill keeps its run folder (ADR `tooling/run-folders`).

## Rationale

One action keeps the fill's private steps private: a second step would need the harness's fill
public and a runner that reuses it. A replica fill would add many paid calls and minutes per BKO
click, and no criterion reads its fidelity. Replay lets the demo run free and repeatably from a
recorded log, with the refusals the evaluate command already applies. A row per click would flood
the register with readings no criterion uses. Extracting the commands' load sequence and wiring
first would edit four commands owned by others and collide with `proposal-guidance-ablation`.
The completeness request is another request, and scoring it would pay twice. Controls add no rows,
so they cannot fill. A confirmation keyed to a run counter cannot carry over to a second click; a
credential pre-check would itself call the API, and a process-wide spend ledger guards a server
this app never runs. A config file is a second home for the same flags. E1's 25 USD ceiling bounds
a whole experiment; one fill makes a few calls.

## Consequences

**Gained**: every path that spends nothing runs in the gate; one paid transaction per click; a
free, repeatable demo from a recorded log.
**Accepted**: another copy of the UCI load sequence and of `_guided`'s wiring; no live rejected-row
counter; no cancel of an in-flight call, and a closed tab or Ctrl-C waits for the running action,
CTGAN and TVAE fits included; the page's numbers never equal E1's mean over its holdouts; clicks
queue during an action; an AppTest that reaches the SDK fails on what AppTest's `run` raises, with
the guard's message on the script thread's exception; DNS rebinding to `127.0.0.1:8501` can still
drive fills, each bounded by the ceiling and a confirmation.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Two steps, fill then evaluate | Needs the harness's fill public and a runner that reuses it | 2026-10-21 |
| A replica fill per BKO click | Many extra paid calls and minutes per click; no criterion reads fidelity | 2026-10-21 |
| Live answers only | A demo would spend on every showing | 2026-10-21 |
| A register row per click | Interactive readings answer no criterion | 2026-10-21 |
| Extracting the commands' load sequence and BKO wiring first | Edits four commands owned by others and collides with `proposal-guidance-ablation` | 2026-10-21 |
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

## Enforcement
- Owner: `src/bko/app/`
- Pinned by: `tests/test_repo_gate.py`, `tests/test_app_page.py`, `tests/test_app_imports.py`

## Reversal Conditions
The app is served beyond loopback (then authentication and a spend ledger); `GuidedArm` gains a
per-batch hook; a criterion reads replica fidelity; a second dataset gains a fill path.

## Related
- [E1 evaluation protocol](e1-evaluation-protocol.md)
- [Call-log replay](call-log-replay.md)
- [Record generation design](record-generation-design.md)
- [Run folders](../tooling/run-folders.md)
- [uv, ruff and ty](../tooling/uv-ruff-ty.md)
- [Paid API request guard](../tooling/paid-api-request-guard.md)
