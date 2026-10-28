---
id: evaluate-e1-second-configuration
title: E1 under the chosen level, bounds and effort
status: ready
depends_on: --
blocks: evaluate-dose-response, evaluate-data-scarcity-curve, evaluate-generator-comparison, generate-bkb-curation
parallel_safe: false
owner: Joe
milestone: final-report
paid: true
adrs: --
---

# E1 under the chosen level, bounds and effort

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

Three changes each make a new configuration of E1: the level G3 chose, the
training-split bounds in the prompt, and the effort the sweep chose. E1 is judged on its latest
configuration (ADR
[`research/e1-evaluation-protocol`](../docs/adr/research/e1-evaluation-protocol.md)), so the three
land first and E1 runs once over all of them, and every power experiment that follows runs on the
configuration that will be judged.

## Scope

**In.**

- `SETTINGS.guidance` set to `summary`, the level the decision rule of
  [`docs/EVIDENCE.md`](../docs/EVIDENCE.md#g3-guidance-level) G3 chooses.
- `SETTINGS.effort` stays `medium`, the effort G4 chose
  ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#g4-effort-at-summary)).
- E1's Configuration in `docs/EVIDENCE.md` restated for the new settings, with the spend and
  duration estimate derived as [`experiments-index.md`](experiments-index.md#spend) says, from
  G4's chosen-effort run's tokens per record asked.
- The `live` generation test, then a paid completeness request and its closure as a C1 Runs row
  (C1's configuration rules that a change to what a call sends supersedes the judged run), stopping
  to report if validity falls below `E1.validity`.
- A free rehearsal on all holdouts with canned replies, then one paid `score` run under
  `E1.spend_ceiling` through the paid-run protocol's launch and stop rules (`docs/DEVELOPMENT.md`, Evaluation); the call log copied
  off the runner before replay; then `score --replay` and `compare`.
- A new judged Runs row with its spend, Result and Read against in E1's entry; `e1-3` keeps its
  row.

**Out.** Any other code change (one that a run needs stops the run and goes through its own PR).

## Acceptance

- Every number regenerates from a committed command and is recorded in `docs/EVIDENCE.md`, with
  the run's spend in its Runs field and the replay byte for byte.
- `make check`.
