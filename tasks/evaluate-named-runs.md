---
id: evaluate-named-runs
title: Named runs beside E1
status: ready
depends_on: --
blocks: proposal-guidance-ablation, generate-effort-sweep, generate-bkb-sampler, evaluate-dose-response, evaluate-data-scarcity-curve, evaluate-generator-comparison, evaluate-synthea-downstream
parallel_safe: false
owner: Joe
milestone: final-report
paid: false
adrs: --
---

# Named runs beside E1

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

`bko.evaluate score` runs `E1` at `SETTINGS`, and no flag changes a value of either (ADR
[`research/e1-evaluation-protocol`](../docs/adr/research/e1-evaluation-protocol.md) rules out
flags for the readings or the holdout count); `compare` refuses a scores file that does not cover
E1's holdouts. Every variant this set runs, a guidance level, an effort, a model, a fill target, a
training fraction, an extra arm, a Synthea frame, needs a configuration fixed in code and
selected by name, so that each run still comes from one committed command and replays at its
commit.

## Scope

**In.**

- `bko.evaluate.experiment` holds a mapping from run name to the `Experiment` and the `Settings`
  it runs at; `e1` names `E1` at `SETTINGS` and is the default. A named run lists its arms, its
  candidate and its comparators, its holdout seeds, its fill target, its ceiling and its criteria,
  which may be none; a run with no criterion is reported with no verdict. A variant whose
  candidate is one BKO arm lists the baselines and that arm, so it pays one arm and its result is
  read beside the judged run's at the same holdouts.
- `score` and `compare` take the run name as a positional, so a value still changes only in code;
  `--log`, `--replay` and `--resume` are unchanged, and a named run's log replays at its commit as
  E1's does. Run folders name the run.
- `compare` on a named run reads every measure `E1`'s `compare` reads, over the run's own holdouts,
  and judges only the criteria the run names.
- ADR `research/e1-evaluation-protocol` gains the named-run rule and why it does not reopen the
  ruled-out flags; [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) gains the command.

**Out.** Any new run configuration (each unit that needs one adds its own); a paid run.

## Acceptance

- `score e1` and `score` write the same bytes on canned replies, pinned by a test, and E1's
  judged scores regenerate through `score e1 --replay` at a commit where the judged log replays.
- A named test run on a fixture split scores and compares with no verdict, pinned by a test.
- `make check`.
