---
id: evaluate-run-protocol
title: The paid-run protocol as committed code
status: ready
depends_on: --
blocks: proposal-guidance-ablation, generate-effort-sweep, generate-bkb-skeleton-fill, evaluate-e1-second-configuration, evaluate-synthea-downstream
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: --
---

# The paid-run protocol as committed code

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

E1's `e1-2` and `e1-3` Runs rows describe how a paid run is launched and watched: detached, kept
awake, a watcher with spend and idle stop rules, the call log copied off the runner before replay.
The launcher, the watcher and the resume script sit in gitignored `scratchpad/`, recorded only by
sha256, so no later unit can cite them, their thresholds are E1's, and `e1-2` was lost when the
machine slept on battery with a call in flight, which no script checked. Every paid unit in this
set runs under this protocol, so it is committed once.

## Scope

**In.**

- `score` gains in-process stop rules, each value held by the run's `Experiment`: a stop when the
  spend after the first holdouts, scaled to the run's holdout count, projects past the ceiling; a
  stop when no call completes within an idle limit; both logged with the measure that fired.
- A `launch` subcommand of `bko.evaluate`: refuses to start when the machine reports battery power
  (`pmset -g batt` on macOS; elsewhere it records that it could not check), starts the run
  detached and kept awake, writes stdout and stderr to the run folder's roles, and prints the
  command that watches the folder.
- `bko.app.runs.ROLES` and the Runs table of [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#runs)
  gain the roles the set's units write: a per-holdout candidates file the BKO arms generated, and a
  named run's scores and criteria.
- [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#evaluation) states the protocol once, and every
  Runs row after this cites it rather than the scripts.

**Out.** A paid run; a watcher on another machine.

## Acceptance

- The stop rules fire on a fake log in tests, each with its logged reason, and never on `e1-3`'s
  replay.
- `launch` refuses on battery in a test with a faked power reading, and a dry run writes the
  folder's roles.
- `make check`.
