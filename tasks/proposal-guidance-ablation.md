---
id: proposal-guidance-ablation
title: Guidance-level ablation of record generation
status: blocked
depends_on: evaluate-named-runs, evaluate-run-protocol
blocks: evaluate-e1-second-configuration
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: --
---

# Guidance-level ablation of record generation

The set's graph is [`proposal-index.md`](proposal-index.md); its dependencies are in
[`experiments-index.md`](experiments-index.md).

## Why this unit exists

The team asks how much guidance and ontology structure record generation needs. `bko.generate`
builds six nested guidance levels, and this unit measures what each one adds. At level `ontology`,
BKO records match their cell's sex, target and age band but not its field distributions: over the
records `e1-3`'s call log decodes, against the real records of each requested cell, `cp` 4 at 0.28
against 0.50, `restecg` 1 at 0.26 and 2 at 0.13 against 0.01 and 0.49, `thal` 6 at 0.25 against
0.06, `ca` 0 at 0.42 against 0.61, `chol`'s standard deviation narrower in every cell, `thalach`
14 to 24 bpm lower in the 55-and-over cells, and `trestbps` on a multiple of 10 in 14.2% of records
against 55.2% of real ones (from `env -u ANTHROPIC_API_KEY uv run --no-sync python -I
scratchpad/quality.py scratchpad/e1-3.jsonl` on the runner's machine, gitignored, sha256
`aa21abfedf484d8f23677ddfe5805bf055906aaf40aef928670403a06ec834fb`); E1's Result reads the effect
as the lowest replica fidelity of every arm. The prompt at level `ontology` gives no frequencies or
spreads; `profile` and `summary` add them, so which level E1 runs is this unit's result.

## Scope

**In.**

- One named run per level of `bko.generate.messages.Level` other than `ontology`
  (`evaluate-named-runs`), each with the baselines and the `bko` arm at `SETTINGS` otherwise
  unchanged, Haiku 5.5 settings and the replica fill included, so the level is the only difference
  from E1's judged run, which is level `ontology` itself. Each run under the protocol of
  `evaluate-run-protocol`, with a spend estimate before it of one arm's share of the judged run's
  spend ([`experiments-index.md`](experiments-index.md#spend)).
- Read `cell` and `coherent` as the no-ontology controls: among the components every level sends
  (ADR [`research/record-generation-design`](../docs/adr/research/record-generation-design.md)),
  `cell` adds none, `coherent` adds the coherence rules, and neither sends a finding, so `cell` is
  the LLM with no BKO guidance.
- Per level: validity, dropped candidates, spend, E1's measures, and the replica fidelity (quality,
  column shapes, column pair trends) and replica utility, paired over the holdouts.
- The decision, stated here before the runs: E1 moves to the level whose replica column shapes
  and column pair trends are both closest to the training rows, the lower level when the two
  measures disagree, or stays at `ontology` when no level beats it on both; the record states
  which measure decided, and [`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md)
  sets `SETTINGS` to it.
- Record the runs in a new `docs/EVIDENCE.md` entry.

**Out.** Changing E1's configuration.

## Acceptance

- Each run's spend and folder are in its `docs/EVIDENCE.md` Runs field, and each replays byte for
  byte.
- The levels' measures come from `score <run>` and `compare <run>`, and the chosen level is
  recorded with the measure that chose it.
- `make check`.
