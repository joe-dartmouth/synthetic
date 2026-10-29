---
id: evaluate-local-thinking-cost
title: What thinking on costs the local generators
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: --
---

# What thinking on costs the local generators

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

G6 runs Gemma 4 26B-A4B and Nemotron 3 Nano 30B-A3B with thinking off (ADR
[`research/local-generator-backend`](../docs/adr/research/local-generator-backend.md)), while Haiku
5.5 and Sonnet 5.5 think adaptively at the judged effort, so no G6 difference between them
separates the model from thinking. A thinking-on run's duration, and whether reasoning survives
`llama.cpp`'s `json_schema` grammar, are unmeasured. Olmo 3 7B Instruct has no thinking mode.

## Scope

**In.**

- `python -m bko.generate.local probe <model>` with thinking on for each thinking model, recording
  whether reasoning tokens appear and whether the records decode valid.
- One named run per model on holdouts 0 to 4 with thinking on.
- Tokens and seconds per record asked, and the projected wall time of a 100-holdout run, recorded
  as rows of G6.

**Out.** A full thinking-on run; changing G6's runs.

## Acceptance

- Each thinking model's probe and five-holdout run recorded with its measured rates and the
  projection's arithmetic.
- `make check`.
