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
adrs: research/local-generator-backend
---

# What thinking on costs the local generators

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

G6 runs Gemma 4 26B-A4B and Nemotron 3 Nano 30B-A3B with thinking off (ADR
[`research/local-generator-backend`](../docs/adr/research/local-generator-backend.md)), while Haiku
5.5 and Sonnet 5.5 think adaptively at the judged effort, so no G6 difference between them
separates the model from thinking. A thinking-on run's tokens and FLOPs, and whether reasoning
survives `llama.cpp`'s `json_schema` grammar, are unmeasured. Olmo 3 7B Instruct has no thinking mode.

## Scope

**In.**

- The `-thinking` model ids of `bko.generate.local` (ADR
  [`research/local-generator-backend`](../docs/adr/research/local-generator-backend.md), Thinking
  variants), with `python -m bko.generate.local probe <model>` run for each, recording its
  `reasoning_chars=` and whether the records decode valid.
- The named runs `g6-gemmathink` and `g6-nemotronthink`, one per model on holdouts 0 to 4 with
  thinking on.
- Tokens and FLOPs per record asked, and the projected tokens and FLOPs of a 100-holdout run, not
  seconds or wall time (the user's decision), recorded in G6's Runs and Result rows as thinking-on
  rows beside its thinking-off runs.

**Out.** A full thinking-on run; changing G6's runs.

## Acceptance

- Each thinking model's probe and five-holdout run recorded with its measured rates and the
  projection's arithmetic.
- `make check`.
