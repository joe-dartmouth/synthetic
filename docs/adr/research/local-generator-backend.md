---
status: accepted
topic: research
created: 2026-10-29
updated: 2026-10-29
---
# On-device generators: a llama.cpp server on the loopback interface, weights fixed by digest

## Context

G6 varies the generator model alone (`docs/EVIDENCE.md`, G6), beside Haiku 5.5 and Sonnet 5.5,
with three open models run on this machine: Gemma 4 26B-A4B, Nemotron 3 Nano 30B-A3B and Olmo 3 7B
Instruct. Every call goes through `bko.generate.flight` as a `Send`, logged, replayable and priced
(ADR `research/record-generation-design`), so an on-device model has to answer the same `Call`
with the same structured-output schema, and the run's spend gates still have to hold.

## Decision Drivers

* **Must answer a `Call` as a `Reply`**, so the flight, the call log, replay and every arm read a
  local model's calls as they read the API's.
* **Must decode to the call's `record_schema`**, the same integers, codes and closed objects.
* **Must send nothing off the machine.**
* **Must fix the weights**, so a run names the exact model it ran.
* **Must keep sampling comparable** to the API models', and fixed across the three.
* **Should add no locked dependency.**

## Decision

- **Backend.** `llama.cpp`'s `llama-server` (Homebrew), one model per process, on
  `http://127.0.0.1:8081` with `--alias <model id> -np 4 -c 81920 --jinja --chat-template-kwargs
  '{"enable_thinking":false}'`: four parallel slots sharing an 81920-token context, 20480 per call.
  Olmo 3 7B Instruct is served with llama.cpp's built-in `--chat-template chatml` instead: its
  file's template pipes an undefined `tools` into `tojson`, which llama.cpp's template parser
  refuses at start, and for a system and a user turn with no tools that template renders ChatML.
  Olmo 3 7B Instruct has no thinking mode.
- **Send.** `bko.generate.local.local_send(model)` posts each call to `/v1/chat/completions` with its
  system and user turns, its `max_tokens`, `temperature` 1.0 and `top_p` 1.0 (the Messages API's
  default temperature), the call's `record_schema` as a strict `json_schema` response format, and
  `chat_template_kwargs.enable_thinking` false. It uses stdlib `urllib` in `asyncio.to_thread`, and
  refuses an address that is not `http://127.0.0.1:`. A refused connection or a status of 500 or
  above raises `GenerationError(retryable=True)`, as the API's connection errors do; another status
  raises one that is not retryable.
- **Reply.** `finish_reason` `stop` reads as `end_turn` and `length` as `max_tokens`; prompt and
  completion tokens are input and output tokens, with no cache.
- **Selection.** `bko.evaluate.context.Context.sender(model)` builds `local_send(model)` for a model
  in `bko.generate.local.LOCAL_MODELS` and the API send otherwise, for a run's calls and a resume's
  live calls. A rehearsal answers from canned replies whatever the model.
- **Price.** Each local model has a zero `Price` in `bko.generate.spend.PRICES`. A local run's
  named run carries a spend ceiling of $1.00, so any call that reached the API would stop it.
- **Weights.** Q8_0 GGUF files, each fixed in `LOCAL_MODELS` by repository, file, byte count and
  sha256 as read from the repository's file listing on 2026-10-29: the publisher's own GGUF
  repository, else `ggml-org`'s, else `unsloth`'s (`unsloth/Olmo-3-7B-Instruct-GGUF`, since Allen AI
  and `ggml-org` publish no Olmo 3 7B Instruct GGUF). `python -m bko.generate.local fetch` writes
  them to `data/models/<model>/<sha256[:16]>/<file>`, verifying size and sha256 before use.
- **Probe.** `python -m bko.generate.local probe <model>` sends one call for four records of
  `<55/female/1` to the running server and logs what decodes; a probe that decodes no record exits 2.
- **Thinking off.** Every local run turns thinking off (`docs/EVIDENCE.md`, G6).

## Rationale

`llama-server` serves GGUF weights through an OpenAI-compatible endpoint that enforces a JSON
schema with a grammar, so every reply decodes to the call's schema as the API's structured output
does. Parallel slots let the flight keep calls in flight as it does against the API. Q8_0 keeps the
weights close to the published models at about half BF16's size and run time. The API's default
temperature, fixed for every local model, keeps their outputs comparable with each other and with
the API models'. Stdlib `urllib` in a thread needs no new dependency, and the zero price with a
$1.00 ceiling turns a misrouted paid call into a stop.

## Consequences

**Gained**: G6 can read three open models through the same harness, logs and replay as the API
models, offline.
**Accepted**: thinking off separates these models from the API models, which think adaptively, so
a G6 difference between them mixes the model with thinking; a local run takes hours, one model at a
time; the weights take about 68 GB under `data/models/`; `llama-server`'s version is recorded per
run rather than locked.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Ollama | Another layer over llama.cpp, with less control of slots and of the exact weights | 2026-10-29 |
| MLX | Slower for Gemma 4 on Apple Silicon | 2026-10-29 |
| Each model's own sampling defaults | Not comparable across models | 2026-10-29 |
| BF16 weights | About twice the run time and size | 2026-10-29 |
| Changing `Context.send`'s signature to take the model | Rewrites every test's fake send | 2026-10-29 |
| Olmo's file template with `tools` set to null | llama.cpp's template parser still refuses it at start (measured 2026-10-29, `llama-server` 0.8.0 build 11960) | 2026-10-29 |

## Enforcement

- Owner: `src/bko/generate/`
- Pinned by: `tests/test_generate_local.py`, `tests/test_evaluate_context.py`,
  `tests/test_generate_spend.py`

## Reversal Conditions

**A model's entry is replaced if its repository no longer serves that digest**; this record and
`LOCAL_MODELS` change together.

**The backend changes if `llama-server` drops `json_schema` response formats.**

**Thinking is revisited for the thinking models if a measurement shows it changes validity or
utility at a run time the project can afford** (unit `evaluate-local-thinking-cost`).

## Related

- [Record generation](record-generation-design.md)
- [Concurrent call waves](concurrent-call-waves.md)
- [The E1 evaluation protocol](e1-evaluation-protocol.md)
