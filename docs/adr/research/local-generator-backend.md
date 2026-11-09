---
status: accepted
topic: research
created: 2026-10-29
updated: 2026-11-09
---
# On-device generators: a llama.cpp server on the loopback interface, weights fixed by digest

## Context

The generator comparison ([G6](../../EVIDENCE.md#g6-generator-model)) varies the generator model alone,
beside Haiku 5.5 and Sonnet 5.5, with three open models run on this machine: Gemma 4 26B-A4B, Nemotron 3 Nano 30B-A3B and Olmo 3 7B
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
  `bko.generate.local.LOCAL_URL` with the model id as its alias, `SLOTS` parallel slots sharing a
  `CONTEXT`-token context, and `--jinja --chat-template-kwargs '{"enable_thinking":false}'`
  (`serve_command`). A `-thinking` id serves with `--jinja --reasoning on --reasoning-format deepseek`
  in place of the chat-template kwargs. The payload's `chat_template_kwargs.enable_thinking`
  overrides the server's `--reasoning` default, so `--reasoning-format deepseek` is the flag that
  matters: it parses the reasoning out of the reply's content into `reasoning_content`.
  Olmo 3 7B Instruct is served with llama.cpp's built-in `--chat-template chatml` instead: its
  file's template pipes an undefined `tools` into `tojson`, which llama.cpp's template parser
  refuses at start, and for a system and a user turn with no tools that template renders ChatML.
  Olmo 3 7B Instruct has no thinking mode.
- **Thinking variants.** `gemma-4-26b-a4b-q8_0-thinking` and
  `nemotron-3-nano-30b-a3b-q8_0-thinking` are `LOCAL_MODELS` entries whose `LocalModel.thinking`
  is true and whose `LocalModel.weights` names their base model, so a variant shares the base's
  repository, file, byte count, sha256 and path under `data/models/`, and fetching it fetches
  nothing when the base's file is present. Olmo 3 7B Instruct has no variant. A base id's argv and
  payload are independent of its variant.
- **Send.** `bko.generate.local.local_send(model)` posts each call to `/v1/chat/completions` with its
  system and user turns, its `max_tokens`, `temperature` 1.0 and `top_p` 1.0 (the Messages API's
  default temperature), the call's `record_schema` as a strict `json_schema` response format, and
  `chat_template_kwargs.enable_thinking` false, or true for a `-thinking` id. It uses stdlib
  `http.client` in `asyncio.to_thread`, so no proxy setting and no redirect applies, and refuses
  any address but a bare `http://127.0.0.1:<port>` (`loopback`). A cancelled call, by the idle
  rule or an interrupt, posts nothing more, even once a check it waited on finishes, and shuts its
  socket, so the thread returns and the server stops generating for it. A failed connection, a
  broken response or a status of 500 or above raises `GenerationError(retryable=True)`, as the
  API's connection errors do; another status, or a body that is not a JSON object, raises one that
  is not. `bko.generate.chat.reply_of` reads the reply's text with `bko.generate.messages.text_of`,
  the reader replay uses (ADR `research/call-log-replay`): the first choice's `message.content`,
  reasoning ignored.
- **Server check.** The first call reads the server's `/props` and refuses a server whose alias is
  not the run's model or whose weights file name is not its pin's, before any call is posted, so a
  server under a base id's alias is refused for its variant and the reverse. For a `-thinking` id
  the check also sends one short generation, the probe's call at `max_tokens` 64
  (`PREFLIGHT_TOKENS`), and refuses a reply with no reasoning, since build 11960's `/props` reports
  no reasoning setting. The refusal names the server's build, the served path, the pinned sha256,
  `preflight_reasoning_chars=0` and the pre-flight reply's stop reason and content length; it is
  kept, so each later call of the run fails with it without another pre-flight, while a retryable
  failure leaves the next call to check again. A checked server logs one
  `local server model=%s thinking=%s path=%s build=%s slots=%s sha256=%s` line: the model, its
  pin's thinking setting (whether the run asks the model to think, not a reading of the server),
  the served file, the server's build and the file's pinned digest; a `-thinking` id's line ends
  ` preflight_reasoning_chars=%d`, the pre-flight reply's reasoning length. Measured 2026-11-09 on
  llama-server 0.8.0 build 11960 serving the Gemma `-thinking` alias, by reading its `/props` and
  running `python -m bko.generate.local probe gemma-4-26b-a4b-q8_0-thinking` under each flag set:
  `/props` reads the same under `--reasoning on`, `--reasoning off`, the reasoning flags dropped,
  and `--reasoning-format none`, apart from its per-start `media_marker`. The first three each
  reply with reasoning to the thinking payload; `--reasoning-format none` leaves the thoughts
  unparsed in the reply's content with no `reasoning_content`, and that server is the one the
  pre-flight refuses.
- **Calls in flight.** A local run keeps at most `SLOTS` calls in flight, whatever `--in-flight`
  asks (`bko.evaluate.scoring`), since a call past the slots would queue inside the server while its
  idle limit ran.
- **Reply.** `finish_reason` `stop` reads as `end_turn` and `length` as `max_tokens`; prompt and
  completion tokens are input and output tokens, with no cache.
- **Selection.** `bko.evaluate.context.Context.sender(model)` builds `local_send(model)` for a model
  in `bko.generate.weights.LOCAL_MODELS` and the API send otherwise, for a run's calls and a resume's
  live calls. A rehearsal answers from canned replies whatever the model.
- **Price.** Each local model has a zero `Price` in `bko.generate.spend.PRICES`.
- **Weights.** Q8_0 GGUF files, each fixed in `LOCAL_MODELS` by repository, file, byte count and
  sha256 as read from the repository's file listing on 2026-10-29: the publisher's own GGUF
  repository, else `ggml-org`'s, else `unsloth`'s (`unsloth/Olmo-3-7B-Instruct-GGUF`, since Allen AI
  and `ggml-org` publish no Olmo 3 7B Instruct GGUF). `python -m bko.generate.local fetch` writes
  them to `data/models/<model>/<sha256[:16]>/<file>` (`bko.generate.weights`), verifying size and
  sha256 before use and removing a download that fails them.
- **Probe.** `python -m bko.generate.local probe <model>` waits for the server's `/health`, sends
  one call for four records of `<55/female/1`, and logs what decodes, how long it took and the
  reply's `reasoning_content` length as `reasoning_chars=`; a probe that decodes no record exits 2,
  so a reply whose content holds unparsed think tags exits 2. A server without reasoning fails the
  pre-flight first, so the probe's own no-reasoning `WARNING` fires only when the pre-flight
  reasoned and the full call did not.
- **Per-call reasoning.** Each live call of a `-thinking` id logs one `reasoning` line: its
  request id, stop reason, the reply's `reasoning_content` length in characters, its reasoning
  tokens, its content length in characters and its output tokens. The reasoning tokens are a
  re-tokenization of `reasoning_content` by the server's `/tokenize`, not a count the server kept
  while generating, since build 11960's reply `usage` holds no reasoning count. `/tokenize` waits at
  most `bko.generate.reasoning.TOKENIZE_SECONDS`; a failed one logs the tokens as `unknown` and
  keeps the reply, and the run's first failure logs one `tokenize failed` `WARNING` naming the
  status or exception. The request id and stop reason are written with whitespace and unprintable
  characters escaped and cut at `FIELD_CHARS`, so a plain id reads as sent. A reply cut at
  `max_tokens` with reasoning and no content logs the line as a `WARNING` ending
  `reason=truncated_while_reasoning`, so thinking that ate the budget reads apart from an answer
  cut short. The line comes from `local_send`, so a replay logs none.
- **Thinking.** G6's runs turn thinking off, the project's choice for the generator comparison;
  the `-thinking` ids measure thinking on, as the named runs `g6-gemmathink` and
  `g6-nemotronthink` on holdouts 0 to 4. G6's thinking-off runs on the same holdouts are the
  comparison, so no thinking-off control run is made.
- **Compute.** A local run's cost is read in tokens and in FLOPs, 2 x active parameters x tokens
  computed, never in wall time. The tokens computed are the prompt tokens less those the server's
  usage reports as cached, plus the generated tokens, reasoning included. Each pinned GGUF header
  holds the experts used and their total in `<arch>.expert_used_count` and `<arch>.expert_count`
  (read from the pinned files: `gemma4` 8 of 128, `nemotron_h_moe` 6 of 128). The parameter
  figures are not among the header's metadata fields, and no field checked here derives them:

  | Model | Active parameters / total | Experts used |
  |-------|---------------------------|--------------|
  | Gemma 4 26B-A4B | 3.823B / 25.233B | 8 of 128 |
  | Nemotron 3 Nano 30B-A3B | 3.580B / 31.578B | 6 of 128 |
  | Olmo 3 7B Instruct | 7.298B, dense | n/a |

## Rationale

`llama-server` serves GGUF weights through an OpenAI-compatible endpoint that enforces a JSON
schema with a grammar, so every reply decodes to the call's schema as the API's structured output
does. Parallel slots let the flight keep calls in flight as it does against the API. Q8_0 keeps the
weights close to the published models at a smaller size and run time than BF16. The API's default
temperature, fixed for every local model, keeps their outputs comparable with each other and with
the API models'. Stdlib `http.client` in a thread needs no new dependency; holding the connection
lets a cancelled call hang up. The server check ties each run's log to the file actually served,
and a `-thinking` run to a server that parses its reasoning out.

## Consequences

**Gained**: G6 can read three open models through the same harness, logs and replay as the API
models, offline.
**Accepted**: thinking off separates G6's local runs from the API models, which think adaptively,
so a difference between them mixes the model with thinking, and the `-thinking` runs measure what
thinking on costs on five holdouts alone; a local run takes hours, one model at a
time; the weights take tens of gigabytes under `data/models/`; `llama-server`'s build is logged per
run (the `local server` line) rather than locked; a `-thinking` run spends one pre-flight
generation of at most 64 tokens before its first call, and one `/tokenize` request per call.

**Measured 2026-10-29** with Homebrew `llama-server` 0.8.0 (build `b11960-8642541b2`) on an Apple M5
Max: `fetch` verified each file against its entry, and `probe` against each served model, every
`local server` line reading build `b11960-8642541b2` and four slots, decoded four of four records
of `<55/female/1` ending `end_turn`: Olmo 3 7B Instruct 567 input and 466 output tokens in 9.3 s,
Gemma 4 26B-A4B 584 and 554 in 10.8 s, Nemotron 3 Nano 30B-A3B 583 and 424 in 6.4 s (the `probe`
lines of `python -m bko.generate.local probe <model>`).

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Ollama | Another layer over llama.cpp, with less control of slots and of the exact weights | 2026-10-29 |
| MLX | Slower for Gemma 4 on Apple Silicon | 2026-10-29 |
| Each model's own sampling defaults | Not comparable across models | 2026-10-29 |
| BF16 weights | Larger and slower to run than Q8_0 | 2026-10-29 |
| Changing `Context.send`'s signature to take the model | Rewrites every test's fake send | 2026-10-29 |
| Olmo's file template with `tools` set to null | llama.cpp's template parser still refuses it at start (measured 2026-10-29, `llama-server` 0.8.0 build 11960) | 2026-10-29 |
| A `Settings.thinking` field | A model id already selects the send, the price and the weights; a variant id keeps thinking out of every API run's settings | 2026-11-01 |
| A `NamedRun` field for thinking | The same: the run's model id carries it | 2026-11-01 |
| Wall-time cost | The user's decision: wall time depends on the machine, so compute cost is tokens and FLOPs | 2026-11-01 |
| Thinking-off control runs on holdouts 0 to 4 | `g6-2` and `g6-3` hold that result | 2026-11-01 |
| Reading the reasoning setting from `/props` | Build 11960 reports none: its `/props` reads the same under `--reasoning on`, `--reasoning off` and `--reasoning-format none` (measured 2026-11-09) | 2026-11-09 |

## Enforcement

- Owner: `src/bko/generate/`
- Pinned by: `tests/test_generate_local.py`, `tests/test_generate_reasoning.py`,
  `tests/test_generate_weights.py`,
  `tests/test_evaluate_context.py`, `tests/test_evaluate_experiment.py`,
  `tests/test_generate_spend.py`

## Reversal Conditions

**A model's entry is replaced if its repository no longer serves that digest**; this record and
`LOCAL_MODELS` change together.

**The backend changes if `llama-server` drops `json_schema` response formats.**

**Thinking is revisited for the thinking models if a measurement shows it changes validity or
utility at a run time the project can afford.**

## Related

- [Record generation](record-generation-design.md)
- [Concurrent call waves](concurrent-call-waves.md)
- [The E1 evaluation protocol](e1-evaluation-protocol.md)
