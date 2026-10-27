---
status: accepted
topic: research
created: 2026-10-18
updated: 2026-10-26
---
# Record generation: one request per cell at a chosen guidance level, every asked record handed on

## Context

RESEARCH, Generate and validate, asks an LLM for candidate records that fill the detected gaps,
guided by the ontology relationships around each one
([`docs/RESEARCH.md`](../../RESEARCH.md#generate-and-validate)). E1 tests whether such records fit
the ontology and add predictive power for the underrepresented cells. It does not test whether an
LLM can emit valid JSON, so a record lost to a format slip measures nothing E1 asks about.

The API facts this record relies on, read 2026-10-18 from
`platform.claude.com/docs/en/build-with-claude/structured-outputs.md`:

- `output_config.format` needs no beta header.
- `enum` takes numbers.
- Every object must set `additionalProperties` to `false`.
- Numeric bounds (`minimum`, `maximum`, `multipleOf`) are unsupported.
- `minItems` takes 0 or 1 only.
- On a `max_tokens` or `refusal` stop the output may not match the schema, and it is billed.

On 2026-10-19 that page lists `claude-haiku-5-5` among its supported models, and
`platform.claude.com/docs/en/build-with-claude/effort.md` lists it for `effort`, with `medium` its
default.

SDK 1.13.0's `Messages.create` takes `output_config` (`effort`, and `format` with `schema` and
`type: "json_schema"`) and takes no `temperature`, `top_p` or `top_k`. The `live` test
`tests/test_generate_live.py` checks the request against the API.

The prices `bko.generate.spend.PRICES` holds were read 2026-10-19 from
`platform.claude.com/docs/en/about-claude/pricing.md`, Model pricing, row Claude Haiku 5.5 for
prompts up to 100,000 tokens: input $0.10, 5-minute cache writes $0.125, cache hits $0.01 and output
$0.50, each per million tokens. A call's prompt runs to a few thousand tokens, so the higher tier
never applies. The API reports a call's tokens and no price.

A record's fields, codes and plausible ranges already have homes: `bko.datasets.uci` (`CODEBOOK`,
`DESCRIPTIONS`, `FRAME_COLUMNS`), `bko.datasets.cells` (`SEXES`, `LABEL_MEANINGS`) and
`bko.ontology.mapping.RANGES`.

## Decision Drivers

* **Must keep the SDK out of every module but one**, so every other module and test runs on a
  fake and no test spends money by accident.
* **Must spend paid records on what E1 measures**, not on shape or code slips.
* **Must make the guidance an ablation axis**, so a run can say which component helps.
* **Must never show the model a training record**, since a copy fails E1's privacy share.
* **Should send no parameter the model rejects.**

## Decision

We will **make one Messages call per chunk of a cell's request, through a repo-owned `Send` seam,
with the record's shape fixed by structured outputs and the guidance chosen from a six-level
ladder.**

- **The seam.** `Send = Callable[[Call], Reply]` in `bko.generate.messages`. `Call` and `Reply`
  are repo-owned values. `bko.generate.api` is the only module that imports `anthropic`: it reads
  an SDK message into a `Reply`, and turns an SDK failure into `GenerationError`, built from the
  error's type, status and request id only and raised `from None`, because an error body can echo
  the credential. A 400 also carries the body's `error.message`, with the client's API key and
  auth token replaced by `[redacted]`, so a rejected schema or parameter names its reason.
- **The schema.** As much as structured outputs can express:
  - each coded field is `integer` with an `enum` of its codebook;
  - the cell's `sex` and `target` are each an `enum` of one value;
  - the whole-number fields are `integer`, and `oldpeak` is `number`;
  - every object is closed and requires all of its properties.

  Plausibility ranges and `oldpeak`'s one decimal stay as prompt instructions.
- **The call.** Model `claude-haiku-5-5`; effort `medium`, set explicitly; `max_tokens` 16000;
  non-streaming; the SDK client's retries and timeout from `bko.generate.api` (`MAX_RETRIES`,
  `TIMEOUT`), which enter no logged request. No `temperature`, `top_p` or `top_k` is sent, and no
  server-side fallback is set. `Settings` holds these values, with the chunk size `per_call` and the
  guidance level.
- **The ladder.** `Level` holds six levels, each adding one component to the one before it:
  - `cell`: the bare cell description;
  - `coherent`: a sentence asking for clinically coherent records;
  - `findings`: the ontology gap findings for the cell;
  - `ontology`: their `is_a` chains, the default;
  - `profile`: the reference group's concept shares;
  - `summary`: the reference group's numeric spreads.
- **The findings.** `bko.generate.guidance.guide` reads a cell's findings from its gaps under the
  run's definition (ADR [`research/bko-formal-gap-definition`](bko-formal-gap-definition.md)).
  A cell with no gap is guided by the statements its own records ground, and a cell with neither
  gets no findings. Each statement maps to the mapping rows that can ground it: those whose
  concept is its antecedent or descends from it, and that admit the cell's label. Statements
  reached by the same rows form one finding, named by those rows' field states, so a record that
  meets one of a finding's states grounds every statement listed under it. A statement no
  compatible row reaches is dropped with a WARNING. A statement reached by a compatible row that
  reads only the label is skipped without a warning: the schema pins the cell's label, so every record
  already meets that row and grounds the statement. The wording follows the basis:
  - absent, for per-subgroup and population-level gaps: no training record of the group shows the
    finding;
  - rare, for frequency-aware gaps: the finding is rarer than among comparable patients;
  - shown, for a cell with no gap: the group's records show the finding.
- **The profile and summary.** Both read the reference group of ADR
  [`research/frequency-aware-gap-score`](frequency-aware-gap-score.md): the training records
  outside the cell that share its label. The profile gives each concept that a label-compatible
  row maps, and that the observed cells count, its share of those records, which is the dataset
  fragment's prior over them. A concept that a row reading only the label maps is left out, since
  every reference record shows it. The
  summary gives each numeric field's quartiles over the same records. A cell with no reference
  record gets neither, with a WARNING.
- **What every level sends:**
  - each field's meaning, codes and unit;
  - each numeric field's plausible range from `RANGES`, with age clipped to the cell's band;
  - the cell and its training record count;
  - an instruction to invent new patients.

  No level sends a record.
- **The decode.** `bko.generate.decode` reads a reply only when the model stopped on `end_turn`.
  Its items pass as decoded, in order, unfiltered and unrepaired, followed by one `Undecoded`
  placeholder per asked record short: `max(0, asked - decoded)` of them. A refusal, a truncation,
  any other stop, missing text, text that is not JSON, or a root without a `records` list decodes
  to no items, so every asked record becomes a placeholder carrying that reason. `Undecoded` is
  frozen and is not a `Mapping`, so validation rejects it under `fields`. Decoding never raises on
  a reply's content. A call that returns no response raises `GenerationError` and owes no
  placeholders.
- **The request.** `bko.generate.calls.generate` takes the quota as `Mapping[Cell, int]` and each
  cell's guidance, not `bko.baselines.request.Request`, so generation imports no downstream stage.
  It refuses a quota entry that is not an `int` of at least 1, or names a cell with no guidance,
  before any call. It asks each cell's rows in chunks of `per_call`, cells in quota order
  (`bko.generate.calls.chunks(quota, per_call)`, the plan the candidates header records), so a
  run makes `sum(ceil(rows / per_call))` calls; it never re-asks a cell or tops up.
- **The log.** `bko.generate.log.CallLog` writes an append-only JSONL file at a required path,
  created exclusively (`O_EXCL`) with mode 0o600 after the settings model's price resolves, so an
  unpriced model or an existing log is refused before any call and only the file's owner can read
  its prompts and responses; the SDK version it records is read before the file is created too.
  Its lines, each flushed as written: one `run` line
  (schema 2, settings, price, SDK version, `replay`: the sha256 of the call log a replay
  answers from, or null, `resume`: the sha256 and answered calls of the stopped log a resume
  continues, or null, and `rehearsal`: true when canned replies answer every call, as
  `score --rehearse` makes them, else false), then per call a `request` line before it is sent and
  a `reply` or `error` line after, then one `end` line with the status (`complete`, `interrupted`
  for `KeyboardInterrupt`, `failed` otherwise), the `run` line's `replay`, `resume` and
  `rehearsal` again, `stop` (the stop rule that ended a failed run: `ceiling`, `projection` or
  `idle`, read from the exception's `stop_rule`, else null), and the run's totals, `live_usd` (the
  spend of calls past the answered ones, a rehearsal's billed spend included) among them. The
  schema stays 2: a replay reads only the `run` line's `kind` and `schema`, and a resume reads
  `replay`, `resume`, `rehearsal` and `stop` as absent when a log predates them. A `reply` line
  holds the whole response and `attempts`, the HTTP requests the SDK made for the call
  (`bko.generate.api.Attempts` counts them). An `error` line holds `attempts` too, null when
  nothing counted them. The `end` line's totals add `retried`, the calls that took more than one
  attempt. Anything raised after a `request` line and before its `reply` line (no
  response, a `KeyboardInterrupt`, a failure reading the reply) writes an `error` line instead,
  so no `request` line is left unpaired; it holds a `GenerationError`'s text, or any other
  exception's type name only, since its text may hold the credential. `ask` and `generate` take their
  settings from the log and refuse others before any line, so the `run` line names the settings
  every call was made and priced under. A call's spend is its usage times `PRICES[settings.model]`, each token kind at
  its own rate; output tokens already count thinking.
- **The command.** `python -m bko.generate` fills E1's count-based gap request on the default
  split: each cell holding fewer than `UNDERREPRESENTED_BELOW` training records is asked for its
  shortfall. With `--request completeness` it fills the completeness request of
  `bko.gaps.definitions.completeness_request` instead, `UNDERREPRESENTED_BELOW` rows for each cell
  holding a per-subgroup gap, and refuses any guidance definition but `per-subgroup`, the one that
  request reads (ADR `research/gap-closure-design`). It rebuilds that request from `bko.gaps.detect.count_gaps` over
  `bko.datasets.cells.cell_counts` rather than importing `bko.evaluate`, and a test holds it equal
  to `bko.evaluate.experiment.gap_request(..., E1, seed=0).quota`. It writes the candidates
  header line first, inside the call log's `with`, then each call's candidates to stdout as the
  call returns, through `generate`'s `on_batch`, so a failed or interrupted run keeps every paid
  batch that returned, and a run that fails before its first call leaves a header and no lines.
  Only the command imports `bko.gaps.scope`, the loader of the UCI split and the releases it
  shares with `python -m bko.gaps`, `python -m bko.fusion` and `bko.app`.

## Rationale

A callable seam follows `UrlOpener`/`open_https` in `bko.datasets.fetch`. A Protocol mirroring
`client.messages` would be the first Protocol in `src/`, and a real client on a mock transport is
refused by the paid-API guard, which patches the request sink.

Structured outputs move the shape and code checks from validation into decoding. Without them the
`code:` checks stay live, but each failure costs a paid record. Prompt-only JSON has no prefill to
lean on, and one slip turns a whole call's records into placeholders.

Effort is set explicitly so a change of the API's default does not move a run. A server-side
fallback would serve some calls on another model, chosen by refusal, so the model behind a record
would depend on its content.

One JSONL line per event, flushed as written, keeps every paid call's request and response
through a failure or a killed process; a single JSON written at the end loses them. Creating it
exclusively keeps a rerun from overwriting a paid log, and resolving the price first keeps an
unpriced model from making a call whose spend reads as zero.

Six levels separate what the ontology adds from what any careful prompt adds: `coherent` controls
for a plain request for clinical sense, and `findings` and `ontology` differ by exactly the `is_a`
chains. A single fixed prompt gives no ablation axis.

The ranges come from `RANGES` rather than a training split's minimum and maximum, because the
prompt states what is possible for a patient, not what one split happens to hold.

A placeholder per missing record keeps validity's denominator at what was asked for, so a refusal
or a short array lowers validity rather than vanishing. `None` cannot be the placeholder, since a
reply can decode to `null`. A truncated output is not salvaged, because its last records may be
cut mid-value and the stop already says the call failed.

Haiku 5.5 is the generator because E1's replica configuration asks for 77400 records a run, of which
the gap request asks 6000, and Opus 5.5's input and output rates are 40 times Haiku 5.5's; E1's
Configuration in `docs/EVIDENCE.md` records the run's estimate. A stopped run resumes only by
paying its stopped call again, so the client retries for minutes, where the SDK's default of 2 retries backs off for about 1.5 s. A client
timeout other than the SDK's default skips the SDK's refusal of a non-streaming call above 21,333
`max_tokens`; `max_tokens` is 16000.

## Consequences

**Gained:**
- No module or test outside `bko.generate.api` touches the SDK.
- Paid records are spent on values, not on shape.
- A run can be repeated at each guidance level.

**Accepted:**
- On a decoded record only `range:<field>` and `precision:oldpeak` can fail validation. `fields`,
  `number:`, `code:` and whole-field `precision:` fail only on placeholders, or on a non-finite
  value the API should not emit. Validity then measures range adherence.
- A plausible value outside a training split's range passes validation, and E1's runner drops it
  (ADR [`research/e1-evaluation-protocol`](e1-evaluation-protocol.md)).
- The Messages API takes no seed, so generated rows cannot be drawn again. A paid run's call log is
  their record, and replay answers a rerun's calls from it (ADR
  [`research/call-log-replay`](call-log-replay.md)).
- The `summary` level's medians pull values toward real records, which presses on E1's privacy
  distance. The spreads are per-field marginals, never a joint row. No floor applies to the
  reference group's size, so a reference group of one record makes them equal that record's
  values; on the UCI split the reference group is the rest of the cell's label group.
- A model may reproduce a public Cleveland record. E1 counts each arm's rows equal to a test
  record and scores every arm on the test split without them (ADR
  [`research/e1-evaluation-protocol`](e1-evaluation-protocol.md)).

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A Protocol mirroring `client.messages` | `src/` has none, and a callable is the repository's seam (Rationale) | 2026-10-18 |
| A real client on a mock transport | The paid-API guard refuses it at the request sink | 2026-10-18 |
| A schema of shape only | The `code:` checks stay live and each failure wastes a paid record | 2026-10-18 |
| Prompt-only JSON | No prefill, and one slip costs a whole call's records | 2026-10-18 |
| `fallbacks: "default"` | Serves some calls on another model, selected by refusal (Rationale) | 2026-10-18 |
| A model flag | One model keeps runs comparable; `Settings` holds it | 2026-10-18 |
| One fixed prompt | No ablation axis | 2026-10-18 |
| Four guidance levels | Cannot separate the `is_a` chains from the findings, or coherence from the cell | 2026-10-18 |
| The training split's minimum and maximum as the ranges | The prompt states what is possible, not what the data holds | 2026-10-18 |
| Findings from the `is_a` edges of a cell's shown concepts only | A gap's statement names what the records lack, which the shown concepts' edges never reach | 2026-10-18 |
| Findings from the whole dataset's concepts | They describe other groups, not the cell asked for | 2026-10-18 |
| The reference group of the same label and sex | Too few records in the smallest cells, such as older women with the condition | 2026-10-18 |
| The whole split as the reference group | It mixes the label the cell is pinned to with the other | 2026-10-18 |
| Few-shot rows | A copied record fails E1's privacy share | 2026-10-18 |
| `None` as the placeholder | Collides with a decoded `null` (Rationale) | 2026-10-18 |
| Salvaging truncated output | Its last records may be cut mid-value (Rationale) | 2026-10-18 |
| One JSON log written at the end of a run | A failure or a kill loses every paid call's record (Rationale) | 2026-10-18 |
| The log as a stdout redirect | Mixes with the records, and a rerun can overwrite it | 2026-10-18 |
| Scrubbing log lines for secrets, as the mht3 research repository does | `bko.generate.api` builds each error without the response body, so no line carries the credential | 2026-10-18 |
| `bko.baselines.request.Request` as `generate`'s input | Imports a downstream stage | 2026-10-18 |
| A shared `bko.core` home for the request type | Nothing else needs it yet | 2026-10-18 |
| The command importing `bko.evaluate` for E1's request | Imports a downstream stage | 2026-10-18 |
| A `--rows` water-fill as the command's request | Needs `bko.baselines`, a downstream stage | 2026-10-18 |
| `claude-opus-5-5` as the generator | Its input and output rates are 40 times Haiku 5.5's, and E1's replica configuration multiplies the records a run asks for | 2026-10-19 |
| The SDK's default 2 retries | About 1.5 s of backoff, so one brief outage stops a run whose resume repays its stopped call | 2026-10-19 |

## Enforcement

- Owner: `src/bko/generate/`
- Pinned by: `tests/test_generate_messages.py`, `tests/test_generate_api.py`,
  `tests/test_generate_prompt.py`, `tests/test_generate_imports.py`,
  `tests/test_generate_decode.py`, `tests/test_generate_contract.py`,
  `tests/test_generate_guidance.py`, `tests/test_generate_spend.py`,
  `tests/test_generate_log.py`, `tests/test_generate_calls.py`, `tests/test_generate_cli.py`,
  `tests/test_generate_candidates.py`, and `tests/test_generate_pinned_releases.py` with
  `--releases`

## Reversal Conditions

**The candidate shape changes when a generation request asks for Synthea records**; the check is a
search of `tasks/` and `src/bko/generate/`.

**The schema loosens if a judged run reports no `range:` failure and the team wants validity to
measure codes**; the check is the run's per-check counts in `docs/EVIDENCE.md`.

## Related

- [Record validation: each field's number, bound and decimals, counted per check](record-validation-design.md)
- [E1's evaluation: repeated holdouts, one classifier, separation judged on paired intervals](e1-evaluation-protocol.md)
- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
- [A frequency-aware gap: a concept credibly under four-fifths of its share in the rest of its label group](frequency-aware-gap-score.md)
- [The UCI evaluation frame: Cleveland records, a binary label and a cell-stratified split](uci-evaluation-frame.md)
- [A session-wide guard on the Anthropic SDK's request sink](../tooling/paid-api-request-guard.md)
