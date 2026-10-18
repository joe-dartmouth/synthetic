---
status: accepted
topic: research
created: 2026-10-18
updated: 2026-10-18
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

SDK 1.13.0's `Messages.create` takes `output_config` (`effort`, and `format` with `schema` and
`type: "json_schema"`) and takes no `temperature`, `top_p` or `top_k`. The `live` test
`tests/test_generate_live.py` checks the request against the API.

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
  the credential. A 400 also carries the body's `error.message`, with the client's credential
  replaced by `[redacted]`, so a rejected schema or parameter names its reason.
- **The schema.** As much as structured outputs can express:
  - each coded field is `integer` with an `enum` of its codebook;
  - the cell's `sex` and `target` are each an `enum` of one value;
  - the whole-number fields are `integer`, and `oldpeak` is `number`;
  - every object is closed and requires all of its properties.

  Plausibility ranges and `oldpeak`'s one decimal stay as prompt instructions.
- **The call.** Model `claude-opus-5-5`; effort `medium`, set explicitly; `max_tokens` 16000;
  non-streaming; the SDK's default retries. No `temperature`, `top_p` or `top_k` is sent, and no
  server-side fallback is set. `Settings` holds these values, with the chunk size `per_call` and
  the guidance level.
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

Six levels separate what the ontology adds from what any careful prompt adds: `coherent` controls
for a plain request for clinical sense, and `findings` and `ontology` differ by exactly the `is_a`
chains. A single fixed prompt gives no ablation axis.

The ranges come from `RANGES` rather than a training split's minimum and maximum, because the
prompt states what is possible for a patient, not what one split happens to hold.

A placeholder per missing record keeps validity's denominator at what was asked for, so a refusal
or a short array lowers validity rather than vanishing. `None` cannot be the placeholder, since a
reply can decode to `null`. A truncated output is not salvaged, because its last records may be
cut mid-value and the stop already says the call failed.

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
- E1's driver "Must reproduce from seeds on one machine" cannot hold for generated rows, because
  the Messages API takes no seed. `cleanup:evaluate/bko-arms-reproduce-from-log` holds the owed
  protocol change.
- A model may reproduce a public Cleveland record; `cleanup:evaluate/generated-copies-of-test-records`
  holds the owed check.

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
| Few-shot rows | A copied record fails E1's privacy share | 2026-10-18 |
| `None` as the placeholder | Collides with a decoded `null` (Rationale) | 2026-10-18 |
| Salvaging truncated output | Its last records may be cut mid-value (Rationale) | 2026-10-18 |

## Enforcement

- Owner: `src/bko/generate/`
- Pinned by: `tests/test_generate_messages.py`, `tests/test_generate_api.py`,
  `tests/test_generate_prompt.py`, `tests/test_generate_imports.py`,
  `tests/test_generate_decode.py`, `tests/test_generate_contract.py`,
  `tests/test_generate_guidance.py`

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
