---
status: accepted
topic: research
created: 2026-10-17
updated: 2026-10-17
---
# Record validation: each field's number, bound and decimals, counted per check

## Context

RESEARCH, Generate and validate, validates each candidate record before augmentation. The validity
criterion asks that at least 95% of generated records pass the model's consistency checks
([`docs/RESEARCH.md`](../../RESEARCH.md#4-success-criteria)). The 2-pager names no check, and it
ties the ontology relationships to generation, not to validation. Candidates will come from the
Messages API as decoded JSON. A candidate can therefore hold any JSON value: a string, `true`,
`null`, `NaN`, an integer beyond float range, or something that is not an object at all.

What the model holds that a record can break:

- **The codebook** of every categorical field (`bko.datasets.uci.CODEBOOK`), and the label's codes
  (`bko.datasets.cells.LABEL_CODES`, beside `LABEL`).
- **The range constraints** of the numeric fields (`bko.ontology.mapping.RANGES`). They catch gross
  errors only (ADR [`research/ontology-mapping-design`](ontology-mapping-design.md),
  Consequences).
- **The archive's precision:** whole numbers, and `oldpeak` at `bko.datasets.uci.FRACTION_DIGITS`
  decimals.
- **No ontology relationship.** The mapping rows record presence only, so a check derived from
  `is_a` rejects nothing (same ADR, Consequences). No other axiom gives a check:
  - Measured on 2026-10-17 on the releases `bko.datasets.sources` holds, HPO declares no
    `disjoint_from`, `relationship` or `intersection_of` (`grep -c` of each tag prints 0).
  - The mapped MONDO concept, MONDO:0006715, carries only `is_a`.
  - The table maps one MONDO concept to a UCI field, and every MONDO class a record reaches is that
    concept or one of its `is_a` ancestors. A satisfiable ontology declares no two of those
    disjoint, since that concept would belong to both.

Measured on 2026-10-17 over the 297 complete Cleveland records:

- Every numeric value lies inside its range.
- No `oldpeak` has more than one decimal.
- 65 records have `thalach` above 220 minus `age`.

## Decision Drivers

* **Must count every candidate generation hands in**, so a malformed one lowers validity rather
  than vanishing.
* **Must name each check**, so the counts say which promise generation breaks.
* **Must never raise on a candidate's content**, since that would lose a paid batch.
* **Must read each bound from its home:** the codebook in `bko.datasets.uci`, the ranges in
  `bko.ontology.mapping`.
* **Should add no check that cannot fail.**

## Decision

We will **check each candidate field by field against the codebook, the mapping table's ranges and
the archive's precision. A candidate is refused, never repaired, and counted under every check it
fails.**

- **Input:** the decoded candidates, one item per record generation produced, unfiltered. A batch
  that is one record, a string, bytes or a frame raises `TypeError`.
- **`fields`:** a candidate that is not a mapping fails it, and nothing else is read. A mapping fails
  it when it names a key outside the evaluation frame's columns; its fields are still checked.
- **One chain per field**, stopped by its first failure:
  - `number:<field>`: an `int`, a `float`, or a numpy integer or floating value; finite; never a
    `bool`. A missing key fails it.
  - Then the bound: `code:<field>` for a coded field and the label, or `range:<field>` (both ends
    inclusive) for a numeric field.
  - Then `precision:<field>` for a numeric field: whole, or `oldpeak` at one decimal, tested as
    `round(value, digits) == value`.
- **No coercion and no repair:** a numeric string fails `number`. A value is never rounded or
  clipped.
- **Counting:**
  - A candidate passes when it fails no check.
  - `generated` counts every item given, and `passed` counts the accepted ones.
  - Each check counts the candidates that failed it, so the counts can sum past the number of
    rejected candidates.
- **Output:**
  - The accepted candidates, as a frame with the evaluation frame's columns and dtypes, indexed by
    input position under `candidate`.
  - Each rejected candidate's position and failed checks.
  - The per-check counts, derived from those.
  - One INFO summary per batch, with counts only, and a WARNING for an empty batch.
- **No threshold:** the stage reports counts, and the final evaluation judges 95%.
- **No ontology check**, for the reasons in Context.
- **Library only**, with the `data` extra. The evaluation frame's columns and dtypes live in
  `bko.datasets.uci`, so this stage imports only upstream stages.

## Rationale

Decoded candidates, not a frame. A frame would force generation to convert types first, so a
string, a `bool` or a missing value would never reach the counts, and validity would rise. One
chain per field has two effects. A check never reads a value an earlier check refused, which is
what keeps `None <= 120` and `[1] in codes` from raising. And each check can be planted alone. A
whole-number check on a coded field could never fail alone, because a non-whole code lies outside
its codebook; so precision applies to the numeric fields only.

The bounds come from their existing homes. A codes constraint in the mapping table would be a
second home for the codebook. The frame's columns and dtypes live in `bko.datasets.uci` with the
frame they describe, rather than in `bko.baselines.records`, a downstream stage.

Repairing a near miss would count a broken record as valid. Judging 95% here would put a second
copy of the criterion in the stage.

## Consequences

**Gained:**
- A validity rate any reader can recompute from named checks.
- Per-check counts that point at the promise generation breaks.
- No exception from model output.

**Accepted:**
- The checks cover format, codes, gross ranges and decimals only. On UCI Heart Disease a rate near
  100% is expected, and it says nothing about clinical plausibility: an in-range, in-codebook
  combination always passes.
- A candidate inside the ranges but outside a training split's range passes here and is refused by
  `bko.baselines.records.check` (`cleanup:tasks/bko-arm-split-range-unhandled`).
- The ranges await clinician review (ADR
  [`research/ontology-mapping-design`](ontology-mapping-design.md)).
- Validation counts only the candidates it is given, so a response generation fails to decode, or
  an attempt the evaluation discards, is outside these counts
  (`cleanup:tasks/validity-denominator-unstated`).
- Accepted candidates are not deduplicated.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| A check derived from `is_a` | Rows record presence only, so it rejects nothing (Context) | 2026-10-17 |
| A pairwise `disjoint_from` check | No record can show two disjoint concepts (Context) | 2026-10-17 |
| Consistency with the fused BKB, reading "the model" as the BKO | Its classes are single-state and its rows record presence, so no record can contradict it | 2026-10-17 |
| Absence or co-occurrence rows, to give validation something to reject | Absence claims need clinical review (ADR `research/ontology-mapping-design`, Ruled Out) | 2026-10-17 |
| `thalach` at most 220 minus `age` | Rejects 65 of the 297 complete Cleveland records | 2026-10-17 |
| Bounds from the training split | They move with each split, and `bko.baselines.records` already applies them to added rows | 2026-10-17 |
| Coercing a numeric string | A JSON number decodes to a number; a string means the model broke the format | 2026-10-17 |
| Rounding or clipping a near miss | Counts a broken record as valid (Rationale) | 2026-10-17 |
| A whole-number check on coded fields | Could never fail alone (Rationale) | 2026-10-17 |
| One count per rejected candidate | Hides which promises a candidate breaks | 2026-10-17 |
| Judging 95% in the stage | The final evaluation owns the criterion | 2026-10-17 |
| A frame as input | Malformed values would never reach the counts (Rationale) | 2026-10-17 |
| Keeping each rejected candidate in the result | The caller holds the input, and the position joins the two | 2026-10-17 |
| Importing the frame's columns from `bko.baselines.records` | A downstream stage (Rationale) | 2026-10-17 |
| A codes constraint in the mapping table | A second home for the codebook | 2026-10-17 |
| A command line | No generator feeds it | 2026-10-17 |
| The `ontology` extra | No release is read | 2026-10-17 |

## Enforcement

- Owner: `src/bko/validate/`
- Pinned by: `tests/test_package_layout.py`

## Reversal Conditions

**A disjointness check joins when the table maps a second MONDO concept to a UCI field**; the check
is a search of `FIELD_MAPPINGS` in `bko.ontology.mapping` for MONDO concepts.

**A relationship check joins when a re-pinned HPO declares `disjoint_from`, `relationship` or
`intersection_of`**; the check is `grep -c` of each tag over the release.

**A cross-field check joins when `ConceptRow` in `bko.ontology.schema` gains a row kind that states
absence or exclusion**; the check is a read of that union.

**A command line joins when a task unit's acceptance names one**; the check is a search of
`tasks/`.

**The candidate shape changes when a generation request asks for Synthea records**; the check is a
search of `tasks/` and `src/bko/generate/`.

## Related

- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [The UCI evaluation frame: Cleveland records, a binary label and a cell-stratified split](uci-evaluation-frame.md)
- [The baselines: each method fills a given quota of added rows, stratum by stratum](baseline-augmentation-design.md)
- [A dataset becomes a fragment of one prior per mapped concept its records show](dataset-fragment-encoding.md)
- [src layout, one subpackage per pipeline stage](../tooling/src-layout.md)
