---
status: accepted
topic: research
created: 2026-11-22
updated: 2026-11-22
---
# `bko-direct` is the candidate of the direct-method study, and E1's candidate stays `bko`

## Context

E1 registered `bko`, the Messages arm guided by the BKO's gaps, as its candidate, and section 4 of
the plan judges it. The direct arm `bko-direct` draws records from the fused BKB with no API call
(ADR [`research/bkb-direct-generation`](bkb-direct-generation.md)), and no section 4 criterion
reads it. E6 read `bko-direct` as the arm with the highest removed-cell recall at full removal
other than `real`. The team's direct-method milestone asks how far that arm goes as a method in its
own right, which needs a named candidate, readings fixed before its controls run, and a statement
of which earlier results the study may lean on.

## Decision Drivers

* **Must keep E1's and E6's registrations intact**, so their verdicts stand as recorded.
* **Must fix how the controls are read before any control runs**, so no reading is chosen after
  its result is seen.
* **Must disclose that the candidate was chosen after E6 was read.**

## Decision

We will make **`bko-direct` the candidate of the direct-method study only**, as
[`docs/RESEARCH.md` section 11](../../RESEARCH.md#11-direct-method-study) states it. E1's and E6's
registered candidate stays `bko`, and their verdicts stand. The study's question, data, ablations,
comparators, published comparisons, synthetic-share ablation, control reading rule and variant
inclusion rule live in section 11 and are not copied here.

## Rationale

The candidate was chosen after E6's results were read, so E6 cannot test it: section 11 names E6
development data for this study and classifies every register entry as a development result, an
ablation, or not read. Keeping E1's `candidate` unchanged leaves its registered criteria and
verdicts as they were judged. Writing the study as its own section, with readings rather than
criteria, keeps section 4 the only home of a criterion. Fixing the two rules in the plan before any
control unit lands means a control's result cannot shape how it is read.

## Consequences

**Gained**: a named candidate, fixed readings and a disclosed development set before the controls
run; E1's and E6's records untouched.
**Accepted**: E6, the result that suggested the candidate, counts only as development evidence, so
the study's claims rest on runs not yet made.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Switching E1's `candidate` to `bko-direct` | Rewrites a registered experiment after its results were read; E1's verdicts would no longer be the ones judged | 2026-11-22 |
| Rewriting section 4's criteria around `bko-direct` | Section 4 judges E1; a study chosen after E6 adds readings, not criteria | 2026-11-22 |
| Keeping `bko` as the study's candidate | The study asks about the direct method; `bko` is one of its ablations | 2026-11-22 |
| A companion page for the task set under `docs/` | A set has no companion page; the plan holds the study and the board holds its units | 2026-11-22 |

## Enforcement

- Owner: `docs/RESEARCH.md`
- Pinned by: nothing -- convention only

## Reversal Conditions

A named comparator reads `below` on the confirmatory primary measure under section 11's control
reading rule; or E1's registered `candidate` changes.

## Related

- [E1 evaluation protocol](e1-evaluation-protocol.md)
- [Diabetes 130 subgroup removal design](diabetes-subgroup-removal-design.md)
- [BKB direct generation](bkb-direct-generation.md)
- [Synthea downstream design](synthea-downstream-design.md)
