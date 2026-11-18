---
id: plan-direct-method-research
title: Write the direct-method study into the research plan
status: ready
depends_on: --
blocks: generate-no-mapping-ablation, baselines-label-bootstrap, evaluate-prior-shift-comparator
parallel_safe: false
milestone: direct-method-research
paid: false
adrs: research/direct-method-primary
---

# Write the direct-method study into the research plan

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

`bko` is the registered candidate, and no criterion reads `bko-direct`
([`docs/RESEARCH.md`](../docs/RESEARCH.md#generate-and-validate)). Adopting an idea changes the
plan first ([`docs/IDEAS.md`](../docs/IDEAS.md)), so every later unit of this set waits on a plan
that names `bko-direct` as its candidate and fixes how the controls are read before any control
runs.

## Scope

**In.**

- A new `## 11. Direct-method study` in [`docs/RESEARCH.md`](../docs/RESEARCH.md), leaving
  sections 3 and 4 as they are. It states:
  - the question;
  - `bko-direct` as the candidate;
  - the BKO variants (`bko`, `bko-population`, `bko-frequency`, `bko-logic`, `bko-skeleton`, the
    guidance levels) as ablations, and the row-adding methods as comparators;
  - UCI, Synthea and Diabetes 130 as development data, and SUPPORT2, NHANES and ACS PUMS as
    confirmatory data;
  - a disclosure that the candidate was chosen after E6;
  - for each [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry, whether the study reads it as a
    development result, an ablation, or not at all;
  - how completeness and validity are read for `bko-direct`;
  - the control reading rule and the variant inclusion rule below, copied in full.
- Positioning against hot-deck imputation, data swapping, covariate-bias augmentation and
  disjoint generation, in [`docs/BACKGROUND.md`](../docs/BACKGROUND.md#where-bko-direct-sits),
  with an entry in [`docs/literature.md`](../docs/literature.md) for each work it first cites.
- A new ADR under `docs/adr/research/` that records the promotion.

**Control reading rule.** Contrasts: `bko-direct` minus each of `label-bootstrap`, `bko-nomap`
and `prior-shift`, on removed-cell recall, flagged AUC-PR and flagged Brier, at each fraction, on
E6's and E3's frames. Margin on recall: 0.0480, the half-width of E6's `bko-direct` minus CTGAN
interval on removed-cell recall at full removal (+0.0595 [+0.0115, +0.1075]). Each contrast falls
in one branch:

| Branch | Interval | Reading |
|---|---|---|
| above | wholly above 0 | The claim is made for that measure, frame and fraction |
| equivalent | wholly inside the margin | A tie is reported; the claim is not made |
| below | wholly below 0 | The control is reported better; it joins the confirmatory primary comparators |
| inconclusive | anything else | Reported as inconclusive; no claim |

What follows: every unit proceeds in every branch. Against `label-bootstrap` or `bko-nomap`,
anything but `above` means pooling is reported as the mechanism at that fraction, and a mapping or
structure claim can come only from the arms of `generate-is-a-conditional-draws` or
`generate-ancestor-shrunk-shares`. The confirmatory primary endpoint
(`evaluate-confirmatory-registration`) is removed-cell recall if the `prior-shift` contrast is
`above` on recall on E6's frame at either fraction; else flagged Brier if that contrast is in
`bko-direct`'s favour; else no superiority hypothesis is registered and the confirmatory run
reports equivalence, privacy and attribution readings.

**Variant inclusion rule.** A variant enters `bko-direct-2` when, on each development frame it
runs on, its target measure improves on `bko-direct`'s (interval excludes 0) and its removed-cell
recall is non-inferior within the margin above. Each variant registers one parameter value, except
the `is_a` pseudo-count: of the registered values that pass, the largest enters.

**Out.**

- Any code.
- RESEARCH sections 3, 4 and 10.
- E1's `candidate`.
- A venue.

## Acceptance

- RESEARCH section 11 states each item under In, and both rules in full.
- `tests/test_docs_citations.py` passes.
- `make check` passes.
