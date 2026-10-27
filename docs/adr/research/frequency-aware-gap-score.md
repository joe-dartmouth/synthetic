---
status: accepted
topic: research
created: 2026-10-17
updated: 2026-10-27
---
# A frequency-aware gap: a concept credibly under four-fifths of its share in the rest of its label group

## Context

ADR `research/bko-formal-gap-definition` runs a frequency-aware score as one of two ablations beside
per-subgroup grounding, its primary definition. Binary grounding flags a concept only when no record
of a cell shows it: it is blind to a concept a cell shows rarely, and in a cell of a few records it
flags absences that the cell's size explains. The research brief (A2) proposed the other reading, a
concept with zero or statistically under-represented support against an expected count under a
Dirichlet prior. Each dataset fragment already carries the share of its records that show each
concept (ADR `research/dataset-fragment-encoding`). On UCI Heart Disease's default training split
(measured 2026-10-17) the cells hold 2 to 58 records, and a concept's share depends mostly on the
label: chest pain shows in 0.218 of the records with disease.

## Decision Drivers

* **Must compare a cell with records of its own label**, since the label alone moves most concepts.
* **Must not flag what a cell's size explains.**
* **Must fix the rule and threshold before E2 is run again or E1 runs.**
* **Should give gaps in the grounding definitions' form**, so E2 and the completeness criterion read
  all three alike.

## Decision

- **Reference.** For each cell of ADR `research/bko-formal-gap-definition` and each concept the
  dataset's mapping rows name, the reference share p is the concept's share among the records
  outside the cell that share its label; where cells carry no label (Synthea), among every record
  outside the cell.
- **Rule.** With k of the cell's n records showing the concept, the concept is under-represented in
  the cell when P(θ < 0.8 p) ≥ 0.95 for θ ~ Beta(k + 1/2, n − k + 1/2), the posterior under
  Jeffreys' prior. A reference share of 0, or no record outside the cell, flags nothing.
- **Level.** The level is a parameter: `gaps` and `under_represented` take it as `credibility`,
  and 0.95, the level E2 ran, is the default. D1 sweeps it over `bko.gaps.sweep`'s
  `CREDIBILITY_LEVELS` and chooses one by a rule stated before its run: on E2's `cell` ablation
  at fraction 0.5 (`cell-50`), among the levels whose frequency-aware sensitivity is at least the
  count-based detector's, the level with the highest frequency-aware specificity, the lower level
  on a tie; no level is chosen when none qualifies. `bko-frequency` keeps 0.95 until
  [`docs/RESEARCH.md`](../../RESEARCH.md#detect) says it reads the chosen level.
- **No minimum cell size.** The posterior is the size guard: an empty cell's posterior is the prior,
  which puts at most 0.705 of its mass below 0.8 p, so an empty cell is never flagged. A cell none
  of whose n records shows a concept is flagged for it only when p exceeds 0.964 at n = 1, 0.711
  at n = 2 and 0.555 at n = 3 (computed 2026-10-17).
- **Gaps.** A cell's frequency-aware gaps are the statements its under-represented concepts, taken
  together, ground by themselves under BKO instantiation over the cell's fragment: those that
  instantiation from the cell's held concepts together with the under-represented ones grounds,
  and from the held concepts without them does not. The held concepts include those the cell
  assumes (ADR `research/bko-formal-gap-definition`), and grounding without the under-represented
  concepts leaves their priors out of the fragment, since a prior grounds its concept whatever the
  anchor. An under-represented concept no record of the cell shows counts too.
- **Home.** `bko.gaps.definitions`, beside the two grounding definitions; its `SHARE_RATIO` and
  `CREDIBILITY` hold the ratio and the default level, and `bko.gaps.sweep` the swept levels and
  the rule that chooses one.

## Rationale

Four-fifths is the line the US Uniform Guidelines on Employee Selection Procedures draw for adverse
impact: a group's selection rate under four-fifths of the highest group's rate is generally regarded
as evidence of it ([`eeoc-1978`](../../literature.md#gap-detection-and-data-quality)). It is an
established convention for material under-representation, not a number the team tuned. The
posterior level of 0.95 asks for credible evidence of that shortfall, not a point estimate under it,
and Jeffreys' prior is the standard uninformative prior for a binomial share, so it adds no
parameter. Leaving the cell out of its own reference keeps a large cell from diluting the contrast
it is measured against.

Measured 2026-10-17 over UCI training splits only, the default split and E1's holdout seeds,
never a test row: the rule flags 5 (cell, concept) pairs on the default split and 2 to 8 per
holdout (median 4). With the cell kept in its reference it flags 1 to 5. A ratio of 1 flags 5 to 12
per holdout, most within sampling noise, and a ratio of 0.5 flags 0 to 3. Against the population
share instead of the label group's, a one-sided binomial test at a false-discovery rate of 0.05
flags 9 to 15 pairs per holdout, nearly all of them disease markers absent from cells without
disease.

Mapping the flagged concepts to the statements they alone ground gives (cell, statement) pairs, the
form the grounding definitions give, so E2's reading and the completeness criterion apply unchanged.

## Consequences

**Gained**: a gap reading that sees rare support, not only none; output in the grounding
definitions' form.
**Accepted**: the ratio and the level are conventions; a label group's age and sex mix still moves
the reference, so a clinically expected difference (abnormal exercise stress tests in women without
disease) reads as a gap; a concept absent from every cell has no reference and is never
flagged, as in E2's `condition` ablation, which population-level grounding catches; an empty cell
is never flagged, and a cell of a few records only for a concept common in the rest of its label
group, which per-subgroup grounding and the count-based detector catch.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The population share as the reference | Flags disease markers absent from cells without disease: a one-sided binomial test against it flags 9 to 15 pairs per UCI holdout, nearly all label effects | 2026-10-17 |
| The label group's share with the cell kept in | A large cell dilutes the reference it is compared with | 2026-10-17 |
| A one-sided binomial test with false-discovery control | Flags any shortfall, with no effect size, outside the Bayesian reading the BKO and the brief share | 2026-10-17 |
| A ratio of 1 | Flags any credible shortfall: 5 to 12 pairs per UCI holdout, most within sampling noise | 2026-10-17 |
| A minimum cell size | The posterior already refuses a cell too small to show a shortfall | 2026-10-17 |
| Leaving the under-represented concepts out of the anchor only | A fragment's prior grounds its concept whatever the anchor, so a concept some record of the cell shows would yield no gap | 2026-10-17 |

## Enforcement

- Owner: `src/bko/gaps/`
- Pinned by: `tests/test_gaps_definitions.py`, `tests/test_gaps_experiment.py`

## Reversal Conditions

**The rule changes only while `docs/EVIDENCE.md` (E2) records no run of the three definitions**;
after that run, a changed rule is a new experiment entry, as
[D1](../../EVIDENCE.md#d1-graded-grounding-and-the-frequency-aware-level) is for the level.

**The score leaves E1 if `docs/EVIDENCE.md` (E2) records it recovering no eligible gap of the `cell`
ablation**; E1 then runs two BKO arms, and this record and ADR `research/e1-evaluation-protocol` say
why.

## Related

- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
- [A dataset becomes a fragment of one prior per mapped concept its records show](dataset-fragment-encoding.md)
- [E1's evaluation: repeated holdouts, one classifier, separation judged on paired intervals](e1-evaluation-protocol.md)
