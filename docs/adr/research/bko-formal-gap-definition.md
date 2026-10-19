---
status: accepted
topic: research
created: 2026-10-14
updated: 2026-10-19
---
# A dataset gap is an ungrounded BKO statement

## Context

The project needs a definition of a dataset gap before anything can be detected, generated or
measured against it. Two sources propose different ones.

The research brief (section A1 and A2) treats "Bayesian knowledge-driven ontologies" as
non-standard terminology, judges PR-OWL and MEBN too heavyweight for the team, and recommends
reframing the idea as a lightweight probabilistic coverage score: a gap is an ontology concept
present in the ontology with zero or statistically under-represented support in the dataset,
relative to an ontology-weighted prior, computed from expected and observed instance counts with a
Dirichlet smoothing prior.

The proposal 2-pager (section 3, Detect) adopts Bayesian Knowledge-driven Ontologies from Santos,
Jurmain and Ragazzi (PLOS ONE 19(3): e0296864, 2024) and applies the BKO instantiation algorithm to
find statements the ontology implies but the data does not support. Santos et al. state, as their
Proposition 5.1.1, that these ungrounded statements contribute nothing to the model's distribution, so the proposal uses them as
the formal definition of a gap. The paper exists and the term is established, so the brief's
premise for reframing does not hold.

## Decision Drivers

* **Must define a gap formally**, so detection is a property of the representation rather than a
  tuned threshold.
* **Should be measurable on the proposal's datasets**: UCI Heart Disease and the Synthea ablation.
* **Must keep a simpler baseline** for comparison, since the formal method is unproven on clinical
  tabular data.

## Decision

A dataset gap is **a statement implied by the fused ontology fragments but ungrounded by a dataset
fragment** under BKO instantiation, over the classes the dataset's own mapping rows reach (ADR
`research/ontology-mapping-design`). Gaps are read per cell, as (cell, statement) pairs, by three
definitions:

- **Per-subgroup, the primary definition**: each cell's records form their own dataset fragment,
  and its gaps are the statements that fragment leaves ungrounded. A cell treats the concepts only
  another label's mapping rows name as held, so a statement only those concepts ground is not its
  gap. The completeness criterion and E1's candidate read this definition.
- **Population-level, an ablation**: one fragment of every record, its gaps holding in every cell.
- **Frequency-aware, an ablation**: the statements a cell's under-represented concepts ground by
  themselves (ADR `research/frequency-aware-gap-score`).

All three read the same cells and are compared in the same places:

- **Cells**: on UCI Heart Disease, the age band, sex and label cells of ADR
  `research/uci-evaluation-frame`; on Synthea, age band and sex, since E2's label is the ablated
  condition itself.
- **Count-based gaps** (fewer than 20 training records per age-sex subgroup-label combination on
  UCI Heart Disease) are detected and evaluated separately, as the comparison the formal
  definitions are measured against.
- **Where they are compared**: E2 scores each definition against the (cell, statement) pairs whose
  statement the removed patients of the cell ground, whether or not the cell's other records
  ground it; E1 runs one BKO arm per definition (ADR `research/e1-evaluation-protocol`).
- **Completeness**: its own request, not E1's. Each UCI cell holding a per-subgroup gap in the
  default training split is asked for 20 rows, the threshold of `docs/RESEARCH.md` section 2. The
  generated records that pass validation join their cell's records, and a gap closes when that
  cell's fragment then grounds it. ADR `research/gap-closure-design` holds which records join and
  how each definition's closure is read.

## Rationale

Santos et al. state, as their Proposition 5.1.1, that an ungrounded PAA leaves every marginal
and posterior the BKO induces unchanged, which makes ungrounded statements a definition rather
than a heuristic. The paper judges grounding against the whole BKO (Definitions 5.1.4 and
5.1.5); this record judges it against the dataset fragment, the project's adaptation.
Rebuilding the paper's MONDO/DO fusion example
([R1](../../EVIDENCE.md#r1-mondo-and-do-fusion-reproduction)) checks that `bko.bkb` runs the
paper's fusion and instantiation; the example holds no ungrounded PAA, so it does not test the
proposition. The coverage score needs a prior and a threshold chosen by the team, and a
different choice moves the gap set. Evaluating count-based gaps separately keeps the simple
definition in the results, so a null result for the formal one is still reportable against it.

Grounding per cell keeps Proposition 5.1.1's reading, since each cell's fused BKB is a BKO whose
ungrounded statements contribute nothing to it, and measures at the project's unit of
underrepresentation (`docs/RESEARCH.md` section 2). A single population fragment grounds a
statement once any record anywhere shows its concept: on UCI Heart Disease it leaves no gap, and
in E2's `cell` ablation the remaining patients still ground every statement the removed ones held
([E2](../../EVIDENCE.md#e2-synthea-gap-detection-ablation)). The two ablations bracket the primary
definition: population-level grounding sees only absence everywhere, and the frequency-aware score
sees rare support but never an empty cell. A cell of one label lacks the other label's
concepts by construction, so counting their statements would fill every UCI cell without disease
with gaps no generation should close. Completeness asks for every cell that holds a gap because E1's
count-based request fills only cells with disease under the threshold, so a gap elsewhere could
never close and the criterion would measure the request rather than the generation. Synthea's E2 cells carry no label because the label there is
the removed condition, whose statements are the truth E2 asks a detector to find.

## Consequences

**Gained**: a gap set that follows from the representation; a citable formal basis.
**Accepted**: gap detection runs on `bko.bkb`, a port of pybkb's core (ADR
`research/bkb-layer-source`), whose update needs complete evidence; the schedule depends on it.
In a fused BKB a source prior supports only its source assignment, and an ontology statement's
class antecedent is supported by any statement with that consequent (Santos et al., Definitions
5.1.4 and 5.1.5), so ontology statements support each other whether or not the dataset holds them:
grounding is judged against a dataset fragment, not the fused BKB; grounding runs once per cell; a
cell of a few records leaves most concepts unshown, so its per-subgroup gaps partly restate its
size, which the count-based detector already flags; E1's paid generation runs once per definition;
E2's complete removal cannot measure how often per-subgroup grounding lists a statement a cell's
population shows, which for records drawn independently is (1 - p)^n for a concept of share p in a
cell of n records, the chance the frequency-aware score weighs.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Coverage score with a Dirichlet prior as the gap definition | Rests on the premise that BKOs are not an established formalism, which Santos et al. contradicts; its threshold and prior are free choices. Kept as the count-based comparison; a coverage-style score with a Beta posterior runs as an ablation (ADR `research/frequency-aware-gap-score`) | 2026-10-13 |
| PR-OWL / MEBN implementation | Heavyweight, with no maintained tooling; BKOs are the formalism the proposal builds on | 2026-10-13 |
| Counting completeness over E1's count-based request | Holds gaps in cells it never fills, which cannot close | 2026-10-17 |
| Population-level grounding as the primary definition | Grounds a statement once any record anywhere shows its concept: no gap on UCI Heart Disease, and none of E2's `cell` ablation | 2026-10-17 |
| Reversing to population-level grounding on E2's count of per-subgroup gaps outside the truth | E2 removes whole patients completely, so a per-subgroup gap new after a removal lies in the truth by construction, and counting every gap after it outside the truth scores absences the removal did not cause, such as a cell with no case, as errors; neither reading tests the definition | 2026-10-17 |

## Enforcement

- Owner: `src/bko/gaps/`
- Pinned by: `tests/test_gaps_detect.py`, `tests/test_gaps_definitions.py`,
  `tests/test_gaps_cli.py`, `tests/test_gaps_experiment.py`

## Reversal Conditions

**The formal definition is replaced by the count-based one if BKO instantiation cannot run on the
fused MONDO, HPO and dataset fragments by [Status Update 1](../../ROADMAP.md#status-update-1)**;
E2 in [`docs/EVIDENCE.md`](../../EVIDENCE.md#e2-synthea-gap-detection-ablation) records whether it
ran. The count-based definition then becomes primary and this record is rewritten.

## Related

- [src layout](../tooling/src-layout.md)
- [Ontology classes become single-state BKB components joined by is_a and exactMatch S-nodes](bko-fragment-encoding.md)
- [The BKB layer is pybkb's core, ported under GPL-3.0 and checked against pybkb's own results](bkb-layer-source.md)
- [A dataset becomes a fragment of one prior per mapped concept its records show](dataset-fragment-encoding.md)
- [A frequency-aware gap: a concept credibly under four-fifths of its share in the rest of its label group](frequency-aware-gap-score.md)
- [E1's evaluation: repeated holdouts, one classifier, separation judged on paired intervals](e1-evaluation-protocol.md)
- [Validated records join the training records of their asked cell, and closure is read per gap definition](gap-closure-design.md)
