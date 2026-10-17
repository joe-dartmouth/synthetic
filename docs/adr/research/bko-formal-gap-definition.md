---
status: proposed
topic: research
created: 2026-10-14
updated: 2026-10-16
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

A dataset gap is **a statement implied by the fused ontology fragments but ungrounded by the
dataset fragment** under BKO instantiation. Count-based gaps (fewer than 20 training records per
age-sex subgroup-label combination on UCI Heart Disease) are detected and evaluated separately, as
the comparison the formal definition is measured against.

The record stays `proposed` until the MONDO/DO fusion from Santos et al. is reproduced with
`bko.bkb`, which is the first milestone in `docs/ROADMAP.md`.

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

## Consequences

**Gained**: a gap set that follows from the representation; a citable formal basis.
**Accepted**: gap detection runs on `bko.bkb`, a port of pybkb's core (ADR
`research/bkb-layer-source`), whose update needs complete evidence; the schedule depends on it.
In a fused BKB a source prior supports only its source assignment, and an ontology statement's
class antecedent is supported by any statement with that consequent (Santos et al., Definitions
5.1.4 and 5.1.5), so ontology statements support each other whether or not the dataset holds them:
grounding is judged against the dataset fragment, not the fused BKB.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Coverage score with a Dirichlet prior as the gap definition | Rests on the premise that BKOs are not an established formalism, which Santos et al. contradicts; its threshold and prior are free choices. Kept as the count-based comparison | 2026-10-13 |
| PR-OWL / MEBN implementation | Heavyweight, with no maintained tooling; BKOs are the formalism the proposal builds on | 2026-10-13 |

## Enforcement

- Owner: `src/bko/gaps/`
- Pinned by: nothing -- convention only

## Reversal Conditions

**The formal definition is replaced by the count-based one if BKO instantiation cannot run on the
fused MONDO, HPO and dataset fragments by Status Update 1** (an acceptance item of
[`proposal-gap-detection`](../../../tasks/proposal-gap-detection.md)). The count-based definition then becomes primary and this record is rewritten.

**The record becomes `accepted` when the MONDO/DO fusion from Santos et al. is reproduced** with the
BKB layer.

## Related

- [src layout](../tooling/src-layout.md)
