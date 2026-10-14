---
status: proposed
topic: research
created: 2026-10-14
updated: 2026-10-14
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
find statements the ontology implies but the data does not support. Santos et al. show these
ungrounded statements contribute nothing to the model's distribution, so the proposal uses them as
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

The record stays `proposed` until the BKB layer exists and the MONDO/DO fusion from Santos et al. is
reproduced, which is the first milestone in `docs/ROADMAP.md`.

## Rationale

By the 2-pager's account, Santos et al. prove that such statements contribute nothing to the
distribution, which would make it a definition rather than a heuristic; the paper is not yet
fetched, and reproducing its MONDO/DO fusion is what checks the account. The coverage score
needs a prior and a threshold chosen by the team, and a different choice moves the gap set.
Evaluating count-based gaps separately keeps the simple definition in the results, so a null result
for the formal one is still reportable against it.

## Consequences

**Gained**: a gap set that follows from the representation; a citable formal basis.
**Accepted**: the team must build or port a BKB inference layer before any gap can be detected; the
schedule depends on it.

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
