---
status: accepted
topic: research
created: 2026-10-17
updated: 2026-11-05
---
# Ontology classes become single-state BKB components joined by is_a and exactMatch S-nodes

## Context

Gap detection fuses ontology fragments with a dataset fragment (ADR
`research/bko-formal-gap-definition`), and the first milestone rebuilds the MONDO/DO fusion
example of Santos et al. ([`santos-2024`](../../literature.md#probabilistic-ontologies-and-bkos),
section 7; `docs/EVIDENCE.md`, R1). Both need one rule that turns a loaded release, a
`bko.ontology.terms.Ontology`, into a `bko.bkb` fragment.

The paper induces one random variable per individual and partition of disjoint classes
(Proposition 4.2.1), reads a subsumption C ⊆ D as P(x ∈ D | x ∈ C) = 1 (section 4.2), and in its
example draws each class membership of one individual a as its own variable with one state. It
bridges MONDO and DO with PTAs from the EMBL-EBI OxO service, which reports no version for its
mappings. A MONDO release carries copies of HP terms beside its own.

## Decision Drivers

* **Must rebuild the paper's 17 PTAs** from the pinned releases.
* **Must feed `fuse`** without its refusals: no repeated S-node, no source component.
* **Should give gap detection one encoding** to join the dataset fragment to.
* **Must read only pinned inputs.**

## Decision

`bko.ontology.fragments` converts a loaded release into a fragment, and `bko.bkb.inference`
reasons over fragments from an anchor.

- **Components**: each class in the caller's scope is one component, named by its CURIE, whose one
  state is `INDIVIDUAL` (`"a"`); the I-node reads "a is a member of the class".
- **`is_a`**: `is_a_fragment(ontology, scope, *, prefix, name)` gives one S-node
  P(parent | child) = 1 per direct `is_a` edge with both ends in scope. An edge to a class outside
  the scope is dropped, never shortened to an ancestor.
- **Bridge**: `exact_match_fragment(subject, subject_scope, target, target_scope, ...)` gives two
  S-nodes, one each way, for each `skos:exactMatch` the subject release declares between scoped
  classes. For MONDO and DO the subject is MONDO.
- **Scope**: a scope names live classes carrying the release's own prefix; anything else raises
  `ScopeError`.
- **Home**: the converter lives in `bko.ontology`, which imports `bko.bkb`; `bko.bkb` imports only
  the standard library.
- **Inference**: `bko.bkb.inference` holds the paper's instantiation from an anchor
  (`instantiate`: each pass adds every S-node whose antecedents hold, as its section 5.3 states),
  the I-nodes it reaches with the S-node credited for each (`reach`), the S-nodes of one inference
  (`derivation`) and their product (`probability`, section 7.2). They are this project's own, not
  pybkb's.

## Rationale

A single state per class is how the paper's example draws its BKO, and it keeps an `is_a` axiom
one S-node. Dropping an out-of-scope edge reproduces the paper's PTA count, since a class keeps
only its in-scope parents; shortening edges to in-scope ancestors would add PTAs the paper does
not have. MONDO's `skos:exactMatch` annotations give exactly the paper's four bridge pairs within
its 11 classes, from a pinned release. The prefix keeps another release's copied classes out of a
fragment. Putting the converter in `bko.ontology` keeps the ported layer free of a stage's types.
The instantiation follows the algorithm the paper states rather than the split its section 7.1
narrates, which contradicts it (`docs/EVIDENCE.md`, R1).

## Consequences

**Gained**: one fragment per release from standard-library values; the paper's example rebuilt
from the pins; the instantiation gap detection reads.
**Accepted**: one individual per BKB, because two individuals as two states of one class component
would exclude each other, which the paper's one variable per individual forbids; no "not in C"
state, so disjointness axioms (PTAs of probability 0) are not encoded; only `is_a` and
`skos:exactMatch` are read, so MONDO's `excluded_subClassOf` curation is invisible; a class with
two scoped children makes the fragment non-mutex. A fused ontology BKB is therefore non-mutex and,
through the bridge, cyclic, so `bko.bkb.reason.log_joint_complete` over it returns masses rather
than P(target, evidence). A fused BKB is valid when `bko.bkb.fusion.check_fused` passes, which
does not require mutual exclusion (ADR `research/data-contradiction-handling`, Decision 3).

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| OxO as the bridge source | A live service whose mappings carry no version, so the pins cannot fix it | 2026-10-17 |
| Transitive `is_a` edges within the scope | Adds PTAs the paper does not have | 2026-10-17 |
| The converter in `bko.bkb` | The ported layer would import a stage's types | 2026-10-17 |
| The converter in `bko.gaps` | The reproduction needs it as well, and it converts ontologies | 2026-10-17 |
| Member and non-member states per class | The paper's example draws no negated assertion | 2026-10-17 |
| Holding bridge PTAs out of the first instantiation pass | Matches the split section 7.1 narrates, but adds a rule the paper does not state | 2026-10-17 |

## Enforcement

- Owner: `src/bko/ontology/fragments.py`, `src/bko/bkb/inference.py`
- Pinned by: `tests/test_ontology_fragments.py`, `tests/test_bkb_inference.py`,
  `tests/test_ontology_santos.py`

## Reversal Conditions

**The component becomes the pair of class and individual when a fragment must hold two
individuals**; the check is a search of `src/bko/` for a second state on a class component.

**Disjointness joins as S-nodes of probability 0 when a unit's acceptance reads a disjointness
axiom**; the check is a search of `tasks/` for one.

## Related

- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
- [The BKB layer is pybkb's core, ported under GPL-3.0 and checked against pybkb's own results](bkb-layer-source.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [Each stage keeps a data contradiction and reports it; none forces the ontology onto the data](data-contradiction-handling.md)
- [src layout](../tooling/src-layout.md)
