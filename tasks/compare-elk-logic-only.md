---
id: compare-elk-logic-only
title: Compare against ELK and related systems, and measure what the BKB's probabilities add
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: research/logic-only-ablation
---

# Compare against ELK and related systems, and measure what the BKB's probabilities add

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

A reader of Santos et al. (2024) asked how a BKO differs from a description-logic reasoner such as
ELK, the OWL 2 EL classifier, which neither the paper nor this repository cites. The two answer
different questions:
- ELK classifies a consistent ontology: it computes every subsumption in polynomial time.
- A BKO weighs conflicting sources and, in this pipeline, the dataset's concept frequencies.

E6 sharpened the question. `bko-direct`, which draws records from the fused BKB with no language
model, was the strongest arm with the cell wholly removed
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e6-diabetes-130-subgroup-removal)). Its draw reads
each concept's share from the BKB's data priors, while the `is_a` closure is recorded and never
read into a field (`bko.generate.direct`). How much of the result comes from the logical structure
and how much from the probabilities has not been measured.

The instructor's feedback on the 2026-11-04 status update named two more systems to read
against: BioGPT, a biomedical language model, and NCATS's Biomedical Data Translator, a
knowledge-graph reasoning program. Neither is cited either.

There is also a precondition to check. MONDO's and HPO's release pipelines may already classify
their ontologies with ELK. If so, the hierarchy `bko.ontology` loads from the pinned releases
already holds ELK's inferred subclasses, and swapping ELK in would change little.

## Scope

**In.**

- **Phase 1, hierarchy check.** Classify the pinned MONDO and HPO OWL releases with ELK through
  ROBOT (`robot reason --reasoner ELK`), which needs a JDK, as `make synthea` already does.
  Compare its subsumptions with the hierarchy `bko.ontology` loads, restricted to the concepts the
  UCI and Diabetes field mappings name and their ancestors. Report each subclass link that one side
  holds and the other lacks.
- **Phase 2, logic-only ablation.** A configuration in which every concept's probability comes
  from logic alone: a concept is present or absent as the mappings and the closure entail, with no
  data prior. The ablation's ADR fixes how presence is drawn without a rate before any run. Two
  readings, both free:
  - E2's gap detection (`python -m bko.gaps synthea`) beside its BKO definitions;
  - `bko-direct` on E6's frame and removal at both fractions, as a new direct arm in a named run,
    paired against `bko-direct` on recall, AUC-PR and distance to closest record.

  If Phase 1 finds differences, each reading also runs on the ELK-classified hierarchy.
- **Phase 3, related work.**
  - Cite ELK, BioGPT and the Biomedical Data Translator in
    [`docs/literature.md`](../docs/literature.md).
  - Add a related-work section to [`docs/RESEARCH.md`](../docs/RESEARCH.md):
    - what a DL classifier such as ELK computes and a BKO does not, and the reverse;
    - how a biomedical language model such as BioGPT compares with the general model the
      Messages arms use, and whether it could stand in as a generator;
    - how the Translator's knowledge-graph reasoning over many sources relates to BKO fusion.
  - Record the result as a new entry in [`docs/EVIDENCE.md`](../docs/EVIDENCE.md).
  - Write a new ADR under `docs/adr/research/` for the ablation's design.

**Out.**
- Replacing `bko.bkb` or `bko.ontology`'s loader with a reasoner.
- A scaling benchmark over whole ontologies or enterprise data, which is a separate unit if taken
  up.
- Any paid arm.

## Acceptance

- Phase 1's comparison is a committed command over the pinned releases, with its differing links
  listed, or an empty list stated.
- The logic-only readings come from committed commands with no API call. They are paired against
  `bko-direct` and the BKO gap definitions on the same holdouts and inputs.
- ELK, BioGPT and the Translator are cited through the registry, and
  `tests/test_docs_citations.py` passes.
- `make check`.
