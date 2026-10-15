---
id: proposal-ontology-mapping
title: MONDO and HPO loading and field mapping
status: ready
depends_on: --
blocks: proposal-mondo-do-fusion, proposal-record-validation
parallel_safe: true
owner: Partiban
milestone: status-update-1
paid: false
adrs: --
---

# MONDO and HPO loading and field mapping

The set's graph is [`proposal-index.md`](proposal-index.md).

## Why this unit exists

Represent loads MONDO and HPO as BKO fragments and maps ontology concepts to dataset fields,
starting with the disease label and chest-pain type
([RESEARCH, Represent](../docs/RESEARCH.md#represent)).

## Scope

**In.**

- Load MONDO and HPO through owlready2, pronto or OLS4 in `src/bko/ontology/`.
- A mapping table from concepts to UCI fields recording each mapping's source, encoding and limits;
  numeric fields get separately specified range constraints; concepts with no defensible field are
  listed as out of scope.
- The same table maps the SNOMED-CT codes of the Synthea conditions the ablation removes (the
  codes `bko.datasets.synthea` names) to MONDO or HPO concepts, each row naming its source, so
  the BKO can ground the Synthea population that E2 ablates.

**Out.** Clinician review, which the 2-pager makes conditional on availability.

## Acceptance

- The mapping table is versioned in the repository and every row names its source.
- `make check`.
