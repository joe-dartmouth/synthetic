---
id: ontology-per-dataset-releases
title: Hold ontology releases per dataset and convert an OWL-only release to OBO at fetch time
status: blocked
depends_on: refactor-evaluate-size-splits
blocks: datasets-acs-frame
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Hold ontology releases per dataset and convert an OWL-only release to OBO at fetch time

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

`RELEASES` (`bko.ontology.mapping`) is global and holds HPO and MONDO only. `load_scope`
(`bko.gaps.scope`) refuses a concept whose prefix has no pinned release. The Occupation Ontology
ships OWL only, and the loader reads OBO.

## Scope

**In.**

- Releases held per dataset, beside `field_rows(dataset, ...)` (`bko.evaluate.data`).
- An OWL release converted to OBO at fetch time by the pinned ROBOT jar
  (`bko.datasets.sources.ROBOT_JAR`).
- ADR [`research/ontology-mapping-design`](../docs/adr/research/ontology-mapping-design.md)
  revised.

**Out.**

- A Python OWL parser.
- SKOS vocabularies.

## Acceptance

- A fixture dataset with its own prefix loads its scope.
- The existing datasets' release tests (`tests/test_ontology_pinned_releases.py`) pass with
  `--releases`.
- `make check` passes.
