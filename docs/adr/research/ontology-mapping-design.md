---
status: accepted
topic: research
created: 2026-10-16
updated: 2026-10-17
---
# The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows

## Context

RESEARCH, Represent, asks the BKO to map ontology concepts to dataset fields, starting with the
disease label and chest-pain type; to document each mapping's source, encoding and limits, with
clinician review where available; to govern numeric fields by separately specified range
constraints; and to exclude from scope any concept without a defensible field
([`docs/RESEARCH.md`](../../RESEARCH.md#represent)). The 2-pager names three tools for ontology
access: owlready2, pronto and OLS4. The gate is offline: it reads no download and makes no request.

Measured on 2026-10-16 on the MONDO and HPO releases `bko.datasets.sources` holds:

- **MONDO carries `property_value: skos:exactMatch SCTID:<code>` annotations.** The Synthea
  ablation's SNOMED-CT code sits on the myocardial ischemia concept and on no other live term.
- **HPO carries no SNOMED-CT reference.**
- **pronto read both releases**, with every term converted to standard-library values, no import
  and no warning.

`bko.datasets.uci` records the codebook of every categorical field, but no unit or range for a
numeric one. owlready2 is the lock's only package that ships an sdist alone, so it builds Cython on
every clone that installs its extra.

## Decision Drivers

* **Must name a source for every row**, so a reader can check each mapping against what it cites.
* **Must keep the gate offline and deterministic.**
* **Must let downstream stages read the table without a parser.**
* **Should build no sdist on a fresh clone.**
* **Should make the BKO's scope explicit**, so an unmapped concept is not read as a gap.

## Decision

We will **read pinned GitHub release OBO files with pronto, and hold the mapping as typed
constants in `bko.ontology.mapping`, each row naming its source.**

- **Ontology access**: the MONDO and HPO GitHub release OBO files, parsed by pronto; DO, which only
  `bko.ontology.santos` reads, from its release tag's `src/ontology/doid.obo`, since DO's GitHub
  releases carry no assets.
- **Pins and fetch**: the releases are `MONDO`, `HPO` and `DO` in `bko.datasets.sources`, with the size
  and sha256 `bko.datasets.sources` holds; `bko.ontology.releases.load_releases` calls
  `bko.datasets.fetch.ensure` for the commands that read them.
- **Loader output**: `bko.ontology.obo` is the only module that imports pronto; it returns frozen
  standard-library `Ontology` and `Term` values with sorted tuples and a sorted term map.
- **Table form**: typed `Final` constants in `bko.ontology.mapping`, with the row types in
  `bko.ontology.schema`.
- **Row meaning**: when every clause of a row holds for a record, the record shows the row's
  concept (SSSOM `skos:broadMatch`, with the dataset state as subject). Rows record presence only.
- **Mapping rule**: a field state maps to a concept when `heart-disease.names` names the concept,
  when a cited guideline defines the concept by a cutoff on the field, or when the Cleveland
  protocol papers (Detrano 1985) define the field's finding. Every row records its limits. The
  resting blood pressure cutoff is 130 mm Hg.
- **Scope**: a concept is in scope when a row maps it or it is an `is_a` ancestor of a mapped
  concept in its release, reached through classes with the release's prefix. Every other concept is
  out of scope; `OUT_OF_SCOPE` records the near misses considered and why each was rejected, and the
  check refuses one that is an `is_a` ancestor of a mapped concept. Each dataset's BKO holds the
  part of this scope its own rows reach: the field mappings' concepts for UCI Heart Disease and the
  code mappings' for Synthea, each with its `is_a` ancestors in its release, following only parents
  with the release's prefix.
- **Range bounds**: published plausibility tables, the OHDSI Data Quality Dashboard v1.4.1 first
  ([`blacketer-2021`](../../literature.md#gap-detection-and-data-quality)) and the MIMIC-III
  benchmark where the Dashboard rejects a Cleveland record
  ([`harutyunyan-2019`](../../literature.md#gap-detection-and-data-quality)); team-specified where
  no published bound exists, with the reason below.
- **SNOMED-CT evidence**: MONDO's `skos:exactMatch SCTID:<code>` annotation, which the check
  requires on the mapped concept and on no other live concept.
- **Offline concept check**: none in the gate. `python -m bko.ontology check` checks every concept
  row against the real releases.
- **Extra**: `ontology = ["pronto>=2.7"]`, installed by `make setup`; owlready2 leaves the lock.
- **Sources vocabulary**: a row's `source` is the `name` of a `Source` in `bko.datasets.sources`
  (`uci-heart-disease` means `heart-disease.names` inside that archive), a key in
  [`docs/literature.md`](../../literature.md), or an ADR id under `docs/adr/`.

## Rationale

pronto, because the 2-pager names it and OBO is the format both projects release. A
standard-library value layer, as `research/bkb-layer-source` chose for the BKB, keeps pronto out
of every consumer and every test, and removes the order pronto's thread pool would impose.
Constants, as `uci.CODEBOOK` holds the codebook, let every stage import the table and let offline
tests hold it to the codebook and the registry. The releases sit with the other downloads in
`bko.datasets.sources`, the one home for pins.

Guideline cutoffs join the text-named rows because the research tests whether medical knowledge
guides generation, and a cited guideline is that knowledge. Text-named rows alone ground four of
the thirteen predictors, which would reduce the BKO arms to label-by-cell counts. The resting blood
pressure cutoff is the 2017 ACC/AHA guideline's stage 1, 130 mm Hg
([`whelton-2018`](../../literature.md#clinical-definitions)), because it is the current US
guideline; JNC 7's 140 mm Hg ([`chobanian-2003`](../../literature.md#clinical-definitions)) is
kept in the row's limits. The `ca` and `thal` readings come from Detrano 1985
([`detrano-1985a`](../../literature.md#clinical-definitions),
[`detrano-1985b`](../../literature.md#clinical-definitions)) because it is the Cleveland group's
own protocol.

The range bounds take the Dashboard first, because one concept-level table covers most fields.
They take MIMIC-III where the Dashboard rejects a Cleveland record (its 50 to 500 mg/dL rejects a
real cholesterol of 564), since a bound that rejects a real record is wrong for this frame. Age
runs 18 to 120 years because the frame is adults and 120 admits every recorded age. oldpeak runs
-10 to 10 mm because the exercise-testing guideline counts both depression and elevation, and
10 mm is far beyond every UCI site.

## Consequences

**Gained**: an offline gate; rows a person can re-verify against the releases with one command;
no sdist build on a fresh clone.
**Accepted**: the gate cannot catch a concept the release lacks; a re-pin can fail rows; OLS4 is
unavailable; labels are copies of release names (MONDO under CC BY 4.0; HPO under its license,
attributed in `bko.datasets.sources`); the range checks catch gross errors only; with
presence-only rows, a record check derived from `is_a` alone rejects nothing, so record
validation leans on the ranges and the codebook; the guideline cutoffs, the Detrano readings and
the team-specified bounds await clinician review (RESEARCH, Represent); load time has no budget.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| OLS4 for ontology access | Live and unversioned; needs the network at runtime, and the gate is offline | 2026-10-16 |
| owlready2 for ontology access | An OWL quadstore, the lock's only sdist-only package, and no unit needs OWL reasoning | 2026-10-16 |
| A hand-written OBO reader | Re-implements escapes and qualifiers; not a tool the 2-pager names | 2026-10-16 |
| The pins in `bko.ontology` | A second home for downloads | 2026-10-16 |
| A `python -m bko.datasets ontologies` fetch command | No fetch-only command exists; baselines and evaluate call `ensure` themselves | 2026-10-16 |
| Returning `pronto.Ontology` | Leaks pronto into every consumer and test, with thread and hash order | 2026-10-16 |
| A committed TSV or full SSSOM table | No package data exists anywhere, and field states are not CURIEs | 2026-10-16 |
| A markdown table of the rows | A second home, unchecked | 2026-10-16 |
| Absence rows | An asymptomatic history alongside exercise angina is clinically consistent; absence claims need review | 2026-10-16 |
| Text-named rows only | Grounds four of thirteen predictors; the BKO arms would reduce to label-by-cell counts, which `research/bko-formal-gap-definition` evaluates separately | 2026-10-16 |
| Listing all of MONDO and HPO as scope | An unbounded default turns every unmapped concept into a gap | 2026-10-16 |
| Data-derived range bounds | `bko.baselines.records` already refuses added rows outside the split's range, and RESEARCH asks for separately specified constraints | 2026-10-16 |
| SNOMED-CT evidence from xref ids | pronto drops xref qualifiers, so equivalent and related cannot be told apart | 2026-10-16 |
| SNOMED-CT evidence from HPO | HPO carries no SNOMED-CT reference | 2026-10-16 |
| Committed release slices and a manifest for an offline concept check | HPO's license bars altered extracts, and the table changes only on a re-pin | 2026-10-16 |
| Keeping owlready2 in the `ontology` extra | A Cython build on every clone for a library nothing imports | 2026-10-16 |
| One scope for every dataset, the union of the table's concepts | A concept only another dataset's rows map can never be held, so its statements would be gaps of the table, not of the data, against RESEARCH's rule that a concept without a defensible field is out of scope | 2026-10-17 |

## Enforcement

- Owner: `src/bko/ontology/`
- Pinned by: `tests/test_ontology_obo.py`, `tests/test_ontology_imports.py`,
  `tests/test_datasets_sources.py`, `tests/test_ontology_mapping.py`, `tests/test_ontology_check.py`,
  `tests/test_ontology_cli.py`
- `python -m bko.ontology check` is the check the gate cannot make; a person runs it after editing
  the table or re-pinning (`docs/DEVELOPMENT.md`, Ontologies).

## Reversal Conditions

**`bko.ontology.obo` is replaced if pronto cannot load a pinned release, or `uv lock` stops
resolving pronto or fastobo on Python 3.14**; the check is `python -m bko.ontology check` and
`uv lock`.

**owlready2 returns when a unit's acceptance needs OWL reasoning**; the check is a search of
`tasks/` for one.

**OLS4 joins when a unit's acceptance reads a concept outside the pinned releases**; the check is
a search of `tasks/` for one.

**A clinician's verdicts replace the team's choices when a clinician reviews the table**; this
record and the rows are then edited.

**The releases are re-pinned when E1's or E2's configuration names another release**; the check
is a comparison of that configuration with `bko.datasets.sources`.

## Related

- [uv, ruff and ty, wrapped by a Makefile](../tooling/uv-ruff-ty.md)
- [src layout, one subpackage per pipeline stage](../tooling/src-layout.md)
- [The Synthea ablation: a pinned generated population, with whole patients removed](synthea-ablation-design.md)
- [The UCI evaluation frame: Cleveland records, a binary label and a cell-stratified split](uci-evaluation-frame.md)
- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
