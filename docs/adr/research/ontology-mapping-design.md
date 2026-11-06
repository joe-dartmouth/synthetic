---
status: accepted
topic: research
created: 2026-10-16
updated: 2026-11-05
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
  Measured on 2026-10-27: the Synthea disorder codes of essential hypertension, type 2 diabetes,
  hyperlipidemia, myocardial infarction, ST-elevation myocardial infarction and stroke each sit on
  one live MONDO concept; the code of acute non-ST-elevation myocardial infarction sits on none.
- **MONDO carries ICD-9 references only as `xref: ICD9:<code>` lines**, never as a
  `skos:exactMatch`. Measured on 2026-11-02: an ICD-9 xref's qualifier takes several shapes (a
  `source` naming another ontology's class, or `MONDO:equivalentTo` or `MONDO:relatedTo` with
  `MONDO:i2s`, or none), and pronto keeps the id only. Of the Diabetes 130 diagnosis categories
  the table maps, 428 sits on heart failure and congestive heart failure and 491 on bronchitis and
  chronic bronchitis; each other sits on one live concept. No live concept carries ICD9:414.
- **HPO carries no SNOMED-CT reference.**
- **pronto read both releases**, with every term converted to standard-library values, no import
  and no warning.

The unit `compare-elk-logic-only` asks whether ELK, a classifier for EL+bottom (EL with the bottom
concept) within the OWL 2 EL profile, derives subclass links the loaded hierarchy lacks, or the
reverse. MONDO's release build runs ELK through ROBOT, a Java command-line tool distributed as a
jar ([`mondo-2026`](../../literature.md#ontologies-reasoners-and-knowledge-graphs)); HPO's release
build runs no reasoner, builds `hp.owl` as a merge of the editors' file and builds `hp.obo` from a
different artefact, its simple non-classified release
([`hpo-2026`](../../literature.md#ontologies-reasoners-and-knowledge-graphs)).

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
  `bko.ontology.schema`. The range constraints belong to a frame, so they are defined in
  `bko.datasets` (`RangeConstraint` in `bko.datasets.frames`, UCI's `RANGES` in
  `bko.datasets.uci`), and `bko.ontology.schema` and `bko.ontology.mapping` re-export them.
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
- **Synthea code rows**: a Synthea disorder code maps to the one live MONDO concept that carries
  its `skos:exactMatch`, applied mechanically: ST-elevation myocardial infarction maps because its
  code has one, beside myocardial infarction. `CODE_MAPPINGS` holds ischemic heart disease and six
  rows for hypertension, type 2 diabetes, hyperlipidemia, myocardial infarction, ST-elevation
  myocardial infarction and stroke. A disorder code of these named conditions with no such
  concept is an `UnmappedCode` row in `UNMAPPED_CODES` with its reason, and the check refuses one
  that gains a live `skos:exactMatch`; the population's other codes with no match, such as type 2
  diabetes complications, have no row. No finding or situation code joins.
- **Synthea field rows**: the Synthea frame's guidance reads field rows derived from
  `CODE_MAPPINGS`, not a second table (ADR
  [`research/synthea-downstream-design`](synthea-downstream-design.md)). Each code row other than
  ischemic heart disease becomes a `FieldMapping` with the clause `Codes(<its indicator column>,
  {1})` and the row's concept, label and source, and ischemic heart disease becomes the label row,
  `Codes(target, {1})`. `bko.evaluate.data.SYNTHEA_FIELD_ROWS` holds them.
- **No out-of-scope row above a mapped concept**: once type 2 diabetes maps, diabetes mellitus
  (MONDO:0005015) is its `is_a` ancestor and so in scope, and it has no out-of-scope row. The
  reason no UCI field grounds it, that `fbs` is one reading, sits in the `fbs` row's limits, which
  name the concept; UCI's scope takes the field mappings' concepts only, so it still excludes
  diabetes mellitus.
- **Diabetes 130 rows**: `bko.ontology.icd9` holds them. An ICD-9 row (`ICD9_MAPPINGS`) maps a
  diagnosis category to a live MONDO concept that carries the category's `xref: ICD9:<code>`, and
  declares every live concept carrying that xref; where two do, the row maps the one that spans
  the category's subcodes, and its limits name the other. A category left out because no live
  concept carries its xref is an `UnmappedIcd9` row in `UNMAPPED_ICD9`, and the check refuses one
  that gains a holder. A lab row (`LAB_MAPPINGS`) maps the archive's result states of
  `A1Cresult` and `max_glu_serum`, verbatim, to an HPO phenotype. The rows' concepts join the
  table's scope.
- **Range bounds**: published plausibility tables, the OHDSI Data Quality Dashboard v1.4.1 first
  ([`blacketer-2021`](../../literature.md#gap-detection-and-data-quality)) and the MIMIC-III
  benchmark where the Dashboard rejects a Cleveland record
  ([`harutyunyan-2019`](../../literature.md#gap-detection-and-data-quality)); team-specified where
  no published bound exists, with the reason below. UCI's bounds are `bko.datasets.uci.RANGES`.
- **SNOMED-CT evidence**: MONDO's `skos:exactMatch SCTID:<code>` annotation, which the check
  requires on the mapped concept and on no other live concept, and on no live concept for an
  excluded code.
- **ICD-9 evidence**: MONDO's `xref: ICD9:<code>` with that exact id and any qualifier, which the
  check requires on the mapped concept; the release's live holders of the xref must equal the
  row's declared holders, and an excluded category must have none.
- **ELK comparison**: `bko.ontology.elk` runs the pinned ROBOT jar with ELK on the pinned OWL
  files of the same MONDO and HPO releases, as a Java subprocess outside the gate, and compares its
  classification with the hierarchy the OBO releases give (`python -m bko.ontology elk`), as
  ancestor pairs a class holds on one side only. ROBOT runs `reason` with
  `--equivalent-classes-allowed all`, so an inferred equivalence is kept rather than stopping the
  run, then `convert --check false`, so the OBO writer skips its document-structure checks rather
  than refusing a class they reject. The pins are `ROBOT_JAR`, `MONDO_OWL` and `HPO_OWL` in
  `bko.datasets.sources`. No Python module parses OWL; `bko.ontology.elk.read_hierarchy` reads
  ROBOT's classified OBO output (Ruled Out, the hand-written reader's carve-out), an
  `equivalent_to:` line as `is_a` both ways. An HPO pair that differs can come from the build as
  well as from classification: `hp.owl` and `hp.obo` are built from different artefacts.
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

ELK runs through ROBOT because that is how MONDO's release build runs it, so the comparison uses
the reasoner and the OWL reader that built MONDO's hierarchy. A subprocess keeps Java out of the
Python process and out of the gate, as the Synthea run already does, and a Python OWL parser
would be a second reader whose disagreements with the OWL API would read as hierarchy
differences.

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
the team-specified bounds await clinician review (RESEARCH, Represent); an ICD-9 xref asserts no
equivalence, so the choice among a category's holders is the team's; load time has no budget; the
ELK comparison needs a JDK and a 16 GB Java heap (`bko.ontology.elk.ELK_MEMORY_GB`), and the gate
cannot catch a change in ROBOT's or ELK's output.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| OLS4 for ontology access | Live and unversioned; needs the network at runtime, and the gate is offline | 2026-10-16 |
| owlready2 for ontology access | An OWL quadstore, and the lock's only sdist-only package | 2026-10-16 |
| A hand-written OBO reader | Re-implements escapes and qualifiers; not a tool the 2-pager names. One carve-out: `bko.ontology.elk.read_hierarchy` reads ROBOT's classified OBO output, not a release, because pronto refuses the classified HPO file (a `SyntaxError` on an imported term's cross-reference `[Pathbase: Curation]`); it reads only the `id:`, `is_a:` and `equivalent_to:` lines of term stanzas (carve-out added 2026-11-05) | 2026-10-16 |
| The pins in `bko.ontology` | A second home for downloads | 2026-10-16 |
| A `python -m bko.datasets ontologies` fetch command | No fetch-only command exists; baselines and evaluate call `ensure` themselves | 2026-10-16 |
| Returning `pronto.Ontology` | Leaks pronto into every consumer and test, with thread and hash order | 2026-10-16 |
| A committed TSV or full SSSOM table | No package data exists anywhere, and field states are not CURIEs | 2026-10-16 |
| A markdown table of the rows | A second home, unchecked | 2026-10-16 |
| Absence rows | An asymptomatic history alongside exercise angina is clinically consistent; absence claims need review | 2026-10-16 |
| Text-named rows only | Grounds four of thirteen predictors; the BKO arms would reduce to label-by-cell counts, which `research/bko-formal-gap-definition` evaluates separately | 2026-10-16 |
| Listing all of MONDO and HPO as scope | An unbounded default turns every unmapped concept into a gap | 2026-10-16 |
| Data-derived range bounds | `bko.baselines.records` already refuses added rows outside the split's range, and RESEARCH asks for separately specified constraints | 2026-10-16 |
| Evidence from an xref's qualifier, for SNOMED-CT or ICD-9 | pronto drops xref qualifiers, so equivalent and related cannot be told apart; SNOMED-CT rows read `skos:exactMatch` instead, and ICD-9 rows, whose only MONDO evidence is the xref id, declare every live holder so the check sees each one | 2026-11-02 |
| SNOMED-CT evidence from HPO | HPO carries no SNOMED-CT reference | 2026-10-16 |
| Committed release slices and a manifest for an offline concept check | HPO's license bars altered extracts, and the table changes only on a re-pin | 2026-10-16 |
| Keeping owlready2 in the `ontology` extra | A Cython build on every clone for a library nothing imports | 2026-10-16 |
| One scope for every dataset, the union of the table's concepts | A concept only another dataset's rows map can never be held, so its statements would be gaps of the table, not of the data, against RESEARCH's rule that a concept without a defensible field is out of scope | 2026-10-17 |
| The release tests inside `make check` | The gate reads no download, and a clone without `data/` would fail them or skip them on a condition that reads as a pass | 2026-10-17 |
| Defining the range constraints in `bko.ontology` | Validation and generation read them, and neither is an ontology stage; they belong to the frame in `bko.datasets` | 2026-10-26 |
| Mapping acute non-ST-elevation myocardial infarction to a broader concept, such as acute myocardial infarction (MONDO:0004781) | A broader concept is not an equivalence; the code rows hold `skos:exactMatch` only | 2026-10-27 |
| Leaving ST-elevation myocardial infarction out because myocardial infarction already maps | The rule is mechanical; a judgment per code would need a source the release does not give | 2026-10-27 |
| An out-of-scope row for diabetes mellitus | Once type 2 diabetes maps it is an is_a ancestor and in scope; the check refuses an out-of-scope row above a mapped concept | 2026-10-27 |
| Parsing the OWL releases in Python, with owlready2 or an RDF library, for the ELK comparison | A second OWL reader beside the OWL API the release builds use, and owlready2 builds an sdist; see Rationale | 2026-11-05 |

## Enforcement

- Owner: `src/bko/ontology/`
- Pinned by: `tests/test_ontology_obo.py`, `tests/test_ontology_imports.py`,
  `tests/test_datasets_sources.py`, `tests/test_ontology_mapping.py`, `tests/test_ontology_check.py`,
  `tests/test_ontology_icd9.py`, `tests/test_ontology_cli.py`, `tests/test_ontology_elk.py` (the
  comparison and ROBOT's argv, with a fake runner), and `tests/test_ontology_pinned_releases.py`
  with `--releases`
- `python -m bko.ontology check` is the check the gate cannot make, and
  `tests/test_ontology_pinned_releases.py` asserts it over the real releases when `--releases` is
  passed; a person runs one of them after editing the table, re-pinning, or changing
  `bko.ontology.obo` or pronto (`docs/DEVELOPMENT.md`, Ontologies).

## Reversal Conditions

**`bko.ontology.obo` is replaced if pronto cannot load a pinned release, or `uv lock` stops
resolving pronto or fastobo on Python 3.14**; the check is
`tests/test_ontology_pinned_releases.py` with `--releases`, and `uv lock`.

**A Python OWL library returns when a unit's acceptance needs OWL axioms read inside `make check`,
where no test runs Java**; the check is a search of `tasks/` for an acceptance that names OWL and
`make check`.

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
- [The Synthea downstream design: a 40-and-over frame, one thinned cell and an in-run reference](synthea-downstream-design.md)
- [The UCI evaluation frame: Cleveland records, a binary label and a cell-stratified split](uci-evaluation-frame.md)
- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
