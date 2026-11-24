---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# The diabetes-codes dataset keeps each ICD-9 diagnosis code and maps it through MONDO's equivalentTo xrefs

## Context

The `diabetes` frame holds one indicator per curated ICD-9 category of `bko.ontology.icd9`
(`ICD9_MAPPINGS`), so its hierarchy holds a handful of concepts. The archive's three diagnosis
fields (`diag_1`, `diag_2`, `diag_3`) hold far more distinct codes than that table has rows. A
share that shrinks a narrow code toward its ancestors needs a coded field at that finer grain.
The `diabetes` frame's rows and draw are held fixed by run folders that are re-read under identity
checks, so a finer mapping is a new dataset name, never an edit to `diabetes`
([`tasks/direct-index.md`](../../../tasks/direct-index.md)).

## Decision Drivers

* **Must leave the `diabetes` frame, `ICD9_MAPPINGS` and `MAPPED` byte for byte**, since E6's and
  E10's folders are re-read under identity checks and the `elk` comparison reads `MAPPED`.
* **Must map a code only where the pinned MONDO release says the code and the concept are
  equivalent**: an `xref: ICD9:<code>` on a live term that MONDO's curators qualify as
  equivalence.
* **Must keep every mapped code to one concept**, since a field row names one concept.
* **Must be regenerated from the pinned release**, since the cohort's hundreds of codes cannot be
  curated by hand.

## Decision

1. **A new dataset.** `diabetes-codes` reads the same pinned archive, cohort, label and age edges
   as `diabetes`.
2. **Columns.** Three categorical columns `diag_1`, `diag_2` and `diag_3` take the place of the
   category indicators, under a fixed reversible integer encoding
   (`bko.datasets.diagnoses.code_number`): `?` is 0; a numeric code is 1_000_000 + whole x 1000 +
   tail; a V code 2_000_000 + the same; an E code 3_000_000 + the same. The tail is 0 with no
   decimal, 100 + d with one decimal digit d, and 200 + dd with two digits dd. The codebook is
   built from the loaded cohort's values.
3. **Matching.** A code maps to the MONDO concept that carries its `ICD9:` xref under the
   qualifier `source="MONDO:equivalentTo"` (an exact holder), the code matched exactly after
   zero-padding the whole part to three digits (`8` is `ICD9:008`). An xref under any other
   qualifier maps nothing. `MONDO:relatedTo` marks a curator-reviewed link weaker than
   equivalence: it has no direction, so it cannot assert that a record holding the code shows
   the concept. An xref whose only `source` names another ontology's class (DOID, EFO and
   others) or that carries no qualifier was imported unreviewed and carries known errors: 575
   (diseases of the gallbladder) sits on intestinal disorder, and 720 (ankylosing spondylitis and
   other inflammatory spondylopathies) on autoimmune disease. A code MONDO links only these
   weaker ways stays a free field of `diabetes-codes`. Decided with the user (Joe Laws) on
   2026-11-24.
4. **Holder rule.** Where `ICD9_MAPPINGS` maps the exact code, the curated row's concept wins,
   and it must be one of the code's live exact holders; else the one live exact holder that every
   other live exact holder descends from by `is_a`; else the code is unmapped. Obsolete holders
   are ignored. V and E codes are not looked up.
5. **Field rows.** One row per (mapped code, diagnosis column), each `Codes(diag_k, {number})`,
   then the lab rows of `diabetes`. Each column's rows partition its mapped codes.
6. **A generated table.** `python -m bko.ontology codes` writes the table from the
   `diabetes-codes` cohort's codes, committed as `bko.ontology.icd9_codes`; it is regenerated,
   never edited. `MAPPED` is not edited, so the `elk` scope is unchanged.
7. **Unmapped codes** are counted by reason in Consequences, with the command and date that
   derived the counts.
8. **The `contradictions` census** keeps its three frames (`uci`, `synthea`, `diabetes`).

## Rationale

An equivalentTo xref is the one ICD-9 statement in MONDO that a curator has reviewed as
equivalence, so a generated row asserts no more than the release does. The curated rows read any
qualifier and stay as they are, since run folders hold them fixed. A code with several live
holders is mapped only where they are nested, because then one of them spans the others and the
choice is the release's own hierarchy; where they are not nested, picking one is a curation call
the generator has no ground for. The curated rows win where they map the exact code, so the two
tables never disagree on a code they share. The integer encoding keeps the frame all int64, as
every reader of a frame expects, and it reads back to the archive's token. One field row per
column keeps each column's rows a partition of its codes, which a field row's `Codes` clause
needs.

## Consequences

**Gained**: a coded frame at the archive's own grain, whose concepts sit in MONDO's hierarchy, for
the ancestor-shrunk share; a table one command regenerates from the pinned release.
**The release's ICD-9 xrefs**: of the pinned release's 5,658 `xref: ICD9:` lines, 1,859 carry
`MONDO:equivalentTo`, 2,378 `MONDO:relatedTo` (one written `MONDO:RelatedTo`), 1,386 a source
only (1,254 naming only outside sources such as DOID, EFO or GARD, and 132 no qualifier), 30 another
MONDO source (29 `MONDO:obsoleteEquivalent`, 1 `MONDO:mondoIsNarrowerThanSource`), and 5
`MONDO:i2s` alone (`grep '^xref: ICD9:'` over the pinned `mondo.obo`, snapshot
`2631e8e165d41def`, MONDO `releases/2026-10-06`, 2026-11-24).
**Unmapped codes**: of the `diabetes-codes` cohort's 889 distinct diagnosis codes, 32 map; 555
have no live xref, 2 have obsolete holders only, 3 are linked by `relatedTo` only, 170 by an
unreviewed source only (169 naming an outside source, 1 with no qualifier), none have exact
holders that are not nested, and 47 V codes and 80 E codes are not looked up
(`uv run --no-sync python -m bko.ontology codes`, MONDO `releases/2026-10-06`, 2026-11-24). A code
whose live xrefs carry no `MONDO:equivalentTo` counts as `relatedTo only` when one carries
`MONDO:relatedTo`, else as `unreviewed source only`.
**`bko-direct` on this frame**: `bko-direct` draws each concept independently and needs a donor
that shows exactly the drawn set of concepts (`bko.generate.direct`), so a draw naming two codes
for one diagnosis field can find no donor. Measured on 2026-11-24 over the 32 rows:
`bko.generate.direct.sample` on the `diabetes-codes` training split at seed 0, nothing removed,
for E6's removal cell `(6-7, male, 1)` with a quota of 100 at seed 0, draws 100 rows, none through
a fallback draw. The same day the `rows` check re-read run `e10-pram-thina`'s folder `e10-42` on
the `diabetes` frame and ended `matches_folder=true`, its store's sha256 `3d015cb23749164c99c70ae54783eae1fb6057748cf85d6112a471dfd33d93fe` (commit `117e8f7`).

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| One indicator column per code | Hundreds of binary columns, and a record could hold codes its three fields cannot | 2026-11-24 |
| Category or prefix matching | MONDO states exact xrefs; a prefix match asserts a mapping the release does not make | 2026-11-24 |
| A data file for the rows | It escapes `make sizes` and the Python checks every other table passes | 2026-11-24 |
| Editing `ICD9_MAPPINGS` or the `diabetes` frame | Run folders re-read under identity checks hold both fixed | 2026-11-24 |
| Following `replaced_by` from an obsolete holder | A replacement need not carry the xref; the release no longer states the mapping | 2026-11-24 |
| Choosing between holders that are not nested | Neither spans the other, so the choice is a curation call with no ground in the release | 2026-11-24 |
| Accepting `MONDO:relatedTo` xrefs | A reviewed link weaker than equivalence, with no direction, cannot assert that a record holding the code shows the concept | 2026-11-24 |
| Accepting source-only xrefs | Imported from DOID, EFO and others without MONDO review, with known errors such as 575 on intestinal disorder and 720 on autoimmune disease | 2026-11-24 |

## Enforcement

- Owner: `src/bko/ontology/icd9.py`, `src/bko/ontology/check.py`, `src/bko/ontology/codes.py`,
  `src/bko/ontology/icd9_codes.py`, `src/bko/datasets/diagnoses.py`,
  `src/bko/ontology/code_frames.py`, `src/bko/evaluate/data.py`
- Pinned by: `tests/test_ontology_icd9.py`, `tests/test_ontology_codes.py` (the equivalentTo
  rule and the unmapped reasons, the holder rule, the curated concept among the exact holders, one
  concept per code, the `icd9-code` check of a row against its exact holders, the committed
  module equal to `render`'s text, the command reading the cohort's codes),
  `tests/test_ontology_obo.py` (the loader keeps each xref's qualifiers),
  `tests/test_datasets_diagnoses.py` (the encoding), `tests/test_ontology_icd9_codes_pinned.py`
  (under `--releases`: the committed rows equal to the generator's output on the pinned release,
  the unmapped tallies above, every field row's code in the frame's codebook, each concept live
  and holding its xref under `MONDO:equivalentTo`), `python -m bko.ontology check` (each row as
  table `icd9-code`),
  `tests/test_evaluate_diabetes_codes.py` (the frame keeps each code, the cohort and label of
  `diabetes`, each column's rows partition the mapped codes, the `diabetes` rows unchanged),
  `tests/test_contradictions_census.py` (the census reads its three frames)

## Reversal Conditions

**A pinned MONDO release that the generator reads differently**: the `--releases` comparison of
the committed rows with the generator's output fails. **A frame that needs a code the table leaves
unmapped**: a task unit's acceptance names a code that the unmapped tallies above count, or
`python -m bko.ontology codes` on a newly pinned MONDO release maps a code of the cohort that
this table leaves unmapped.

## Related

- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [The Diabetes 130 subgroup-removal design: the full adult frame, one thinned cell at the prevalence cutoff, every arm](diabetes-subgroup-removal-design.md)
