---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# The diabetes-codes dataset keeps each ICD-9 diagnosis code and maps it through MONDO's exact xrefs

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
* **Must map a code only where the pinned MONDO release says so**, with the evidence the check
  already reads for ICD-9 rows: an `xref: ICD9:<code>` on a live term.
* **Must keep every mapped code to one concept**, since a field row names one concept.
* **Must be regenerated and compared against the pinned release**, since hundreds of rows cannot
  be curated by hand.

## Decision

1. **A new dataset.** `diabetes-codes` reads the same pinned archive, cohort, label and age edges
   as `diabetes`.
2. **Columns.** Three categorical columns `diag_1`, `diag_2` and `diag_3` take the place of the
   category indicators, under a fixed reversible integer encoding
   (`bko.datasets.diagnoses.code_number`): `?` is 0; a numeric code is 1_000_000 + whole x 1000 +
   tail; a V code 2_000_000 + the same; an E code 3_000_000 + the same. The tail is 0 with no
   decimal, 100 + d with one decimal digit d, and 200 + dd with two digits dd. The codebook is
   built from the loaded cohort's values.
3. **Matching.** A code maps to the MONDO concept that carries its `ICD9:` xref, matched exactly
   after zero-padding the whole part to three digits (`8` is `ICD9:008`).
4. **Holder rule.** Where `ICD9_MAPPINGS` maps the exact code, the curated row's concept wins;
   else the one live holder that every other live holder descends from by `is_a`; else the code
   is unmapped. Obsolete holders are ignored. V and E codes are not looked up.
5. **Field rows.** One row per (mapped code, diagnosis column), each `Codes(diag_k, {number})`,
   then the lab rows of `diabetes`. Each column's rows partition its mapped codes.
6. **A generated table.** `python -m bko.ontology codes` writes the table, committed as
   `bko.ontology.icd9_codes`; `python -m bko.ontology check` checks each row as table
   `icd9-code`, and the `--releases` tests regenerate it and compare. `MAPPED` is not edited, so
   the `elk` scope is unchanged.
7. **Unmapped codes** are counted by reason in Consequences, with the command and date that
   derived the counts.
8. **The `contradictions` census** keeps its three frames (`uci`, `synthea`, `diabetes`).

## Rationale

The exact xref is the evidence the check already reads for the curated ICD-9 rows, so the
generated rows rest on the same release statement and the same check. A code with several live
holders is mapped only where they are nested, because then one of them spans the others and the
choice is the release's own hierarchy; where they are not nested, picking one is a curation call
the generator has no ground for. The curated rows win where they map the exact code, so the two
tables never disagree on a code they share. The integer encoding keeps the frame all int64, as
every reader of a frame expects, and it reads back to the archive's token. One field row per
column keeps each column's rows a partition of its codes, which a field row's `Codes` clause
needs.

## Consequences

**Gained**: a coded frame at the archive's own grain, whose concepts sit in MONDO's hierarchy, for
the ancestor-shrunk share; a table that the `--releases` tests regenerate, so a new release's
disagreement shows as a test failure.
**Accepted**: `bko-direct` draws each concept independently and needs a donor that shows exactly
the drawn set of concepts (`bko.generate.direct`), so on this frame a draw naming two codes for
one diagnosis field has no donor.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| One indicator column per code | Hundreds of binary columns, and a record could hold codes its three fields cannot | 2026-11-24 |
| Category or prefix matching | MONDO states exact xrefs; a prefix match asserts a mapping the release does not make | 2026-11-24 |
| A data file for the rows | It escapes `make sizes` and the Python checks every other table passes | 2026-11-24 |
| Editing `ICD9_MAPPINGS` or the `diabetes` frame | Run folders re-read under identity checks hold both fixed | 2026-11-24 |
| Following `replaced_by` from an obsolete holder | A replacement need not carry the xref; the release no longer states the mapping | 2026-11-24 |
| Choosing between holders that are not nested | Neither spans the other, so the choice is a curation call with no ground in the release | 2026-11-24 |

## Enforcement

- Owner: `src/bko/ontology/icd9.py`, `src/bko/ontology/check.py`
- Pinned by: `tests/test_ontology_icd9.py`

## Reversal Conditions

**A pinned MONDO release that the generator reads differently**: the `--releases` comparison of
the committed rows with the generator's output fails. **A frame that needs a code the table leaves
unmapped**: a later unit names one.

## Related

- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [The Diabetes 130 subgroup-removal design: the full adult frame, one thinned cell at the prevalence cutoff, every arm](diabetes-subgroup-removal-design.md)
