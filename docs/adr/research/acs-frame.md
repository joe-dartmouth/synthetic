---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# ACS PUMS's frame: Massachusetts 2024, income above $50,000, occupation on the Occupation Ontology

## Context

**The survey.** Which coded columns of the ACS person file reach an openly licensed ontology that
converts to OBO? Each column below was looked up in the OBO Foundry registry
([`obofoundry-nd`](../../literature.md#ontologies-reasoners-and-knowledge-graphs), read
2026-11-25) by title, description, domain and tags; a candidate's licence is quoted from the
ontology file's own header, and its conversion was tried with the pinned ROBOT jar and the argv
`bko.ontology.convert` runs.

| Column | ACS name | Ontology looked for | Licence, as its header gives it | Converts to OBO |
|---|---|---|---|---|
| Industry | `INDP`, `NAICSP` | None in the registry | | |
| Field of degree | `FOD1P`, `FOD2P` | None in the registry; OMRSE holds degree classes such as academic degree, not fields of study | | |
| Insurance | `HICOV`, `PRIVCOV`, `PUBCOV`, `HINS1` to `HINS7` | OMRSE, the Ontology for Modeling and Representation of Social Entities, release 2026-09-03, which holds insurance policy and Medicaid policy holder role | `<dcterms:license>https://creativecommons.org/licenses/by/4.0/</dcterms:license>` | No: ROBOT exits 1, "OBO STRUCTURE ERROR ontology cannot be saved in OBO format. Please use '--check true' to see cause."; with `--check true` the cause is "multiple name tags not allowed" on an imported GO term |
| Military service | `MIL` | None in the registry; OMRSE names no military class | | |
| Place of birth | `POBP` | GAZ, the Gazetteer; GEO, the Geographical Entity Ontology | GAZ: the registry's "CC0 1.0" and its Zenodo record's `cc-zero`; GEO: `<terms:license>https://creativecommons.org/licenses/by/4.0/</terms:license>` | No: GAZ's `gaz.obo` header has no `data-version` line, which `bko.ontology.obo` refuses, and reads "This is an alpha version and is only for experimental implementation."; GEO's OWL names two `owl:imports`, which `bko.ontology.convert` refuses before ROBOT runs |
| Class of worker | `COW` | OMRSE, which holds employer role and US Census unemployed role | as for insurance | No, as for insurance |
| Education | `SCHL` | OMRSE, which holds achieved primary education to achieved doctoral or equivalent level education | as for insurance | No, as for insurance |

So no surveyed column is mapped. Industry, field of degree and military service found no
ontology; insurance, place of birth, class of worker and education found one that does not
convert to OBO the loader reads.

**The occupation ontology.** The Occupation Ontology, OccO, release 2025-10-06
([`occo-2025`](../../literature.md#ontologies-reasoners-and-knowledge-graphs)), is OWL only, its
header reads `<terms:license rdf:resource="http://creativecommons.org/licenses/by/4.0/"/>`, it
names no `owl:imports`, and the pinned ROBOT converts it. A class's SOC id is the literal
annotation `OCCO:00000008`: `11-0000` for a major group, `XX-X000` for a minor group, `XX-XXX0`
for a broad occupation, the O*NET id `XX-XXXX.00` for most detailed occupations, and `XX-XXXX`
for a detailed occupation whose O*NET classes are numbered from `.01`, such as `11-3031`. Each
O*NET class also has a job-zone parent.

**The file.** The ACS 2024 1-Year person file for Massachusetts, `psam_p25.csv` inside
`csv_pma.zip`, served from the Census Bureau's 2024 1-Year PUMS directory
([`acs-nd`](../../literature.md#datasets)). The 2024 data dictionary gives `OCCP` as
"Occupation recode for 2018 and later based on 2018 OCC codes", 4 characters, and `SOCP` as
"Standard Occupational Classification (SOC) codes for 2018 and later based on 2018 SOC codes", 6
characters: a SOC code with its hyphen dropped, or an aggregate whose trailing digits are `X`
(such as `15124X`) or `Y` (such as `1721YY`, "ENG-Other Engineers"). `OCCP` is not a SOC code, so
an occupation reaches OccO through the file's own `SOCP` beside it. `SEX` is 1 male, 2 female.
The six disability items are `DEAR` "Hearing difficulty", `DEYE` "Vision difficulty", `DREM`
"Cognitive difficulty", `DPHY` "Ambulatory difficulty", `DDRS` "Self-care difficulty" and
`DOUT` "Independent living difficulty", each 1 yes and 2 no.

**The label.** Folktables' ACSIncome task
([`ding-2021`](../../literature.md#datasets)), read in `folktables/acs.py` at commit `ebace19d`,
filters with `df = df[df['AGEP'] > 16]`, `df = df[df['PINCP'] > 100]`, `df = df[df['WKHP'] > 0]`
and `df = df[df['PWGTP'] >= 1]`, and labels with `target_transform=lambda x: x > 50000` on
`PINCP`. Its features are `AGEP`, `COW`, `SCHL`, `MAR`, `OCCP`, `POBP`, `RELP`, `WKHP`, `SEX` and
`RAC1P`. The 2024 file names the relationship column `RELSHIPP`; it holds no `RELP`.

**The CRC excerpts.** NIST's ACS excerpts hold 24 fields
([`sdnist-2025`](../../literature.md#datasets)): `PUMA`, `AGEP`, `SEX`, `MSP`, `HISP`, `RAC1P`,
`NOC`, `NPF`, `HOUSING_TYPE`, `OWN_RENT`, `DENSITY`, `INDP`, `INDP_CAT`, `EDU`, `PINCP`,
`PINCP_DECILE`, `POVPIP`, `DVET`, `DREM`, `DPHY`, `DEYE`, `DEAR`, `WGTP` and `PWGTP`. They hold
no occupation code and four of the six disability items.

## Decision Drivers

* **Must fix the state-year, the label and the filters before any record is counted**, so no
  figure picks them.
* **Must map an occupation only through the file's own SOC code**, and refuse a file where one
  `OCCP` code carries two.
* **Must map a disability item only to a term whose name names it**, and list the rest as
  unmapped.
* **Must keep ACS and the CRC excerpts apart**, since they share no schema.

## Decision

**1. The state-year.** Massachusetts, ACS 2024 1-Year, fixed here before any count. The pin is
the zip as the Census Bureau serves it, `agreement=None`, as ADR
[`research/confirmatory-dataset-selection`](confirmatory-dataset-selection.md) point 4 fixes, and
the frame follows that ADR's reading of the Census re-identification clause (its point 5).

**2. The cohort and the label.** Folktables' four filters, applied in its order to every row of
the file: `AGEP` above 16, `PINCP` above 100, `WKHP` above 0 and `PWGTP` at least 1; a blank value
meets no filter. The label `target` is 1 when `PINCP` is strictly above 50,000, applied to the
unadjusted `PINCP`, as Folktables applies it; `ADJINC` is not read. `record_id` is a kept row's
position among the kept rows in file order, from 1. Every column is read as text, so `OCCP` keeps
its leading zeros, and no value is read as missing.

**3. The fields.** Folktables' ACSIncome features, with `RELSHIPP` in place of `RELP`, plus the
six disability items: `age` (`AGEP`), `sex` (`SEX`, 1 to 1, 2 to 0), `COW`, `SCHL`, `MAR`,
`OCCP`, `POBP`, `WKHP`, `RAC1P`, `RELSHIPP`, `DEAR`, `DEYE`, `DREM`, `DPHY`, `DDRS` and `DOUT`.
`age` and `WKHP` are numeric; the rest are coded, each with the data dictionary's codes. The age
bands start at 30, 45 and 60. A blank in a used column of a kept row is refused; no field needs a
not-recorded code.

**4. Occupation.** Each `OCCP` code reaches OccO through the `SOCP` code its kept rows carry,
and the loader refuses the file if one `OCCP` code carries two. A `SOCP` code maps by its SOC
level to the class whose `OCCO:00000008` is:

- a detailed code, ending in a digit other than 0: `XX-XXXX.00`, else `XX-XXXX` when no class
  holds `.00`;
- a code ending `0` but not `000`, SOC's broad occupation: `XX-XXX0`;
- a code ending `000`, SOC's minor group: `XX-X000`;
- an aggregate with one trailing `X` or `Y`: its broad class, `XX-XXX0`;
- an aggregate with two or three trailing `X` or `Y`: its minor class, `XX-X000`.

A code no live class holds stays unmapped and is counted by reason. The rows are generated from
the release and the file and committed, as ADR
[`research/diabetes-diagnosis-codes`](diabetes-diagnosis-codes.md) does for ICD-9 codes. A row
holds the `OCCP` code, the SOC id, the class and its name; the check refuses a row whose class is
gone, renamed, or carries another SOC id.

**5. Disability.** An item maps, on code 1, to the live HPO term whose name, or an exact synonym
of it, names the item's difficulty: `DEAR` to Hearing impairment, `DEYE` to Visual impairment,
`DREM` to Cognitive impairment, and `DPHY` to Gait disturbance, whose exact synonyms include
"Difficulty in walking" and which replaced the obsolete term "Difficulty walking". `DDRS` and
`DOUT` are unmapped: no term names self-care or independent living.

**6. Releases.** ACS's catalog entry holds its own release map, HP and OccO
(ADR [`research/ontology-mapping-design`](ontology-mapping-design.md)). OccO is pinned as its
tagged OWL file and converted to OBO at fetch time. A surveyed column whose ontology is openly
licensed and converts to OBO would be mapped as `OCCP` is, with its own pin and release-map entry;
none is.

**7. The CRC excerpts.** The CRC reading is a separate experiment on NIST's own schema (unit
`evaluate-crc-acs-benchmark`): a frame over the excerpt's own 24 fields that shares the rows of
the four disability items it holds. No data is pooled.

**8. The order.** The rows are committed; then the training split
(`split(records, seed=0, edges=frame.edges).train`) is counted; then the gap listing and the
census are read. Consequences record each step's commit.

## Rationale

Folktables' filters and threshold make the label the one ACS fairness work reads, so a later
comparison reads the same task. Applying the threshold to unadjusted income follows Folktables;
one state-year needs no adjustment across years. Massachusetts is one of the excerpts' targets,
`MA`, so the two experiments read the same state.

Mapping an occupation through `SOCP` uses the Census Bureau's own crosswalk on each record rather
than a crosswalk written here. Reading a `SOCP` code by its SOC level keeps each code's meaning:
a broad occupation such as `472040` names the broad class, not a detailed one. An aggregate's
trailing `X` or `Y` marks the SOC digits it leaves open, so it maps to the narrowest class that
holds every code it may stand for.

A disability item is a yes or no to a difficulty, and the HPO terms named for an impairment of
that faculty say the same of a person. Gait disturbance is HPO's term for difficulty in walking,
and NIST's dictionary calls `DPHY` "Ambulatory (walking) difficulty".

## Consequences

**The order, by commit.** The rows are at `107c2e5` (Add the ACS loader, frame, field rows and
catalog entry). The training split's count is recorded in `836f4f3`, before any gap listing or
census ran: `CATALOG["acs"].read` on the data root, then `split(records, seed=0,
edges=frame.edges).train`. The file holds 74,617 rows; Folktables' filters keep 63,191 above age
16, 58,029 with income above $100, 42,543 with usual hours above 0, and all 42,543 with a person
weight of at least 1. The cohort holds 42,543 records, 24,896 of them with income above $50,000;
the training split holds 34,035 records, 19,918 of them above $50,000, a rate of 0.5852, digest
`b5d8bd0639ab25c6e3ca45e8ec810f9cf7c4a3f418de7af9f1935d8beee862ba`. The split is above the 5,000
records condition 5 asks. Each of the 517 `OCCP` codes among the kept rows carries one `SOCP`
code, and no used column is blank among them. The readings below are recorded in `db62ab4`.
These are commits of pull request #136, kept at
`refs/pull/136/head` after its squash merge.

**The readings, at `836f4f3`, after the count.** `uv run --no-sync python -m bko.ontology check`
exited 0, its log ending `checked 609 rows: 0 disagree`, OccO converted by the pinned ROBOT. The
gap listing is run `d1-5`, `uv run --no-sync python -m bko.gaps acs` into
`runs/d1-5/gaps-acs.csv` and `runs/d1-5/gaps-acs.log`, exit 0, closing `gaps rows=10111
sha256=530bd629acc8870dbfb18fc5692ba927ce66a2edb5ca9f03889688150e03dbfd`: per-subgroup 8,320 gaps
in 16 cells, population-level 176, frequency-aware 1,615 in 16 cells, and no cell below the count
threshold; 511 of the 515 mapped concepts are held in the training split. Its csv sha256 is
`530bd629acc8870dbfb18fc5692ba927ce66a2edb5ca9f03889688150e03dbfd` and its log sha256
`9fd2a2aa21ba0e276a320ac02a24c5b6be112e3f832f230107414d6c4d265cf4`; its `data` line gives the
42,543 records' digest `1dd0fbe777cdcdfcbb16a42f412a3d37afd6cb6093e7efd096949b4e7f6226e6` and its
`split` line the test fingerprint
`d2d2760fd6745a51963c83ac5f01d96fc207dd7e0563bc33d489bdf3d00f74ff`. The census is run `d2-5`,
`uv run --no-sync python -m bko.evaluate contradictions --dataset acs` into
`runs/d2-5/contradictions.csv` and `runs/d2-5/contradictions.log`, exit 0, over 16 of 16 cells,
520 rows and 247 `is_a` pairs: `narrow-without-broad` 1,791, `child-share-above-parent` 841,
`empty-cell` 0, `all-show` 0 and `code-overlap` 0. Every pair is one occupation class under
another the rows also map, and every record that shows the narrower class lacks the broader one,
since a record holds one `OCCP` code and a row maps a code to one class, not to its ancestors. Its
csv sha256 is `2270e982928841ed92f067d7605d4f61c949207f1f044e1256bde20586303872` (2,632 rows)
and its log sha256 `d5f7c1cf6f150b3fa6e34de33cf27e69728c911e18977191fe4665c932e50287`. Both read
the archive sha256 `bb7d4009feaf8d190d687fc091fc7a56e2aa82b9972ae14a16d4ad70a8e0eef7`, HPO
`hp/releases/2026-09-01` and OccO `releases/2025-10-06`.

**The occupation rows.** 516 of the 517 `OCCP` codes map; `9830`, `SOCP` `559830`
("MIL-Military, Rank Not Specified"), reaches no class and is the one `no class` code, with none
for `several classes`. 221 rows sit in the low part and 295 in the high one.

**Gained**: a non-health frame whose occupation code reaches a five-level hierarchy, from SOC
major group to O*NET occupation, through rows the check keeps against the release. **Accepted**:
no surveyed column beyond occupation is mapped; the income label is unadjusted; the census reads
every occupation pair as a contradiction, because a row maps an `OCCP` code to its own class and
leaves its ancestors to the hierarchy.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Mapping `OCCP` codes to SOC through a crosswalk written here | The file carries `SOCP` beside `OCCP` on every record; a written crosswalk would be a second home | 2026-11-25 |
| Reading `PINCP` adjusted by `ADJINC` | Folktables applies its threshold to unadjusted `PINCP` | 2026-11-25 |
| `RELP` as Folktables names it | The 2024 file has no `RELP`; `RELSHIPP` is its relationship column | 2026-11-25 |
| Mapping education, class of worker or insurance through OMRSE | The pinned ROBOT cannot write OMRSE as OBO | 2026-11-25 |
| Mapping place of birth through GAZ or GEO | GAZ's OBO file has no `data-version`, and GEO's OWL names imports | 2026-11-25 |
| Impairment of activities of daily living for `DDRS` or `DOUT` | Its name names neither self-care nor independent living | 2026-11-25 |
| Pooling ACS with the CRC excerpts | The excerpts hold no occupation code and another schema | 2026-11-25 |

## Enforcement

- Owner: `src/bko/ontology/acs_rows.py`
- Pinned by: `tests/test_datasets_acs.py::test_the_label_is_income_strictly_above_50000`,
  `tests/test_datasets_acs.py::test_the_frame_holds_exactly_the_named_fields`,
  `tests/test_datasets_acs.py::test_occp_keeps_its_leading_zeros`,
  `tests/test_evaluate_acs.py::test_every_occp_code_is_in_exactly_one_row`,
  `tests/test_evaluate_acs.py::test_each_disability_item_with_an_exact_hpo_term_maps_to_it`,
  `tests/test_evaluate_acs.py::test_acs_reads_its_own_releases`,
  `tests/test_ontology_occupation_codes.py::test_a_detailed_soc_code_maps_to_its_onet_00_class`,
  `tests/test_ontology_occupation_codes.py::test_a_five_digit_aggregate_maps_to_its_broad_class`,
  `tests/test_ontology_occupation_codes.py::test_a_shorter_aggregate_maps_to_its_minor_class`,
  `tests/test_ontology_occupations.py::test_an_occupation_row_whose_soc_id_differs_fails_the_check`,
  `tests/test_datasets_acs.py::test_each_folktables_filter_drops_its_rows`,
  `tests/test_datasets_acs.py::test_occp_to_socp_must_be_a_function`,
  `tests/test_ontology_occupation_codes_pinned.py` and `tests/test_datasets_acs_pinned.py` (with
  `--releases`)

## Reversal Conditions

**OMRSE, GAZ or GEO publishes a release the pinned ROBOT converts and the loader reads**: the
column it covers is then mapped as `OCCP` is. **OccO re-numbers a SOC id or drops a class**, which
the check reports. **The Census Bureau re-issues the 2024 file** with other bytes. **The training
split falls below 5,000 records**, which ADR `research/confirmatory-dataset-selection` condition
5 excludes.

## Related

- [Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page](confirmatory-dataset-selection.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [SUPPORT2's frame: in-hospital death as the label, physiology coded at SIRS and SOFA cutoffs](support2-frame.md)
