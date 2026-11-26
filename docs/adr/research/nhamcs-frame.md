---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# NHAMCS 2022 ED's frame: hospital admission as the label, ICD-10-CM diagnoses and chronic flags on MONDO

## Context

ADR [`research/confirmatory-dataset-selection`](confirmatory-dataset-selection.md) chose NHAMCS
2022 ED as a health dataset whose concepts come from coded diagnoses. It quotes the ED
documentation ([`nhamcs-nd`](../../literature.md#datasets)): "Each record on the ED file
represents one visit in the sample of 16,025 visits."; `ADMITHOS` is "Admit to this hospital";
`AGE` is in years, top-coded at "94 = 94 years or older"; `SEX` is coded 1 female, 2 male. It
records the NCHS terms the file falls under and, in its point 5, the team's reading of the clause
on assessing disclosure methods: the privacy readings score the rows a method writes, measure that
method's disclosure, and identify no person, so they lie outside the clause.

NCHS serves the public-use file as `ed2022.zip` from
`https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS/`, with the Stata dictionary
`eddict2022.dct` and the documentation `doc22-ed-508.pdf` under
`https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/`, all read
2026-11-25. The archive holds one member, `ed2022`: fixed-width ASCII, one visit per line, 2,382
characters and a CRLF per line. The dictionary gives each item's columns (1-based, inclusive), and
the documentation's item list gives the same columns:

| Item | Columns | Codes |
|---|---|---|
| `AGE` | 16-18 | 0 under 1 year, 1 to 93, 94 "94 years or older" |
| `SEX` | 25 | 1 Female, 2 Male |
| `DIAG1` to `DIAG5` | 122-125, 126-129, 130-133, 134-137, 138-141 | ICD-10-CM, four characters |
| `ETOHAB` to `SUBSTAB` | 152 to 173, one column each | 0 No, 1 Yes |
| `NOCHRON` | 174 | 0 No, 1 Yes, 2 "Entire item blank including “None” box" |
| `ADMITHOS` | 499 | 0 No, 1 Yes |
| `HOSPCODE` | 544-546 | "A unique code assigned to all the records from a particular hospital." |
| `PATCODE` | 547-549 | "A number assigned to identify each individual record from a particular hospital." |

The documentation describes the diagnosis fields: "For the NHAMCS ED public use file, only the
first four digits of the diagnosis codes are included. There is an implied decimal between the
third and fourth digits and inapplicable 4th digits are dash-filled." Its examples are `F321`,
F32.1, and `I10-`, I10. Beside the codes, `-9` is "Field is blank (in contrast to an actual entry
of “Blank”)", written `-9` and two spaces, and `ZZZ0`, `ZZZ2` to `ZZZ5` are NCHS's own codes for
an entry that is no diagnosis, such as "Noncodable diagnosis, insufficient information for coding,
illegible diagnosis".

The chronic items sit under "DOES PATIENT HAVE:", one checkbox per condition, in this order:
`ETOHAB` "Alcohol misuse, abuse, or dependence", `ALZHD` "Alzheimer’s disease/dementia", `ASTHMA`
"Asthma", `CANCER` "Cancer", `CEBVD` "Cerebrovascular disease/history of stroke (CVA) or transient
ischemic attack (TIA)", `CKD` "Chronic kidney disease (CKD)", `COPD` "Chronic obstructive pulmonary
disease (COPD)", `CHF` "Congestive heart failure", `CAD` "Coronary artery disease (CAD), ischemic
heart disease (IHD), or history of myocardial infarction (MI)", `DEPRN` "Depression", `DIABTYP1`
"Diabetes mellitus (DM) – Type I", `DIABTYP2` "Diabetes mellitus (DM) – Type II", `DIABTYP0`
"Diabetes mellitus (DM) – Type unspecified", `ESRD` "End-stage renal disease (ESRD)", `HPE`
"History of pulmonary embolism (PE) or deep vein thrombosis (DVT), or venous thromboembolism
(VTE)", `EDHIV` "HIV infection/AIDS", `HYPLIPID` "Hyperlipidemia", `HTN` "Hypertension", `OBESITY`
"Obesity (BMI >= 30)", `OSA` "Obstructive sleep apnea (OSA)", `OSTPRSIS` "Osteoporosis" and
`SUBSTAB` "Substance abuse or dependence". `NOCHRON` is "None of the above". A quotation is the
documentation's text as `pdftotext -layout` reads it, whitespace runs collapsed to one space.

The diagnosis rows follow ADR [`research/diabetes-diagnosis-codes`](diabetes-diagnosis-codes.md),
which maps Diabetes 130's ICD-9 codes to MONDO through exact cross-references only.

## Decision Drivers

* **Must fix the label and every row before any count is read**, so no figure picks them.
* **Must map a diagnosis only where MONDO's curators state equivalence**, as the ICD-9 rows do.
* **Must let an unanswered chronic item show no concept**, so a visit whose item is blank shows
  neither a condition nor its absence.
* **Must keep the visit's outcome out of the fields**: nothing recorded at or after the
  disposition enters.

## Decision

The pin is one `bko.datasets.sources` entry, the archive as NCHS serves it, `agreement=None`, as
ADR `research/confirmatory-dataset-selection` point 4 fixes; its size equals the host's
content-length, 1,666,841 bytes, and its sha256 is measured from the bytes served.

**1. The records.** Every line of `ed2022` is one record; a line other than 2,382 characters is
refused, naming its line. `record_id` is `HOSPCODE` x 1000 + `PATCODE`, so a record's id does not
depend on its line; a repeated id is refused. Every visit enters: all ages, no filter.

**2. The label.** `target` is `ADMITHOS`, 1 admitted to this hospital, 0 otherwise. `age` is
`AGE` as published, top-coded at 94, so 94 means 94 and over; `sex` is `SEX`, 1 to 0 (female) and
2 to 1 (male). Age bands split at 18, 45 and 65.

**3. The diagnosis fields.** `DIAG1` to `DIAG5` each hold the four-character token as an integer:
blank (`-9`) is 0, and any other token is its value as a four-digit base-37 number over the
digits `-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`, so `-` is digit 0 and the encoding is reversible.
`A00-` and `A000` are different numbers. A token that is not blank, not `ZZZ0` and `ZZZ2` to
`ZZZ5`, and not a letter, a digit, then two characters of digits and letters with a dash allowed
last, is refused. `ZZZ1` is refused with them: the documentation lists it, on its page 19, under
"This code was previously discontinued:", and its `DIAG1` entry, on page 41, says it "was actually
discontinued in 2018". The codebook is built from the loaded records' tokens.

**4. Matching a diagnosis.** A token reads as an ICD-10-CM id by restoring the implied decimal:
`I509` is `ICD10CM:I50.9`, and a dash-filled `A00-` is the category `ICD10CM:A00`. Blank and the
`ZZZ` tokens read as no id. The id maps to the MONDO concept that carries it under the qualifier
`source="MONDO:equivalentTo"`, as Decisions 3 and 4 of ADR `research/diabetes-diagnosis-codes` do
for ICD-9: an xref under any other qualifier maps nothing; obsolete holders are ignored; with
several live exact holders, the one every other descends from by `is_a` maps, else none. There
are no curated rows. A four-character token stands for every code below its truncation, so a
token maps only to the concept MONDO equates with that truncated id, never to a longer code's.
`python -m bko.ontology codes --system icd10cm` writes the rows from the loaded records' tokens,
committed as a generated module and regenerated, never edited; the unmapped tokens are counted by
reason.

**5. The diagnosis rows.** One row per (mapped token, diagnosis column), each
`Codes(DIAGk, {number})`, so each column's rows partition its mapped tokens and a token maps the
same concept in any position.

**6. The chronic flags.** The 22 items enter as fields: 0 no, 1 yes, and 2 not recorded where
`NOCHRON` is 2, the entire item blank. A record whose `NOCHRON` contradicts its flags, 0 with no
flag, 1 or 2 with a flag, is refused. A flag maps, on code 1, to the live MONDO concept whose
name or exact synonym is the condition its item names, matched ignoring case and the item's
abbreviation; where MONDO holds none, to HPO's. An item naming several conditions maps to the one
every other descends from by `is_a` in MONDO, and is dropped where there is none. The rows, labels
the pinned releases' names:

| Item | Concept | Read through |
|---|---|---|
| `ALZHD` | MONDO:0001627 dementia | Alzheimer disease descends from dementia |
| `ASTHMA` | MONDO:0004979 asthma | name |
| `CANCER` | MONDO:0004992 cancer | name |
| `CEBVD` | MONDO:0011057 cerebrovascular disorder | exact synonym; stroke disorder and transient ischemic attack descend from it |
| `CKD` | MONDO:0005300 chronic kidney disease | name |
| `COPD` | MONDO:0005002 chronic obstructive pulmonary disease | name |
| `CHF` | MONDO:0005009 congestive heart failure | name |
| `DEPRN` | MONDO:0002050 depressive disorder | exact synonym depression |
| `DIABTYP1` | MONDO:0005147 type 1 diabetes mellitus | exact synonym type I diabetes mellitus |
| `DIABTYP2` | MONDO:0005148 type 2 diabetes mellitus | exact synonym type II diabetes mellitus |
| `DIABTYP0` | MONDO:0005015 diabetes mellitus | name; the type is unspecified |
| `ESRD` | MONDO:0004375 end stage renal failure | exact synonym end stage renal disease |
| `EDHIV` | MONDO:0005109 HIV infectious disease | exact synonym HIV infection; AIDS descends from it |
| `HYPLIPID` | MONDO:0021187 hyperlipidemia | name |
| `HTN` | HP:0000822 Hypertension | HPO's name; MONDO holds no exact term |
| `OBESITY` | MONDO:0011122 obesity disorder | exact synonym obesity |
| `OSA` | MONDO:0007147 obstructive sleep apnea syndrome | exact synonym obstructive sleep apnea |
| `OSTPRSIS` | MONDO:0005298 osteoporosis | name |

`ETOHAB`, `CAD`, `HPE` and `SUBSTAB` enter as fields and map nothing (Ruled Out).

**7. What never enters.** The disposition block, columns 487 to 502, but `ADMITHOS`; `ADMIT`,
columns 503-504, through `BOARDED`, columns 2379-2382, every item recorded once the patient is
admitted or leaves, but `HOSPCODE` and `PATCODE`, columns 544-549, which are read only to form
`record_id` and enter no field; `PATWT` and every other weight. No item the frame does not name
enters.

**8. The order.** The rows are committed; then the training split (`split(records, seed=0,
edges=frame.edges).train`) is counted, its record count, its admissions and their rate; then the
gap listing and the census are read. Consequences record each step's commit. A training split
below 5,000 records excludes the frame until ADR `research/confirmatory-dataset-selection` is
revised.

**9. Every reading.** The NCHS reading of ADR `research/confirmatory-dataset-selection` (points 4
and 5) lets every reading run on this frame, the `bko` arm and the privacy readings included.
This frame fits and scores nothing; a later unit runs them.

## Rationale

The label is one checkbox of the visit's own record, so the frame reduces nothing to make it.
Admission follows the visit's diagnoses, which the provider writes before the disposition; every
item recorded at or after the disposition stays out, so no field restates the label.

The diagnosis rule is the ICD-9 rule because its reasons hold unchanged: an exact cross-reference
is the only one with a direction that lets a record holding the code show the concept. The file's
truncation only shortens the codes: the id a token reads as is a real ICD-10-CM category or
subcategory, and MONDO's xrefs name such ids. Keying the base-37 number on the dash keeps a
category and its first subcategory apart. The dash does not say the code stops at three
characters: the `DIAG1` entry reads "for inapplicable or masked fourth digits, a dash is inserted",
so a dashed token may hide a fourth digit. The row reads it as the category it names, which covers
every code below it, as Decision 4 reads every truncated token.

Coding a blank chronic item as its own state keeps "no" meaning the box was left unticked on a
form whose item was answered. The flag rows take a term whose name or exact synonym is the item's
words, and the broader of nested conditions, so a row never asserts more than the box says.

## Consequences

**The order, by commit.** The rows are at `a1ce476` (Add the NHAMCS frame, field rows and catalog
entry), the generated ICD-10-CM rows at `a6a08e1` before it. The training split's count is
recorded in the commit after `a1ce476`, before any gap listing or census ran:
`CATALOG["nhamcs"].read` on the data root, then `split(records, seed=0, edges=frame.edges).train`,
as this command does:

    uv run --no-sync python -c 'from bko.datasets.fetch import default_data_dir, open_https; from bko.datasets.frames import digest; from bko.datasets.split import split; from bko.ontology.catalog import CATALOG; records, frame = CATALOG["nhamcs"].read(default_data_dir(), open_url=open_https); train = split(records, seed=0, edges=frame.edges).train; print(len(records), int(records["target"].sum()), len(train), int(train["target"].sum()), round(float(train["target"].mean()), 4), digest(train))' </dev/null

Run at `a1ce476`, the loader's `cohort` line reads `rows=16025 records=16025 admitted=2121
chronic_blank=259`: every one of the 16,025 visits is a record, 2,121 of them admitted, and 259
with the chronic item blank. The training split holds 12,823 records, 1,698 of them admitted, a
rate of 0.1324, digest `faecee0ac1a59afaeceaf4f1ce2e42bd9e06479f36f05935d3ea6f0ba2eb939b`. The
split is above the 5,000 records condition 5 of ADR `research/confirmatory-dataset-selection`
asks.

**The diagnosis rows.** `python -m bko.ontology codes --system icd10cm` at `a6a08e1`'s code, over
MONDO `releases/2026-10-06`, read 2,114 distinct tokens other than blank and mapped 178, its log
closing `codes system=icd10cm rows=178 tokens=2114
sha256=0ea05ce957168cbfa512d854090bf80ab835666eefdb375fff935d5cc7b66fac`. The unmapped tokens by
reason: no live xref 1,920, obsolete holder only 0, relatedTo only 1, unreviewed source only 11,
holders not nested 0, and not ICD-10-CM 4, the `ZZZ` tokens. The same release holds 1,520 distinct
`ICD10CM:` ids under `MONDO:equivalentTo` of the shape a token reads as. The flag rows map 18 of
the 22 chronic items.

**The readings, at `90ffff7`, after the count.** `uv run --no-sync python -m bko.ontology check`
exited 0, its log ending `checked 810 rows: 0 disagree`. The gap listing is run `d1-8`, `uv run
--no-sync python -m bko.gaps nhamcs` into `runs/d1-8/gaps-nhamcs.csv` and
`runs/d1-8/gaps-nhamcs.log`, exit 0, closing `gaps rows=11476
sha256=3a34ca90f0f3767e685aea135bd1dbadc05589604f19d6b4abe980cf479a0f64`: per-subgroup 9,864
rows, population-level 62 gaps in each of the 16 cells, frequency-aware 620 rows, and no cell
below the count threshold of 20. Its csv sha256 is
`3a34ca90f0f3767e685aea135bd1dbadc05589604f19d6b4abe980cf479a0f64` and its log sha256
`35973a0ed50c86e5e5ae1e97365bee82abbf50d20e3fd239a499798249fa004d`; its `data` line gives the
16,025 records' digest `ac84a442710ad8a09a90ec0c1c6f29e12630e78ab99714ddec50e50b92ce44e0` and its
`split` line the test fingerprint
`83ec4ee9084ee19540a5d544a65c910bc3e1b8684ddf7afbd967c3b03553a86b`. The census is run `d2-8`, `uv
run --no-sync python -m bko.evaluate contradictions --dataset nhamcs` into
`runs/d2-8/contradictions.csv` and `runs/d2-8/contradictions.log`, exit 0, over 16 of 16 cells, 908
field rows and 29 `is_a` pairs among the mapped concepts: `census kind=narrow-without-broad` 126,
`child-share-above-parent` 91, `empty-cell` 0, `all-show` 0 and `code-overlap` 0. Most of the
narrow-without-broad findings are the chronic checklist's own structure: a visit ticking type 2
diabetes, 921 training records, need not also tick the type-unspecified box, which maps diabetes
mellitus; end stage renal failure, 146 records, shows without chronic kidney disease in 58. Its
csv sha256 is `150d189b8f8e6bf16fc88f76722928fb86a6677d6927c99e00e1347a23037b17` and its log
sha256 `ce41e63922327a7e6c1889b817a9a69f72b4886d74504cf064d5f5b90a0e4984`. Both read the pinned
archive, MONDO `releases/2026-10-06` and HPO `hp/releases/2026-09-01`, and each log's `versions`
line reads `dirty=false`. Runs `d1-9` and `d2-9` read them again at `9584f87`, once the loader's
`cohort` line counted blank first diagnoses and NCHS's `ZZZ` tokens: each csv is byte-identical to
its earlier run's, and each log's `cohort` line reads `records=16025 admitted=2121
chronic_blank=259 first_blank=227 not_icd=265`. These are commits of pull request #139, kept at
`refs/pull/139/head` after its squash merge; [`EVIDENCE.md`](../../EVIDENCE.md) keeps the runs'
rows.

**Gained**: a health frame whose concepts come from coded diagnoses and a chronic-condition
checklist, reaching MONDO through curator-stated equivalence. **Accepted**: diagnoses MONDO links
only weakly, and the four-character truncation of longer codes, leave tokens unmapped; they stay
free fields. Age 94 means 94 and over.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Mapping a token to the concept of a longer code it may stand for | The file keeps four characters; a truncated code covers every code below it, and the record does not say which | 2026-11-25 |
| Mapping a diagnosis through a `MONDO:relatedTo` or unreviewed xref | Neither states equivalence, as ADR `research/diabetes-diagnosis-codes` records for ICD-9 | 2026-11-25 |
| `ETOHAB` on alcohol abuse or alcohol dependence | MONDO holds no term for misuse, and alcohol abuse and alcohol dependence do not nest | 2026-11-25 |
| `CAD` on coronary artery disorder | The item also names ischemic heart disease and myocardial infarction; MONDO nests none of the three under another | 2026-11-25 |
| `HPE` on venous thromboembolism | MONDO holds no exact term for deep vein thrombosis, and pulmonary embolism does not descend from venous thromboembolism there | 2026-11-25 |
| `SUBSTAB` on substance abuse or substance dependence | The two do not nest in MONDO | 2026-11-25 |
| `HTN` on MONDO's hypertensive disorder | Its name and exact synonyms do not include hypertension; HPO's Hypertension is the item's word | 2026-11-25 |
| Reading a blank chronic item as no | It would put visits whose form left the item unanswered among those without the condition | 2026-11-25 |
| The line number as `record_id` | It changes with the file's order; `HOSPCODE` and `PATCODE` identify the visit | 2026-11-25 |
| Weighting records by `PATWT` | The frame scores records, not national estimates | 2026-11-25 |

## Enforcement

- Owner: `src/bko/ontology/nhamcs_rows.py`
- Pinned by: `tests/test_ontology_icd10cm.py::test_a_four_character_token_reads_as_a_dotted_code`,
  `tests/test_ontology_icd10cm.py::test_a_padded_token_reads_as_its_category`,
  `tests/test_ontology_icd10cm.py::test_only_exact_cross_references_map`,
  `tests/test_ontology_icd10cm.py::test_the_token_number_is_reversible_and_distinct`,
  `tests/test_datasets_nhamcs.py::test_the_positions_neither_overlap_nor_run_past_the_record`,
  `tests/test_datasets_nhamcs.py::test_admithos_is_the_label`,
  `tests/test_datasets_nhamcs.py::test_the_disposition_and_stay_fields_never_enter`,
  `tests/test_datasets_nhamcs.py::test_the_frame_holds_exactly_the_named_fields`,
  `tests/test_datasets_nhamcs.py::test_a_visit_with_the_chronic_item_blank_shows_no_flag_concept`,
  `tests/test_evaluate_nhamcs.py::test_each_diagnosis_columns_rows_partition_its_mapped_codes`,
  `tests/test_evaluate_nhamcs.py::test_the_flag_rows_map_these_items_on_code_1`,
  `tests/test_ontology_icd10cm_codes_pinned.py` and `tests/test_datasets_nhamcs_pinned.py` (with
  `--releases`)

## Reversal Conditions

**NCHS re-issues `ed2022.zip`** with other bytes. **MONDO adds or drops an exact ICD-10-CM
cross-reference** on a token the records hold, which the regenerated rows would show. **The
training split falls below 5,000 records**, which ADR `research/confirmatory-dataset-selection`
would then revise.

## Related

- [Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page](confirmatory-dataset-selection.md)
- [The diabetes-codes dataset keeps each ICD-9 diagnosis code and maps it through MONDO's equivalentTo xrefs](diabetes-diagnosis-codes.md)
- [SUPPORT2's frame: in-hospital death as the label, physiology coded at SIRS and SOFA cutoffs](support2-frame.md)
- [NHANES's frame: adults 20 and over, a cardiovascular composite label, labs coded at cited cutoffs](nhanes-frame.md)
