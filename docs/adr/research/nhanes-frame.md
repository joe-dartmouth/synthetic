---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# NHANES's frame: adults 20 and over, a cardiovascular composite label, labs coded at cited cutoffs

## Context

ADR [`research/confirmatory-dataset-selection`](confirmatory-dataset-selection.md) chose NHANES
August 2021 to August 2023 as the general-population health dataset, and bounded the composite
outcome's size from the Medical Conditions codebook (MCQ_L,
[`nhanes-2024`](../../literature.md#datasets)): among adults 20 and over, congestive heart failure
(`MCQ160b`) 343 yes answers, coronary heart disease (`MCQ160c`) 402, heart attack (`MCQ160e`) 332
and stroke (`MCQ160f`) 367, each of roughly 7,800 answers. The union of the four so holds between
402 and 1,444 adults. That ADR also records the NCHS terms the files fall under
([`nchs-nd`](../../literature.md#datasets)) and its reading of them.

The survey publishes one SAS XPORT file per component, each row one respondent, joined on `SEQN`,
the respondent sequence number. Each file is served from
`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/` beside its codebook page, read
2026-11-25. The codebooks give:

- DEMO_L: `RIDAGEYR` "Age in years of the participant at the time of screening. Individuals 80
  and over are top-coded at 80 years of age."; `RIAGENDR` 1 Male, 2 Female; `RIDSTATR` 1
  "Interviewed only", 2 "Both interviewed and MEC examined".
- MCQ_L: each of the four items asks "Has a doctor or other health professional ever told
  {you/SP} that {you/s/he} . . .had" the condition, for "Both males and females 20 YEARS - 150
  YEARS", coded 1 Yes, 2 No, 7 Refused, 9 Don't know, and missing. The codebook writes the names
  with a lower-case letter, `MCQ160b`; the file's own names are upper case, `MCQ160B`.
- BPXO_L: three oscillometric readings, systolic `BPXOSY1` to `BPXOSY3` and diastolic `BPXODI1`
  to `BPXODI3`, mm Hg.
- GHB_L: `LBXGH` "Glycohemoglobin (%)"; its code 0, "No Lab Result", holds no record.
- TCHOL_L: `LBXTC` "Total Cholesterol (mg/dL)"; HDL_L: `LBDHDD` "Direct HDL-Cholesterol (mg/dL)".
- BMX_L: `BMXBMI` "Body Mass Index (kg/m**2)".

A mapped lab or vital needs a cited cutoff, read in its source before the row is written. Two of
the three guideline sources could not be read; each cutoff is read in an open record that
reproduces it, as ADR [`research/support2-frame`](support2-frame.md) reads SIRS and SOFA in
Sepsis-3:

- **Blood pressure.** The 2017 ACC/AHA guideline
  ([`whelton-2018`](../../literature.md#clinical-definitions)) sits behind a bot check at its
  publisher on every route tried, a browser included, and Europe PMC lists no PubMed Central
  copy. Muntner et al. ([`muntner-2018`](../../literature.md#clinical-definitions)), read from
  PubMed Central on 2026-11-25, are seven of the guideline's writing committee, Whelton among
  them; their Table 1,
  "Blood pressure levels used to define hypertension, recommend antihypertensive medication, and
  treatment goal according to the 2017 ACC/AHA guideline, the JNC7 guideline and the JNC8 panel
  member report.", gives the 2017 ACC/AHA definition for the general population as systolic "≥
  130" and diastolic "≥ 80" mm Hg. They read NHANES as this frame does: "For each participant,
  blood pressure was measured three times following a standardized protocol and averaged."
- **Glycated haemoglobin.** The ADA's Standards of Care
  ([`ada-2026`](../../literature.md#clinical-definitions)), read from PubMed Central on
  2026-11-25: Table 2.1 "A1C ≥6.5% (≥48 mmol/mol)" for diabetes, Table 2.2 "A1C 5.7–6.4% (39–47
  mmol/mol)" for prediabetes. Its footnote to Table 2.1 asks for two abnormal results to
  diagnose diabetes.
- **Cholesterol.** The JAMA executive summary of ATP III
  ([`ncep-2001`](../../literature.md#clinical-definitions)) is paywalled. The NHLBI publishes the
  panel's executive summary as NIH Publication No. 01-3670, May 2001
  ([`nhlbi-2001`](../../literature.md#clinical-definitions)), read 2026-11-25; its Table 2, "ATP
  III Classification of LDL, Total, and HDL Cholesterol (mg/dL)", gives total cholesterol "≥240"
  "High" and HDL cholesterol "<40" "Low".

A quotation is the page's text with markup dropped and whitespace runs collapsed to one space.

## Decision Drivers

* **Must fix the label before any count is read**, so no figure picks it.
* **Must keep the label free of leakage**: the four questions that make it never enter as fields.
* **Must map a lab or vital only through a cited cutoff**, read in its source's text.
* **Must let a missing value show no concept**, so an adult the exam never measured shows neither
  a finding nor its absence.
* **Must fix the rows before the training split is counted**, and count the split before any gap
  is read.

## Decision

The pins are seven `bko.datasets.sources` entries, one per file, the loader
`bko.datasets.nhanes`, the frame `bko.datasets.nhanes_frame` and the rows
`bko.ontology.nhanes_rows`.

**1. The pins.** DEMO_L, MCQ_L, BPXO_L, BMX_L, TCHOL_L, HDL_L and GHB_L, each the `.xpt` file as
NCHS serves it, `agreement=None`, as ADR `research/confirmatory-dataset-selection` point 4 fixes.
Each size equals the host's content-length, and each sha256 is measured from the bytes served.
The fasting-subsample files are left out.

**2. The join.** Each file is read whole from its verified bytes by
`pandas.read_sas(..., format="xport")`; its columns must be the published file's, in order, and
its `SEQN` values whole and unique. DEMO_L and MCQ_L join inner on `SEQN`; BPXO_L, BMX_L,
TCHOL_L, HDL_L and GHB_L join left, so an adult with no exam or no blood sample stays a record.
`record_id` is `SEQN`.

**3. The adults.** A record is kept when `RIDAGEYR` is 20 or more: the universe the four questions
are asked of. `age` is `RIDAGEYR` as published, top-coded at 80, so an 80 reads 80 and means 80
and over. `sex` is `RIAGENDR`, 1 to 1 (male) and 2 to 0 (female).

**4. The label**, fixed here before any count. `target` is 1 when any of `MCQ160B`, `MCQ160C`,
`MCQ160E` and `MCQ160F` is 1 (yes); 0 when all four are 2 (no); otherwise the record is left out.
So an adult with no yes and any refused, don't-know or missing answer is left out, and a yes
counts whatever the other three answers are. The four questions, the angina item `MCQ160D` and
the survey weights never enter.

**5. The fields.** `age` and `sex`, then six coded fields, each `0` = not measured, else `1` plus
the number of "at or above" cutoffs at or below the value, every one an "at or above" cutoff:

| Field | Read from | Cutoffs | Source |
|---|---|---|---|
| `BPXOSY` | the mean of `BPXOSY1` to `BPXOSY3` | 130 | `whelton-2018`, read in `muntner-2018` |
| `BPXODI` | the mean of `BPXODI1` to `BPXODI3` | 80 | `whelton-2018`, read in `muntner-2018` |
| `LBXGH` | `LBXGH` | 5.7, 6.5 | `ada-2026` |
| `LBXTC` | `LBXTC` | 240 | `ncep-2001`, read in `nhlbi-2001` |
| `LBDHDD` | `LBDHDD` | 40 | `ncep-2001`, read in `nhlbi-2001` |
| `BMXBMI` | `BMXBMI` | the file's tertiles | none (below) |

A blood pressure is the mean of the readings present; with none it is not measured. Each value,
and each mean, is read at 2 decimals before it is coded. `BMXBMI` has no cited cutoff here, so it
is coded at the tertiles of its measured values over every record of BMX_L, each rounded to 2
decimals, and maps nothing.

**6. The rows.** One `Codes` row per mapped abnormal state, to HPO, labels the pinned release's
names: `BPXOSY` at least 130 Elevated systolic blood pressure; `BPXODI` at least 80 Elevated
diastolic blood pressure; `LBXGH` at least 5.7, both of its upper states, Elevated hemoglobin
A1c; `LBXTC` at least 240 Hypercholesterolemia; `LBDHDD` below 40 Decreased circulating HDL-C
concentration. A blood-pressure or lipid row's source is the reproduction read, and its limits
name the original.

**7. The order.** The rows are committed; then the training split
(`split(records, seed=0, edges=frame.edges).train`, age bands from 40 and 60) is counted, its
record count and its composite count; then the gap listing and the census are read.
Consequences record each step's commit.

**8. The bounds.** The composite over the whole adult frame must lie in 402 to 1,444, the bounds
ADR `research/confirmatory-dataset-selection` derives from the codebook's whole-file counts, and
the training split must hold at least 5,000 records (its condition 5). Either failing excludes
the frame until that ADR is revised.

**9. Every reading.** The NCHS reading of ADR `research/confirmatory-dataset-selection` (points 4
and 5) lets every reading run on this frame, the `bko` arm and the privacy readings included.
This frame fits and scores nothing; a later unit runs them.

## Rationale

Condition 2 of the selection rule excludes a label the frame would have to reduce from a finer
one, such as a per-hour sepsis flag read off as one per stay. The composite reduces nothing: each
of the four items is the respondent's own answer, once per person, and the union is 1 exactly
when the respondent reported any of the four. No finer record is collapsed to make it.

Leaving out a record with no yes and an unknown answer keeps 0 meaning four noes: counting a
don't-know as no would put an adult who may have one of the conditions among the negatives. A
yes stands whatever the rest say, since one yes makes the union 1.

Reading the guideline cutoffs in records written by the guideline's own committee, or published
by the panel's institute, is the closest readable text to the originals. Averaging the readings
present follows Muntner et al.'s reading of NHANES. One `LBXGH` row on both upper states keeps
the row a finding, not a diagnosis: the ADA asks for two abnormal results before diabetes is
diagnosed, and the file holds one.

## Consequences

**Gained**: a general-population health frame whose blood pressure, glycated haemoglobin and
lipids reach HPO through cutoffs read in their sources. **Accepted**: two of the three guideline
originals stay unread; a re-read that differs from its reproduction would change a row. The
label leaves out adults whose answers cannot settle it. Age 80 means 80 and over.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Reading the 2017 ACC/AHA cutoffs in the guideline itself | Its publisher's pages and PDF refused every route behind a bot check, a browser included, and Europe PMC lists no PubMed Central copy; Muntner et al. reproduce the definition | 2026-11-25 |
| Reading the ATP III cutoffs in the JAMA executive summary | Paywalled; the NHLBI's own executive summary of the panel's report is open | 2026-11-25 |
| Counting a don't-know, refused or missing answer as no | It would put adults who may hold a component among the negatives | 2026-11-25 |
| Mapping `LBXGH` at least 6.5 to diabetes mellitus | One result; the ADA asks for two abnormal results to diagnose diabetes | 2026-11-25 |
| Mapping `LBXGH` 5.7 to 6.4 to MONDO's prediabetes syndrome | One result, as for diabetes; Elevated hemoglobin A1c names the finding | 2026-11-25 |
| Hypertension, `HP:0000822`, on either blood-pressure field | HPO defines it as chronic increased pressure, which one visit's readings do not show | 2026-11-25 |
| `BMXBMI` at a cited obesity cutoff | No registered source for a BMI cutoff is read here; it is coded at tertiles and maps nothing | 2026-11-25 |
| Weighting records by the survey weights | The frame scores records, not population estimates; the unit leaves weights out | 2026-11-25 |
| The fasting-subsample laboratory files | No field of the frame reads them | 2026-11-25 |

## Enforcement

- Owner: `src/bko/ontology/catalog.py`
- Pinned by: nothing -- convention only

## Reversal Conditions

**A read of the 2017 ACC/AHA guideline or the JAMA ATP III summary differs from its
reproduction** in a cutoff used here. **NCHS re-issues a component file** with other bytes. **The
training split falls below 5,000 records, or the composite over the adult frame falls outside 402
to 1,444**, which ADR `research/confirmatory-dataset-selection` would then revise.

## Related

- [Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page](confirmatory-dataset-selection.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [SUPPORT2's frame: in-hospital death as the label, physiology coded at SIRS and SOFA cutoffs](support2-frame.md)
- [ACS PUMS's frame: Massachusetts 2024, income above $50,000, occupation on the Occupation Ontology](acs-frame.md)
