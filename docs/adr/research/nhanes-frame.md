---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-26
---
# NHANES's frame: adults 20 and over, a cardiovascular composite label, labs coded at cited cutoffs

## Context

ADR [`research/confirmatory-dataset-selection`](confirmatory-dataset-selection.md) chose NHANES
August 2021 to August 2023 as the general-population health dataset, and bounded the composite
outcome's size from the Medical Conditions codebook (MCQ_L,
[`nhanes-2024`](../../literature.md#datasets)): among adults 20 and over, congestive heart failure
(`MCQ160b`) 343 yes answers, coronary heart disease (`MCQ160c`) 402, heart attack (`MCQ160e`) 332
and stroke (`MCQ160f`) 367, each of roughly 7,800 answers. The adults with any of the four number at
least the largest count, 402, and at most the sum of the four, 1,444. That ADR also records the NCHS
terms the files fall under ([`nchs-nd`](../../literature.md#datasets)) and its reading of them.

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
  publisher on every route tried, a browser included, and Europe PMC lists no PubMed Central copy.
  Muntner et al. ([`muntner-2018`](../../literature.md#clinical-definitions)), read from PubMed
  Central on 2026-11-25, reproduce its definition. Their Table 1, "Blood pressure levels used to
  define hypertension, recommend antihypertensive medication, and treatment goal according to the
  2017 ACC/AHA guideline, the JNC7 guideline and the JNC8 panel member report.", gives the 2017
  ACC/AHA definition for the general population as systolic "≥ 130" and diastolic "≥ 80" mm Hg. They
  read NHANES as this frame does: "For each participant, blood pressure was measured three times
  following a standardized protocol and averaged."
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
`pandas.read_sas(..., format="xport")`, a warning that the file may be corrupted refusing it;
its columns must be the published file's, in order, its `SEQN` values whole and unique, and every
other file's `SEQN` values among DEMO_L's. DEMO_L and MCQ_L join inner on `SEQN`; BPXO_L, BMX_L,
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
| `BMXBMI` | `BMXBMI` | the kept records' tertiles | none (below) |

A blood pressure is the mean of the readings present; with none it is not measured. Each value,
and each mean, is read at 2 decimals before it is coded. `BMXBMI` has no cited cutoff here, so it
is coded at the tertiles of its measured values over the records the frame keeps, the adults of
20 and over whose answers settle the label, each rounded to 2 decimals, and maps nothing, each
tertile an "at or above" cutoff. BMX_L's codebook gives `BMXBMI` from age 2, and its children's
values stay out of the tertiles. `bko.datasets.nhanes.tertile_base` returns the kept records'
measured `BMXBMI` values, and the loader logs one `tertile dataset=nhanes field=BMXBMI low=<c>
high=<c> base=<n>` line before its `cohort` line, `base` the kept records measured in it:

| Field | Lower tertile | Upper tertile | Base |
|---|---|---|---|
| `BMXBMI` | 25.9 | 31.6 | 5,946 |

This command derived them from the pinned files in the data root, printing each field, its base
count and its two tertiles:

    uv run --no-sync python -c 'from bko.datasets.fetch import default_data_dir, read_verified; from bko.datasets.nhanes import COMPONENTS, DECIMALS, read_component, tertile_base; from bko.datasets.sources import NHANES; from bko.datasets.states import tertiles; base = tertile_base({name: read_component(read_verified(source, default_data_dir()), name) for name, source in zip(COMPONENTS, NHANES, strict=True)}); [print(name, len(values), *(round(c, DECIMALS) for c in tertiles(values))) for name, values in base.items()]' </dev/null

At tag `run/d1-12` it printed `BMXBMI 5946 25.9 31.6`. Over every record of BMX_L, children
included, the tertiles would be 23.4 and 29.6; this command derived those from the pinned file:

    uv run --no-sync python -c 'from bko.datasets.fetch import default_data_dir, read_verified; from bko.datasets.nhanes import DECIMALS, TERTILES, measured, read_component; from bko.datasets.sources import NHANES_BMX_L; from bko.datasets.states import tertiles; raw = read_component(read_verified(NHANES_BMX_L, default_data_dir()), "BMX_L"); [print(name, *(round(c, DECIMALS) for c in tertiles(measured(raw[name])))) for name in TERTILES]' </dev/null

At the tag it printed `BMXBMI 23.4 29.6`.

`pandas.read_sas` reads a stored 0 as 5.397605346934028e-79: DEMO_L holds 189 such ages, the
infants of age 0, and reading at 2 decimals turns each back to 0. An `LBXGH` of 0, the codebook's
"No Lab Result", is not measured; the published file holds none. This command counted the ages:

    uv run --no-sync python -c 'from bko.datasets.fetch import default_data_dir, read_verified; from bko.datasets.nhanes import read_component; from bko.datasets.sources import NHANES_DEMO_L; raw = read_component(read_verified(NHANES_DEMO_L, default_data_dir()), "DEMO_L"); age = raw["RIDAGEYR"]; print(int((age < 1e-6).sum()), int((age == 0).sum()), repr(age.min()))' </dev/null

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

Reading the guideline cutoffs in a table that states them according to the 2017 ACC/AHA guideline,
or in the executive summary the panel's institute publishes, is the closest readable text to the
originals. Averaging the readings present follows Muntner et al.'s reading of NHANES. One `LBXGH`
row on both upper states keeps the row a finding, not a diagnosis: the ADA asks for two abnormal
results before diabetes is diagnosed, and the file holds one.

Taking `BMXBMI`'s tertiles over the records the frame keeps cuts it where the coded adults' values
fall, so each tertile state holds a third of the records it codes; over every record of BMX_L,
children's lower values would pull both cuts down.

## Consequences

**The order, by commit.** The rows are at `be72ef5` (Add the NHANES loader, frame, field rows and
catalog entry). The training split's count is recorded in the commit after it, before any gap
listing or census ran: `CATALOG["nhanes"].read` on the data root, then `split(records, seed=0,
edges=frame.edges).train`, as this command does:

    uv run --no-sync python -c 'from bko.datasets.fetch import default_data_dir, open_https; from bko.datasets.frames import digest; from bko.datasets.split import split; from bko.ontology.catalog import CATALOG; records, frame = CATALOG["nhanes"].read(default_data_dir(), open_url=open_https); train = split(records, seed=0, edges=frame.edges).train; print(len(records), int(records["target"].sum()), len(train), int(train["target"].sum()), round(float(train["target"].mean()), 4), digest(train))' </dev/null

Run at `31b8c6f`, the loader's `cohort` line reads `demographics=11933 with_conditions=11744
adults=7809 any_yes=937 all_no=6832 left_out=40 records=7769 composite=937 measured_BPXOSY=5839
measured_BPXODI=5839 measured_LBXGH=5744 measured_LBXTC=5476 measured_LBDHDD=5476
measured_BMXBMI=5946`: 11,744 respondents hold a medical-conditions record, 7,809 of them are 20 and
over, and 40 adults are left out with no yes and an unknown answer. The adult frame holds 7,769
records, 937 with the composite, inside the 402 to 1,444 the selection ADR bounds. The training
split holds 6,215 records, 750 of them with the composite, a rate of 0.1207, digest
`9b1553fa494ea6d9d10b6f2f115837714523cc10819c738efc6ceaad2c00163c`. The split is above the 5,000
records condition 5 asks. The readings below are recorded in the commit after the count's. These are
commits of pull request #137, kept at `refs/pull/137/head` after its squash merge.

**The readings, at `31b8c6f`, after the count.** `uv run --no-sync python -m bko.ontology check`
exited 0 at `db2cc4f`, its log ending `checked 614 rows: 0 disagree`. The gap listing is run
`d1-7`, `uv run --no-sync python -m bko.gaps nhanes` into `runs/d1-7/gaps-nhanes.csv` and
`runs/d1-7/gaps-nhanes.log`, exit 0, closing `gaps rows=63
sha256=9e86c72e13a9a51081505de299539f102cdd7da48e48d65a89aa9a8e934b1979`: per-subgroup 11 gaps in
one cell, `<40/female/1`, whose 10 records hold one of the five mapped concepts; population-level
none; frequency-aware 50 in 7 cells; and two cells below the count threshold of 20,
`<40/female/1` with 10 records and `<40/male/1` with 9. Its csv sha256 is
`9e86c72e13a9a51081505de299539f102cdd7da48e48d65a89aa9a8e934b1979` and its log sha256
`51c17cf09f58a40c5e657b7fcd4e10f357d69753678d68f154979a7f2c5b8648`; its `data` line gives the
7,769 records' digest `602b6ed761f5afe22b0ce58d95c3c987a75d6d0834dbabdba268bace5b3e11ad` and its
`split` line the test fingerprint
`66905e31f88bee07a0ee5586d12b50b6d3876fe05ac29136e9b20bb0a81bd510`. The census is run `d2-7`,
`uv run --no-sync python -m bko.evaluate contradictions --dataset nhanes` into
`runs/d2-7/contradictions.csv` and `runs/d2-7/contradictions.log`, exit 0: every `census kind=`
count is 0 (`narrow-without-broad`, `child-share-above-parent`, `empty-cell`, `all-show`,
`code-overlap`), over 12 of 12 cells and 5 rows, with no `is_a` pair among the mapped concepts.
Its csv sha256 is `b8fbd7f8c95b38fa93262f7e8f92dfd28ed305a3e784342d1b32db9707aa0056` (the header
alone) and its log sha256 `de0e62bdc35e7aeaac96cee9b322402f64a82d09532e681c699adc3ad9be150c`.
Both read the seven pinned files and HPO `hp/releases/2026-09-01`, and each log's `versions` line
reads `dirty=false`. Each csv is byte for byte the one runs `d1-6` and `d2-6` wrote at `db2cc4f`
from a tree holding one untracked file; [`EVIDENCE.md`](../../EVIDENCE.md) keeps their rows.

**The kept-record tertiles change the frame under its own name**, `nhanes`, because no
`bko.evaluate.named_runs.RUNS` entry reads it, so no scored run folder is re-read under the
frame's identity checks. The `d1-7` and `d2-7` readings above are of the frame before the change.

**The readings at tag `run/d1-12`**, commit `633c0cf`, the frame under the kept-record tertiles.
The 7,769 records' digest is `c2da01f35b27b47ca2421b64705bcf7fab64b0079de6a794fea32815248174b6`
(each run's `data` line), where the frame before the change gave
`602b6ed761f5afe22b0ce58d95c3c987a75d6d0834dbabdba268bace5b3e11ad`; the training split's 6,215
records' digest is `f823ce4fc0902aadfc9aafb622ed92bd48498524491490af63c34403e8c4a7ed` (the
census's `census data` line), where it was
`9b1553fa494ea6d9d10b6f2f115837714523cc10819c738efc6ceaad2c00163c`. The test fingerprint stays
`66905e31f88bee07a0ee5586d12b50b6d3876fe05ac29136e9b20bb0a81bd510` and the `cohort` line is the
one quoted above. Each run ran from a detached worktree at the tag as `env
PYTHONPATH=<worktree>/src .venv/bin/python -m <command> --data-dir data`, exit 0, each log's
`versions` line reading `dirty=false`. The gap listing is run `d1-13`, `bko.gaps nhanes` into
`runs/d1-13/gaps-nhanes.csv` and `runs/d1-13/gaps-nhanes.log`, closing `gaps rows=63
sha256=9e86c72e13a9a51081505de299539f102cdd7da48e48d65a89aa9a8e934b1979`, byte for byte `d1-7`'s
csv; its log sha256 is `31ad134a2e4e19e032dbb840b95473350e8163e08119c92af861bd82fadcd6a8` and
holds the `tertile` line of Decision 5. The census is run `d2-13`, `bko.evaluate contradictions
--dataset nhanes` into `runs/d2-13/contradictions.csv` and `runs/d2-13/contradictions.log`: every
`census kind=` count is 0, over 12 of 12 cells and the 5 rows of `d2-7` (digest
`5f59a89da7abd9f31ea8c62d1b2563616ca66d9fc85f8368a3aca2cb5224284d`), no `is_a` pair. Its csv
sha256 is `b8fbd7f8c95b38fa93262f7e8f92dfd28ed305a3e784342d1b32db9707aa0056` (the header alone)
and its log sha256 `41dfd891b152f390e67867a859cff9b5ebf2a75d12856b08edd8dd704fb6bff1`.

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
| `BMXBMI`'s tertiles over every record of BMX_L | Children's values, from age 2, and adults the frame leaves out would set the cuts of a field coded only on kept adults (Rationale) | 2026-11-26 |
| Weighting records by the survey weights | The frame scores records, not population estimates; the unit leaves weights out | 2026-11-25 |
| The fasting-subsample laboratory files | No field of the frame reads them | 2026-11-25 |

## Enforcement

- Owner: `src/bko/ontology/nhanes_rows.py`
- Pinned by: `tests/test_datasets_nhanes.py::test_any_yes_is_label_1_and_four_noes_are_label_0`,
  `tests/test_datasets_nhanes.py::test_a_record_with_no_yes_and_an_unknown_answer_is_left_out`,
  `tests/test_datasets_nhanes.py::test_only_adults_of_20_and_over_enter`,
  `tests/test_datasets_nhanes.py::test_the_four_questions_never_enter_the_frame`,
  `tests/test_datasets_nhanes.py::test_the_frame_holds_exactly_the_named_fields`,
  `tests/test_datasets_nhanes.py::test_each_cited_field_is_coded_at_its_cutoffs`,
  `tests/test_datasets_nhanes.py::test_a_missing_value_shows_no_concept`,
  `tests/test_datasets_nhanes.py::test_bmi_tertile_base_is_the_kept_adults`,
  `tests/test_evaluate_nhanes.py::test_the_rows_map_these_bands_to_these_concepts`,
  `tests/test_evaluate_nhanes.py::test_every_cutoff_row_cites_a_registered_source_and_names_its_original`,
  `tests/test_datasets_nhanes_pinned.py` (with `--releases`)

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
