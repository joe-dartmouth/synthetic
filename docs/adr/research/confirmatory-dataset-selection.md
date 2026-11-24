---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page

## Context

The direct-method study ([RESEARCH section 11](../../RESEARCH.md#11-direct-method-study)) reads
its candidate on new datasets after the development data (UCI Heart Disease, the Synthea
population, Diabetes 130). A confirmatory dataset chosen after seeing results confirms nothing, so
the rule that picks the datasets, and the rule that later picks each dataset's removed cell, are
fixed here before any candidate's records are loaded.

Every statement below that a dataset's choice rests on is quoted from the dataset's primary
page, read on 2026-11-24. A quotation is the page's text with markup dropped and whitespace runs
collapsed to one space; a PDF's text is `pdftotext -layout` output. This record loads no
candidate's records, so where no page states a figure, it records what the page does state.

Each dataset's `Source` pin (`bko.datasets.sources`) carries an `agreement`: `None` for open data,
which `permits_sending` lets go to a third-party API, or an `Agreement(name, sending)`, which
permits sending only when `sending` names a permission.

## Decision Drivers

* **Must be fixed before any record is loaded**, so the choice cannot follow a result.
* **Must rest on primary pages**: a licence, an access route or a record count read second-hand
  is not evidence.
* **Must let every arm and every reading run**, the `bko` arm (which sends records to the
  Anthropic API) and the privacy readings included, or say which may not.
* **Must keep a removed cell a function of the training split alone**, as E6's is.

## Decision

**1. The rule.** A dataset is a confirmatory candidate when all five hold:

1. Open access, or an agreement the team holds. A dataset under an agreement the team does not
   hold is excluded.
2. A binary outcome.
3. Numeric age and binary sex, the two person axes of the cell grid.
4. Concepts mapped from a code crosswalk or a cited cutoff, with at least one hierarchical coded
   field for the non-health family.
5. At least 5,000 records, the count taken from the dataset's own documentation.

**2. The removed cell.** Each chosen dataset's removed cell is picked by E6's mechanical rule,
gates G1 to G5 and the cell choice of ADR
[`research/diabetes-subgroup-removal-design`](diabetes-subgroup-removal-design.md), run on the
training split alone. A tie goes to the cell earlier in grid order. Unit
`evaluate-confirmatory-registration` runs the rule once per dataset and logs the run.

**3. The ranking.** The sixteen candidates, best first. Each row quotes the licence and access
lines of the page named.

| Rank | Dataset | Licence and access, quoted from the primary page | Verdict |
|---|---|---|---|
| 1 | SUPPORT2 ([hbiostat.org/data](https://hbiostat.org/data/)) | "Permission is granted to anyone wishing to use the data sets provided here. Please reference the original paper which, for most data sets, is given in our notes linked below, and note “Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.”" | Chosen, health |
| 2 | PhysioNet/CinC Challenge 2012 ([physionet.org](https://physionet.org/content/challenge-2012/1.0.0/)) | "Anyone can access the files, as long as they conform to the terms of the specified license." Licence: "Open Data Commons Attribution License v1.0" | Chosen, health |
| 3 | NHAMCS 2022 ED ([readme2022.txt](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/readme2022.txt)) | "Please Read Carefully before Using NCHS Public Use Survey Data"; the four terms in point 5 below | Chosen, health |
| 4 | NHANES August 2021 to August 2023 ([NCHS Data User Agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html)) | "Researchers who use NCHS public-use data files can only use the information for statistical analysis or reporting purposes." | Chosen, health |
| 5 | ACS PUMS ([Census terms of service](https://www.census.gov/data/developers/about/terms-of-service.html)) | "All of the content, documentation, code and related materials made available to you through the API are subject to these terms. Access to or use of the API or its content constitutes acceptance to this Agreement." | Chosen, exploratory non-health family |
| 6 | PhysioNet/CinC Challenge 2019 ([physionet.org](https://physionet.org/content/challenge-2019/1.0.0/)) | "Anyone can access the files, as long as they conform to the terms of the specified license." Licence: "Creative Commons Attribution 4.0 International Public License" | Runner-up: meets the rule; ranked below the 2012 set |
| 7 | BRFSS 2024 ([cdc.gov](https://www.cdc.gov/brfss/annual_data/annual_2024.html)) | No licence line on the page; "There are 457,670 records for 2024. More information on participation is available in the states conducting surveillance, by year table. The data files are provided in ASCII and SAS Transport formats." | Runner-up: ranked below NHANES |
| 8 | UCI Census-Income KDD ([UCI 117](https://archive.ics.uci.edu/dataset/117/census+income+kdd)) | "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." | Runner-up, non-health: ranked below ACS PUMS |
| 9 | MEPS ([data use agreement](https://meps.ahrq.gov/data_stats/data_use.jsp)) | "1. No one is to use the data in this data set in any way except for statistical reporting and analysis; and" and "Your signature below will indicate that you have carefully read and understood the above statements, and your agreement to comply." | Excluded: a signed agreement the team does not hold |
| 10 | HMDA ([ffiec.cfpb.gov](https://ffiec.cfpb.gov/data-publication/)) | No licence line on the page; "The snapshot files contain the national HMDA datasets as of a fixed date for all HMDA reporters, as modified by the Bureau to protect applicant and borrower privacy." | Excluded, non-health: a loan record, ranked below ACS PUMS |
| 11 | UCI Adult ([UCI 2](https://archive.ics.uci.edu/dataset/2/adult)) | "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." | Excluded: no hierarchical coded field |
| 12 | CDC Diabetes Health Indicators ([UCI 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)) | "See linked dataset for licensing information." | Excluded: age is bucketed |
| 13 | SPARCS 2022 ([health.data.ny.gov](https://health.data.ny.gov/d/5dtw-tffi)) | Licence field: "See Terms of Use" | Excluded: age is grouped |
| 14 | eICU demo ([physionet.org](https://physionet.org/content/eicu-crd-demo/2.0.1/)) | "Anyone can access the files, as long as they conform to the terms of the specified license." Licence: "Open Data Commons Open Database License v1.0" | Excluded: too few records |
| 15 | Framingham teaching set ([BioLINCC](https://biolincc.nhlbi.nih.gov/teaching/)) | "The NHLBI has prepared three datasets suitable for use in an undergraduate or graduate level biostatistics instruction program. These datasets are freely available upon request." | Excluded: unsuitable for publication |
| 16 | MIMIC-IV demo ([physionet.org](https://physionet.org/content/mimic-iv-demo/2.2/)) | "Anyone can access the files, as long as they conform to the terms of the specified license." Licence: "Open Data Commons Open Database License v1.0" | Excluded: too few records |

Each chosen dataset against conditions 2, 3 and 5, from its documentation:

- **SUPPORT2.** The Vanderbilt codebook reads "9105 observations and 47 variables, maximum # NAs:5641"
  and lists `hospdead` as "Death in Hospital", with `age` stored as a double and `sex` with two
  levels. No page read states the in-hospital death rate; the frame unit counts it from the file.
  SUPPORT2's terms at ICPSR (study 2957, the DOI the UCI page gives) are unopened: the page
  returned HTTP 403 to curl and to a browser, and stayed on a bot check. UCI's page says only "See
  linked dataset for licensing information." The frame unit pins the Vanderbilt archive, whose
  permission line is quoted above.
- **PhysioNet/CinC Challenge 2012.** "The data used for the challenge consist of records from
  12,000 ICU stays." The outcome is "In-hospital death (0: survivor, or 1: died in-hospital)", with
  "Age (years)" and "Gender (0: female, or 1: male)" among the descriptors. The challenge paper
  ([`silva-2012`](../../literature.md#datasets)) states "The overall mortality rate was 14.2%." over
  the 12,000 stays. The page's file list holds the outcomes of sets A, B and C.
- **NHAMCS 2022 ED.** The documentation is "for users of the Emergency Department public use"
  file: "Each record on the ED file represents one visit in the sample of 16,025 visits."
  `ADMITHOS` is "Admit to this hospital"; `AGE` is in years, top-coded at "94 = 94 years or older";
  `SEX` is coded 1 female, 2 male. The file is public-use under the NCHS terms; it is not held
  under a separate agreement.
- **NHANES August 2021 to August 2023.** The page states no size for the composite outcome. The
  Medical Conditions codebook (MCQ_L) gives each component's Yes count among adults 20 and over:
  congestive heart failure (MCQ160b) 343, coronary heart disease (MCQ160c) 402, heart attack
  (MCQ160e) 332, stroke (MCQ160f) 367, each of roughly 7,800 answers. The composite of the four so
  holds between 402 and 1,444 adults; the frame unit computes the union. DEMO_L gives age "Age in
  years of the participant at the time of screening. Individuals 80 and over are top-coded at 80
  years of age." and a two-valued `RIAGENDR`.
- **ACS PUMS.** The 2024 user guide states "PUMS files for an individual year contain data on
  approximately one percent of the United" States population; no page read states the row count
  of one state-year, which the frame unit records from its file before any gap is read. The 2024
  data dictionary gives `AGEP` as numeric, "1..99  .1 to 99 years (Top-coded)", and `SEX` with two
  codes. The outcome is income above the Folktables threshold, and the hierarchical coded field is
  occupation.

**4. The `Agreement` each chosen dataset's `Source` carries.** All five carry `agreement=None`. No
page read imposes a use agreement that bars sending records to a third-party API:

| Dataset | `agreement` | Why |
|---|---|---|
| SUPPORT2 | `None` | The Vanderbilt permission line grants use to anyone and asks only for a citation |
| PhysioNet/CinC Challenge 2012 | `None` | Open Data Commons Attribution, open to anyone who conforms to the licence |
| NHAMCS 2022 ED | `None` | Public-use file; the NCHS terms limit use to statistical reporting and analysis and bar identification and linkage, not sending; generating records to score classifiers is statistical analysis |
| NHANES | `None` | The same NCHS terms as NHAMCS |
| ACS PUMS | `None` | The terms bar identification, linkage and identifying publication, not sending |

**5. Re-identification clauses and the team's reading.**

- **NHANES and NHAMCS 2022 ED.** Both fall under the NCHS terms. `readme2022.txt` lists them as
  four points, of which point 4 reads (the file's line breaks and indentation kept, its trailing
  spaces dropped):

  ```
         4.  Not engage in any efforts to assess disclosure methodologies applied to protect
             individuals and establishments or any research on methods of re-identification of
             individuals and establishments.
  ```

  The NCHS Data User Agreement page carries the same points, among them "Make no attempt to learn
  the identity of any person or establishment included in these data." The reading: the privacy
  readings (membership inference, NNDR, distance to closest record, the match attack, donor reuse)
  score the generated rows a method writes, measure that method's disclosure, and attempt to
  identify no person. They assess no NCHS disclosure method and do no re-identification research
  on the NCHS files, so they lie outside the clause. Every reading runs on both datasets.
- **ACS PUMS.** "Therefore, users will not use these data, alone or in combination with any other
  Census or non-Census data, to identify any individual person, household, business or other
  entity; not link or combine these data with information in any other Census or non-Census
  dataset in a manner that identifies an individual person, household, business or other entity;
  not publish information from these data files, particularly in combination with any other Census
  or non-Census data, in a manner that identifies any individual person, household, business or
  other entity; and not use the identity of any person or establishment discovered inadvertently
  and advise the Director of the Census Bureau of any such discovery by calling the U.S. Census
  Bureau’s Policy Coordination Office at 301-763-6440." The reading is NHANES's: the readings
  identify no one and publish no identity, and every reading runs. The terms' scope is content
  "made available to you through the API"; the clause is the Census Bureau's stated condition on
  its public data, and the frame follows it whichever route its file comes by.
- **SUPPORT2.** The Vanderbilt and UCI pages hold no re-identification clause; UCI's page notes
  "Yes. There is information about race, gender, income, and education level." ICPSR's terms are
  unopened. Every reading runs.
- **PhysioNet/CinC Challenge 2012.** The page and its licence line hold no re-identification
  clause. Every reading runs.

**6. Families.** SUPPORT2, NHANES, the PhysioNet/CinC Challenge 2012 set and NHAMCS 2022 ED form
the health multiplicity family. ACS PUMS is its own exploratory family, outside it.

## Rationale

The rule's conditions are the ones the study's machinery needs: the cell grid reads age and sex,
the classifiers read a binary label, gap detection reads mapped concepts, and E6's gates need
every kept cell to hold at least 20 training rows and 33 records. 5,000 records leaves that floor
headroom on a grid of up to 18 cells; the 303-record UCI frame cannot separate the arms.

The ranking puts the two intensive-care sets first because their outcome is defined per stay and
their physiology maps through cited cutoffs. NHAMCS adds coded diagnoses on a public-use file.
NHANES is the one candidate whose laboratory values map through cited cutoffs in a general
population, which is why BRFSS, a telephone survey with no laboratory measures, ranks below it.
The 2019 challenge set meets the rule, but its `SepsisLabel` is set per hour, to 1 from six
hours before a sepsis patient's onset. A per-stay outcome would be a reduction the challenge does
not define; the 2012 set's outcome is per stay.

ACS PUMS leads the non-health candidates because its occupation code is a hierarchical field,
which unit `datasets-acs-frame` maps to the Occupation Ontology. UCI Census-Income KDD also has
"occupation code" under "major occupation code" and ranks second; unit
`datasets-non-health-selection` ranks the non-health candidates again under its own rule.

E6's rule picks the removed cell because it is already a function of the training split and its
gates are written; reusing it keeps the confirmatory cell as mechanical as the development one.

## Consequences

**Gained**: the five frame units start from quoted terms, a fixed `agreement` value and a stated
reading of each re-identification clause. The `bko` arm and every privacy reading run on all five.
**Accepted**: three figures the unit asked for are not on any page read: SUPPORT2's in-hospital
death rate, the NHANES composite's size (bounded above), and one ACS state-year's row count. Each
frame unit records its figure from the file before reading any gap. ICPSR's terms for SUPPORT2
stay unread. The NCHS readings rest on the team's reading of the NCHS terms, not on a written
permission from NCHS.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Choosing the removed cell after scoring | A cell chosen after a score is read confirms nothing; the cell is E6's rule on the training split | 2026-11-24 |
| PhysioNet/CinC Challenge 2019 | Hourly label shifted six hours before onset; no per-stay outcome is defined | 2026-11-24 |
| BRFSS | Telephone survey with no laboratory measures; NHANES holds the general-population slot | 2026-11-24 |
| UCI Census-Income KDD | Ranked below ACS PUMS for the one non-health family; re-ranked by `datasets-non-health-selection` | 2026-11-24 |
| MEPS | Use needs a signed agreement the team does not hold, limited to the purpose the data were supplied for | 2026-11-24 |
| HMDA | Loan-level records with no hierarchical coded field read; re-ranked by `datasets-non-health-selection` | 2026-11-24 |
| UCI Adult | Its coded fields are flat: occupation is "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces." | 2026-11-24 |
| CDC Diabetes Health Indicators | Fails numeric age: its preprocessing reads "Bucketing of age" | 2026-11-24 |
| SPARCS 2022 | Fails numeric age: "Patient age is presented in age groups: 0 to 17, 18 to 29, 30 to 49, 50 to 69, and 70 or Older." | 2026-11-24 |
| eICU demo | Fails the record minimum: "over 2,500 unit stays" | 2026-11-24 |
| Framingham teaching set | "Users are cautioned that teaching datasets are completely unsuitable for publication purposes since specific statistical measures were used to create anonymous versions." | 2026-11-24 |
| MIMIC-IV demo | Fails the record minimum: "a subset of 100 patients" | 2026-11-24 |

## Enforcement

- Owner: `docs/RESEARCH.md`
- Pinned by: nothing -- convention only

## Reversal Conditions

**A re-read of a quoted page differs from its quote**, in a licence, an access route, a record
count or a re-identification clause. **NCHS or the Census Bureau states that scoring the privacy
of generated rows falls under its re-identification clause.** **ICPSR's terms, once opened, bar a
use the study makes.**

## Related

- [The Diabetes 130 subgroup-removal design: the full adult frame, one thinned cell at the prevalence cutoff, every arm](diabetes-subgroup-removal-design.md)
- [Membership inference, verbatim matches and NNDR read an arm's added rows against a null arm, outside the removed cell](membership-inference-reading.md)
