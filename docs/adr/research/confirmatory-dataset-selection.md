---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-25
---
# Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page

## Context

The direct-method study ([RESEARCH section 11](../../RESEARCH.md#11-direct-method-study)) reads
its candidate on new datasets after the development data (UCI Heart Disease, the Synthea
population, Diabetes 130). A confirmatory dataset chosen after seeing results confirms nothing, so
the rule that picks the datasets, and the rule that later picks each dataset's removed cell, are
fixed here before any candidate's records are loaded.

Each candidate's licence, access and re-identification terms, and the record counts and field
definitions cited for it, are quoted from the dataset's primary page, read on 2026-11-24; the
2012 challenge's death rate is quoted from its challenge paper,
[`silva-2012b`](../../literature.md#datasets). The other grounds that rank or exclude a candidate
(BRFSS's lack of laboratory measures, HMDA's loan record, the reach of the cutoff mappings) are
the team's reading of those pages. A quotation is the page's text with markup dropped, CRLF line
ends stripped and whitespace runs collapsed to one space, and it runs to a phrase end; a fenced
listing instead keeps its file's line breaks and indentation. A PDF's text is
`pdftotext -layout` output. This record loads no candidate's records, so where no page states a
figure, it records what the page does state.

Each dataset's `Source` pin (`bko.datasets.sources`) carries an `agreement`: `None` for open data,
which `permits_sending` lets go to a third-party API, or an `Agreement(name, sending)`, which
`permits_sending` lets go only when `sending` holds non-blank text; any such text passes
(`cleanup:datasets/agreement-sending-placeholder` in
[`tasks/CLEANUP.md`](../../../tasks/CLEANUP.md)). Each chosen dataset's frame ADR records its pin;
point 4 fixes the value each pin carries.

## Decision Drivers

* **Must be fixed before any record is loaded**, so the choice cannot follow a result.
* **Must rest on primary pages**: a licence, an access route or a record count read second-hand
  is not evidence.
* **Must let every arm and every reading run**, the `bko` arm (which sends records to the
  Anthropic API) and the privacy readings included, or say which may not.
* **Must fix each removed cell from the training split alone, before any classifier is fit**, so
  no score can steer it.

## Decision

**1. The rule.** A dataset is a confirmatory candidate when all six hold:

1. Open access, or an agreement the team holds. A dataset under an agreement the team does not
   hold is excluded.
2. A binary outcome that the dataset itself defines once per record (a person, a stay or a
   visit). A label the frame would have to reduce from a finer one is excluded.
3. Numeric age and binary sex on a person record, the two person axes of the cell grid.
4. Concepts mapped from a code crosswalk or a cited cutoff. The non-health family also needs at
   least one hierarchical coded field; a health dataset needs none, since its concepts map
   through crosswalks and cutoffs.
5. At least 5,000 records. Where the dataset's documentation states its record count, that count
   is read. Where it states none, the frame unit counts its training split before any gap is
   read, and a dataset below 5,000 is excluded and this record revised before registration. The
   minimum counts records, not positives: prevalence is not a condition, and the removed-cell
   filter (point 2) holds every kept cell, positive cells included, to its own floor.
6. Terms and documentation that let results be published. A dataset its documentation calls
   unsuitable for publication is excluded.

**2. The removed cell.** Each chosen dataset's removed cell is fixed from its training split
alone, before any classifier is fit. Of E6's gates (ADR
[`research/diabetes-subgroup-removal-design`](diabetes-subgroup-removal-design.md)), G1 and G3 to
G5 read fitted scores (`real` and `none` recall, MDE, the seed sets), so none of them runs before
registration. G2's count filter runs, on the training split: f is the smallest 3-decimal fraction
of at least 0.900 that leaves R, the cell's kept training rows, in [5, 19]; every other cell
holds at least 20 training rows; and the flagged set is the cell alone on every seed. E6's further
floor of 33 records per cell counts the test split, so it does not apply. Of the cells that pass,
the removed cell is the first in grid order, the order of `bko.datasets.cells.cell_counts`: age
band ascending, then female before male, then label 0 before 1. E6's own cell choice reads scores
and has no tie rule, so this order is this record's. The cutoff the registered run uses is the
training split's prevalence to 4 decimals; G1's choice of 0.5 reads recall, so it does not apply.
A dataset on which no cell passes is excluded, and this record is revised before registration.
Unit `evaluate-confirmatory-registration` records each chosen cell with the sha256 of the
training split it was read from.

**3. The ranking.** The sixteen candidates, best first. The chosen datasets lead, the health
family in the order Rationale gives, then ACS PUMS. The runners-up meet the rule and follow, each
below the chosen dataset that holds its slot. The excluded candidates come last, ordered by the
first condition each fails, and within one condition health before non-health, then
alphabetically. Each row cites the dataset's registry entry and quotes the licence and access
lines of the page that entry names.

| Rank | Dataset | Licence and access, quoted from the primary page | Verdict |
|---|---|---|---|
| 1 | SUPPORT2 ([`vanderbilt-2026`](../../literature.md#datasets)) | "Permission is granted to anyone wishing to use the data sets provided here. Please reference the original paper which, for most data sets, is given in our notes linked below, and note “Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.”" | Chosen, health |
| 2 | PhysioNet/CinC Challenge 2012 ([`silva-2012a`](../../literature.md#datasets)) | "Anyone can access the files, as long as they conform to the terms of the specified license." Licence: "Open Data Commons Attribution License v1.0" | Chosen, health |
| 3 | NHAMCS 2022 ED ([`nhamcs-nd`](../../literature.md#datasets)) | "Please Read Carefully before Using NCHS Public Use Survey Data"; the four terms in point 5 below | Chosen, health |
| 4 | NHANES August 2021 to August 2023 ([`nhanes-2024`](../../literature.md#datasets); terms [`nchs-nd`](../../literature.md#datasets)) | "Researchers who use NCHS public-use data files can only use the information for statistical analysis or reporting purposes." | Chosen, health |
| 5 | ACS PUMS ([`acs-nd`](../../literature.md#datasets)) | "All of the content, documentation, code and related materials made available to you through the API are subject to these terms. Access to or use of the API or its content constitutes acceptance to this Agreement." | Chosen, exploratory non-health family |
| 6 | BRFSS 2024 ([`brfss-nd`](../../literature.md#datasets)) | No licence line on the page; "There are 457,670 records for 2024. More information on participation is available in the states conducting surveillance, by year table. The data files are provided in ASCII and SAS Transport formats." | Runner-up: meets the rule; ranked below NHANES |
| 7 | UCI Census-Income KDD ([`censuskdd-2000`](../../literature.md#datasets)) | "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." | Runner-up, non-health: meets the rule; ranked below ACS PUMS |
| 8 | MEPS ([`meps-nd`](../../literature.md#datasets)) | "1. No one is to use the data in this data set in any way except for statistical reporting and analysis; and" and "Your signature below will indicate that you have carefully read and understood the above statements, and your agreement to comply." | Excluded, condition 1: a signed agreement the team does not hold |
| 9 | PhysioNet/CinC Challenge 2019 ([`reyna-2019`](../../literature.md#datasets)) | "Anyone can access the files, as long as they conform to the terms of the specified license." Licence: "Creative Commons Attribution 4.0 International Public License" | Excluded, condition 2: an hourly label, no per-stay outcome |
| 10 | CDC Diabetes Health Indicators ([`cdcdiabetes-2017`](../../literature.md#datasets)) | "See linked dataset for licensing information." | Excluded, condition 3: age is bucketed |
| 11 | SPARCS 2022 ([`sparcs-nd`](../../literature.md#datasets)) | Licence field of the dataset's metadata record: "See Terms of Use" | Excluded, condition 3: age is grouped |
| 12 | HMDA ([`hmda-nd`](../../literature.md#datasets)) | No licence line on the page; "The snapshot files contain the national HMDA datasets as of a fixed date for all HMDA reporters, as modified by the Bureau to protect applicant and borrower privacy." | Excluded, non-health, condition 3: a loan record, not a person record |
| 13 | UCI Adult ([`becker-1996`](../../literature.md#datasets)) | "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." | Excluded, non-health, condition 4: no hierarchical coded field |
| 14 | eICU demo ([`johnson-2021`](../../literature.md#datasets)) | "Anyone can access the files, as long as they conform to the terms of the specified license." Licence: "Open Data Commons Open Database License v1.0" | Excluded, condition 5: too few records |
| 15 | MIMIC-IV demo ([`johnson-2023`](../../literature.md#datasets)) | "Anyone can access the files, as long as they conform to the terms of the specified license." Licence: "Open Data Commons Open Database License v1.0" | Excluded, condition 5: too few records |
| 16 | Framingham teaching set ([`framingham-nd`](../../literature.md#datasets)) | "The NHLBI has prepared three datasets suitable for use in an undergraduate or graduate level biostatistics instruction program. These datasets are freely available upon request." | Excluded, condition 6: unsuitable for publication |

Each chosen dataset against conditions 2, 3 and 5, from its documentation. The death rate and
the composite's size are recorded for the frame units, not against a condition:

- **SUPPORT2.** The Vanderbilt codebook ([`vanderbilt-2026`](../../literature.md#datasets))
  reads "9105 observations and 47 variables, maximum # NAs:5641" and lists `hospdead` as "Death
  in Hospital", with `age` stored as a double and `sex` with two levels. No page read states the
  in-hospital death rate; the frame unit counts it from the training split. SUPPORT2's terms at
  ICPSR (study 2957, the DOI the UCI page gives) are unopened: the page returned HTTP 403 to curl
  and to a browser, and stayed on a bot check. UCI's page says only "See linked dataset for
  licensing information." The frame unit pins the Vanderbilt archive, whose permission line is
  quoted above.
- **PhysioNet/CinC Challenge 2012** ([`silva-2012a`](../../literature.md#datasets)). "The data
  used for the challenge consist of records from 12,000 ICU stays." The outcome is "In-hospital
  death (0: survivor, or 1: died in-hospital)", with "Age (years)" and "Gender (0: female, or 1:
  male)" among the descriptors. The challenge paper
  ([`silva-2012b`](../../literature.md#datasets)) states "The overall mortality rate was 14.2%."
  over the 12,000 stays. The page's file list holds the outcomes of sets A, B and C.
- **NHAMCS 2022 ED** ([`nhamcs-nd`](../../literature.md#datasets)). The ED documentation serves
  users of the emergency-department public-use file: "Each record on the ED file represents one
  visit in the sample of 16,025 visits." `ADMITHOS` is "Admit to this hospital"; `AGE` is in
  years, top-coded at "94 = 94 years or older"; `SEX` is coded 1 female, 2 male. The file is
  public-use under the NCHS terms; it is not held under a separate agreement.
- **NHANES August 2021 to August 2023.** No page read states the composite outcome's size. The
  Medical Conditions codebook (MCQ_L, [`nhanes-2024`](../../literature.md#datasets)) gives each
  component's Yes count among adults 20 and over: congestive heart failure (MCQ160b) 343,
  coronary heart disease (MCQ160c) 402, heart attack (MCQ160e) 332, stroke (MCQ160f) 367, each of
  roughly 7,800 answers. The composite of the four so holds between 402 and 1,444 adults; the
  frame unit computes the union. No page read states the adult frame's record count as one
  figure, so the frame unit counts its training split before any gap is read, and condition 5
  excludes it below 5,000. DEMO_L gives age "Age in years of the participant at the time of
  screening. Individuals 80 and over are top-coded at 80 years of age." and a two-valued
  `RIAGENDR`.
- **ACS PUMS** ([`acs-nd`](../../literature.md#datasets)). The 2024 user guide says one year's
  PUMS files hold about one percent of the United States population. No page read states the row
  count of one state-year, so the frame unit counts its training split before any gap is read,
  and condition 5 excludes it below 5,000. The 2024 data dictionary gives `AGEP` as numeric,
  "1..99 .1 to 99 years (Top-coded)", and `SEX` with two codes. The outcome is income above the
  Folktables threshold, and the hierarchical coded field is occupation, `OCCP`: "Occupation recode
  for 2018 and later based on 2018 OCC codes".

**4. The `agreement` each chosen dataset's `Source` pin carries.** Each frame unit's pin carries
`agreement=None`. No page read imposes a use agreement that bars sending records to a third-party
API:

| Dataset | `agreement` | Why |
|---|---|---|
| SUPPORT2 | `None` | The Vanderbilt permission line grants use to anyone and asks only for a citation |
| PhysioNet/CinC Challenge 2012 | `None` | Open Data Commons Attribution, open to anyone who conforms to the licence |
| NHAMCS 2022 ED | `None` | Public-use file; the NCHS terms bind by use, limit it to statistical reporting and analysis, and bar identification and linkage, not sending; see the NCHS reading below |
| NHANES | `None` | The same NCHS terms as NHAMCS |
| ACS PUMS | `None` | The API terms bar identification, linkage and identifying publication, not sending. The index page of the file route (the 2024 1-Year PUMS directory) states no terms, so the API terms are read as the Census Bureau's terms for its data |

**The NCHS reading.** NHANES and NHAMCS 2022 ED keep `agreement=None`, and sending is allowed. The
NCHS terms bind by use alone, with no signed agreement: "By using these data you signify your
agreement to comply with the above-stated statutorily based requirements." MEPS differs: it asks
for a signature before access, "Your signature below will indicate that you have carefully read
and understood the above statements, and your agreement to comply." `Agreement`'s docstring
describes data held under a data use agreement; the NCHS public-use terms are read as conditions
of use, not a held agreement in that sense, so the pins carry `None`. The terms allow use "for
statistical analysis or reporting purposes", and generating records to score classifiers is
statistical analysis. This is the team's reading, confirmed by Joe Laws on 2026-11-24; it is not
legal advice.

**5. Re-identification clauses and the team's reading.**

- **NHANES and NHAMCS 2022 ED.** Both fall under the NCHS terms. `readme2022.txt` lists them as
  four points, of which point 4 reads (the file's line breaks and indentation kept, its trailing
  spaces dropped):

  ```
         4.  Not engage in any efforts to assess disclosure methodologies applied to protect
             individuals and establishments or any research on methods of re-identification of
             individuals and establishments.
  ```

  The NCHS Data User Agreement page ([`nchs-nd`](../../literature.md#datasets)) carries the
  same points, among them "Make no attempt to learn the identity of any person or establishment
  included in these data." The reading: the privacy readings (membership inference, NNDR,
  distance to closest record, the match attack, donor reuse) score the generated rows a method
  writes, measure that method's disclosure, and attempt to identify no person. They assess no
  NCHS disclosure method and do no re-identification research on the NCHS files, so they lie
  outside the clause. Every reading runs on both datasets.
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
  "made available to you through the API", and the file route's index page states none of its
  own; the clause is read as the Census Bureau's stated condition on its public data, and the
  frame follows it whichever route its file comes by.
- **SUPPORT2.** The Vanderbilt and UCI pages hold no re-identification clause; UCI's page notes
  "Yes. There is information about race, gender, income, and education level." ICPSR's terms are
  unopened. Every reading runs.
- **PhysioNet/CinC Challenge 2012.** The page and its licence line hold no re-identification
  clause. Every reading runs.

**6. Families.** SUPPORT2, NHANES, the PhysioNet/CinC Challenge 2012 set and NHAMCS 2022 ED form
the health multiplicity family. ACS PUMS is its own exploratory family, outside it.

## Rationale

The rule's conditions are the ones the study's machinery needs: the cell grid reads age and sex,
the classifiers read a binary label, gap detection reads mapped concepts, and the removed-cell
filter needs every kept cell to hold at least 20 training rows. 5,000 records leaves that floor
headroom on a grid of up to 18 cells at a common label; the 303-record UCI frame cannot separate
the arms. A rare label can leave positive cells below the floor; the filter checks every cell,
positive cells included, so the record minimum need not count positives.

The ranking puts the two intensive-care sets first because their outcome is defined per stay and
their physiology maps through cited cutoffs. NHAMCS adds coded diagnoses on a public-use file.
NHANES is the one candidate whose laboratory values map through cited cutoffs in a general
population, which is why BRFSS, a telephone survey with no laboratory measures, ranks below it.
The 2019 challenge set fails condition 2: its `SepsisLabel` is set per hour, to 1 from six hours
before a sepsis patient's onset. A per-stay outcome would be a reduction the challenge does not
define; the 2012 set's outcome is per stay.

ACS PUMS leads the non-health candidates because its occupation code is a hierarchical field,
which unit `datasets-acs-frame` maps to the Occupation Ontology. UCI Census-Income KDD also has
"occupation code" under "major occupation code" and ranks second; unit
`datasets-non-health-selection` ranks the non-health candidates again under its own rule.

The removed cell keeps G2's count filter because it reads counts alone and is already written.
G1 and G3 to G5 read fitted scores, and a cell chosen from a score confirms nothing. Grid order
settles the choice among passing cells because it is fixed before any record is loaded.

## Consequences

**Gained**: the five frame units start from quoted terms, a fixed `agreement` value and a stated
reading of each re-identification clause. The `bko` arm and every privacy reading run on all five.
**Accepted**: three figures the frame units need are on no page read: SUPPORT2's in-hospital
death rate, the NHANES composite's size (bounded above), and one ACS state-year's row count. Each
frame unit records its figure from the training split before reading any gap. An ACS state-year
or an NHANES adult frame below 5,000 records is excluded by condition 5, and a dataset on which
no cell passes the removed-cell filter is excluded by point 2; either way this record is revised
before registration. The removed cell gives up E6's score-based choice: the first passing cell in
grid order may resolve its contrasts less well than the cell E6's rule would pick. ICPSR's terms
for SUPPORT2 stay unread. The NCHS readings rest on the team's reading of the NCHS terms, not on
a written permission from NCHS.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Choosing the removed cell after scoring | A cell chosen after a score is read confirms nothing; the cell is G2's count filter on the training split, first in grid order | 2026-11-24 |
| E6's full rule, G1 to G5 and its cell choice | G1 and G3 to G5 read fitted scores; see Rationale | 2026-11-24 |
| PhysioNet/CinC Challenge 2019 | Fails condition 2: hourly label shifted six hours before onset; no per-stay outcome is defined | 2026-11-24 |
| BRFSS | Meets the rule; a telephone survey with no laboratory measures, so NHANES holds the general-population slot | 2026-11-24 |
| UCI Census-Income KDD | Meets the rule; ranked below ACS PUMS for the one non-health family; re-ranked by `datasets-non-health-selection` | 2026-11-24 |
| MEPS | Fails condition 1: use needs a signed agreement the team does not hold, limited to the purpose the data were supplied for | 2026-11-24 |
| HMDA | Fails condition 3: a loan record, not a person record; re-ranked by `datasets-non-health-selection` | 2026-11-24 |
| UCI Adult | Fails condition 4: its coded fields are flat: occupation is "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces." | 2026-11-24 |
| CDC Diabetes Health Indicators | Fails condition 3, numeric age: its preprocessing reads "Bucketing of age" | 2026-11-24 |
| SPARCS 2022 | Fails condition 3, numeric age: "Patient age is presented in age groups: 0 to 17, 18 to 29, 30 to 49, 50 to 69, and 70 or Older." | 2026-11-24 |
| eICU demo | Fails condition 5, the record minimum: "over 2,500 unit stays" | 2026-11-24 |
| Framingham teaching set | Fails condition 6: "Users are cautioned that teaching datasets are completely unsuitable for publication purposes since specific statistical measures were used to create anonymous versions." | 2026-11-24 |
| MIMIC-IV demo | Fails condition 5, the record minimum: "a subset of 100 patients" | 2026-11-24 |

## Enforcement

- Owner: `docs/RESEARCH.md`
- Pinned by: nothing -- convention only

## Reversal Conditions

**A re-read of a quoted page differs from its quote**, in a licence, an access route, a record
count or a re-identification clause. **NCHS or the Census Bureau states that scoring the privacy
of generated rows falls under its re-identification clause.** **NCHS states how its public-use
terms treat sending records to a third party for processing.** **Terms found on the ACS PUMS file
route differ from the API terms.** **ICPSR's terms, once opened, bar a use the study makes.**

## Related

- [The Diabetes 130 subgroup-removal design: the full adult frame, one thinned cell at the prevalence cutoff, every arm](diabetes-subgroup-removal-design.md)
- [Membership inference, verbatim matches and NNDR read an arm's added rows against a null arm, outside the removed cell](membership-inference-reading.md)
