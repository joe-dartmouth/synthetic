---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
---
# PhysioNet/CinC Challenge 2012's frame: in-hospital death as the label, each stay's 48 hours reduced to its minimum and maximum

## Context

ADR [`research/confirmatory-dataset-selection`](confirmatory-dataset-selection.md) chose the
PhysioNet/CinC Challenge 2012 set ([`silva-2012a`](../../literature.md#datasets)) as a health
dataset, all 12,000 stays, with in-hospital death as the outcome. The challenge page, read
2026-11-25, describes the data: "The data used for the challenge consist of records from 12,000
ICU stays. All patients were adults who were admitted for a wide variety of reasons to cardiac,
medical, surgical, and trauma ICUs. ICU stays of less than 48 hours have been excluded." and "Up
to 42 variables were recorded at least once during the first 48 hours after admission to the
ICU." It names six general descriptors, among them "Gender (0: female, or 1: male)" and "ICUType
(1: Coronary Care Unit, 2: Cardiac Surgery Recovery Unit, 3: Medical ICU, or 4: Surgical ICU)",
and 37 time series, among them "MAP [Invasive mean arterial blood pressure (mmHg)]", "NIMAP
[Non-invasive mean arterial blood pressure (mmHg)]", "MechVent [Mechanical ventilation
respiration (0:false, or 1:true)]", "WBC [White blood cell count (cells/nL)]", "Platelets
(cells/nL)" and "GCS [Glasgow Coma Score (3-15)]". It says: "All valid values for general
descriptors, time series variables, and outcome-related descriptors are non-negative (≥ 0). A
value of -1 indicates missing or unknown data (for example, if a patient's height was not
recorded)." and "In a few cases, such as blood pressure, different measurements made using two or
more methods or sensors may be recorded with the same or only slightly different time-stamps.
Occasional outliers should be expected as well." The outcomes file of each set holds "SAPS-I
score", "SOFA score", "Length of stay (days)", "Survival (days)" and "In-hospital death (0:
survivor, or 1: died in-hospital)". The page asks for two citations: the standard citation for
PhysioNet ([`pollard-2026`](../../literature.md#datasets)), and the challenge paper "when
referencing the Challenge" ([`silva-2012b`](../../literature.md#datasets)), which states "The
overall mortality rate was 14.2%." over the 12,000 stays (read from PubMed Central on
2026-11-25). The licence is "Open Data Commons Attribution License v1.0". A quotation is the
page's text with markup dropped and whitespace runs collapsed to one space.

The files, under `https://physionet.org/files/challenge-2012/1.0.0/`, which redirects to an HTTPS
S3 address: sets A and B as zips, set C only as a gzipped tar, and one outcomes file per set. Each
archive holds a directory entry `set-x/` and 4,000 members `set-x/<RecordID>.txt`. A member is
CSV with LF line ends: the header `Time,Parameter,Value`, then one observation per line, its time
`hh:mm` since admission, at most `48:00`. The descriptors sit at `00:00`, `RecordID` on the first
line; in some stays a time series observed at `00:00` sits among them. Read 2026-11-25 from the
downloaded bytes: `Age` is never -1 and runs from 15 to 90, 464 stays at 90 against 120 at 89, and
the page states no top code; `Gender` is -1 in 12 stays; `ICUType` is always 1 to 4. `MechVent`
is only ever 1, in 91,644 lines. `Temp` holds 59 values below 0 that are not -1, 55 of them
-17.8. 83 values are written in exponent form, such as `1.213e+04`. Set C holds 2,403 lines in
1,594 stays whose `Parameter` is empty; sets A and B hold none. Each outcomes file has the header
`RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death` and one line per stay of its set.

A mapped lab or vital needs a cited cutoff. As ADR [`research/support2-frame`](support2-frame.md)
records, the SIRS criteria ([`bone-1992`](../../literature.md#clinical-definitions)) and SOFA
([`vincent-1996`](../../literature.md#clinical-definitions)) are read in the Sepsis-3 consensus
paper's reproduction ([`singer-2016`](../../literature.md#clinical-definitions), Box 1 and Table
1, read from PubMed Central). Table 1 gives SOFA's coagulation score 1 as "<150" platelets in
"×10 3 /µL", and its central nervous system scores as "15" for score 0 and "13–14" for score 1,
"Glasgow Coma Scale scores range from 3–15".

## Decision Drivers

* **Must fix the reduction rule before any gap listing is read**, so no figure picks it.
* **Must keep the label free of leakage**: no outcome-related descriptor but the label enters.
* **Must map physiology only through a cited cutoff**, read in its source's text before the row
  is written.
* **Must let a missing value show no concept**: a variable a stay never measured shows neither a
  finding nor its absence.
* **Must refuse a malformed file rather than drop a line silently**: every line the loader skips
  is counted.

## Decision

The pins are six `bko.datasets.sources` entries, `agreement=None`, as ADR
`research/confirmatory-dataset-selection` point 4 fixes: `set-a.zip`, `set-b.zip`,
`set-c.tar.gz`, `Outcomes-a.txt`, `Outcomes-b.txt` and `Outcomes-c.txt`, each size the host's
content-length and each sha256 measured from the bytes served:

| File | Bytes | sha256 |
|---|---|---|
| `set-a.zip` | 7,938,449 | `1433638192fda622e8bb11a4559b8af2dbb1c69f0e1c3b5b606cc12aabd0a4c5` |
| `set-b.zip` | 7,958,979 | `54094c39631797af554ad4233164e004a699521351bd2f2bed50d6564ebc56dc` |
| `set-c.tar.gz` | 6,600,293 | `a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1` |
| `Outcomes-a.txt` | 79,219 | `2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2` |
| `Outcomes-b.txt` | 79,149 | `1305f29734da374a809483229a99f33a9230aa8d4fa74b1a2d6ad788136423d5` |
| `Outcomes-c.txt` | 79,191 | `0b32dcb7753c0f18f2f5fde498a28618b8879c5e527ce0dae8ad120dd7e205c9` |

**1. Reading the archives.** Each archive is read in memory, never extracted. A member must be a
regular file named `set-x/<digits>.txt` of its own set; a directory entry is skipped. A link, a
device, an absolute or parent path, a repeated name, a member over 64 MiB, more than 256 MiB in
all, and a damaged archive are refused.

**2. The records.** One record per stay, `record_id` its `RecordID`, which must equal its member's
name; a `RecordID` repeated across the three sets is refused. Each set's outcomes file must hold
exactly one line per stay of that set: a stay with no outcome line and an outcome line with no
stay are refused. A stay must end in a newline, start with the header, and hold three fields per
line, a time from `00:00` to `48:00`, and a finite number. `RecordID`, `Age`, `Gender`, `Height`
and `ICUType` each appear once, at `00:00`; `Age` must be whole, `Gender` 0, 1 or -1, `ICUType`
1 to 4. A `Parameter` that is none of the six descriptors and 37 time series is refused, except
an empty one, which names no variable: its line is skipped and counted.

**3. The label and the cohort.** `target` is `In-hospital_death`, 0 or 1. A stay whose `Age` or
`Gender` is -1 is left out and counted. `age` is `Age`; `sex` is `Gender`, 0 female and 1 male.
Age bands start at 50, 65 and 80.

**4. The reduction rule.** Per stay and variable, over the whole 48 hours and over the values at
or above 0: the minimum and the maximum. A value below 0, -1 or another, is not a reading; such
lines of the variables the frame reads are counted. A stay with no value at or above 0 of a
variable did not measure it.

**5. The fields.** `age`, `sex`, `ICUType` (its four codes, as the page gives them), then these
coded states (`bko.datasets.states.coded`, `0` = not measured), each cutoff as Singer et al.
reproduce it, then `mechvent`:

| Field | Variable and reduction | Cutoff | Source as reproduced |
|---|---|---|---|
| `hr_max` | `HR`, maximum | above 90 | Box 1 (SIRS): "Heart rate >90/min" |
| `resprate_max` | `RespRate`, maximum | above 20 | Box 1 (SIRS): "Respiratory rate >20/min or PaCO 2 <32 mm Hg (4.3 kPa)" |
| `temp_max` | `Temp`, maximum | above 38 | Box 1 (SIRS): "Temperature >38°C or <36°C" |
| `temp_min` | `Temp`, minimum | below 36 | the same |
| `wbc_max` | `WBC`, maximum | above 12 | Box 1 (SIRS): "White blood cell count >12 000/mm 3 or <4000/mm 3 or >10% immature bands" |
| `wbc_min` | `WBC`, minimum | below 4 | the same |
| `map_min` | `MAP`, minimum; `NIMAP`'s where the stay has no `MAP` | below 70 | Table 1 (SOFA cardiovascular, score 1): "MAP <70 mm Hg" |
| `creatinine_max` | `Creatinine`, maximum | 1.2 and above | Table 1 (SOFA renal, score 1): "1.2–1.9 (110–170)" mg/dL |
| `bilirubin_max` | `Bilirubin`, maximum | 1.2 and above | Table 1 (SOFA liver, score 1): "1.2–1.9 (20–32)" mg/dL |
| `platelets_min` | `Platelets`, minimum | below 150 | Table 1 (SOFA coagulation, score 1): "<150" |
| `gcs_min` | `GCS`, minimum | below 15 | Table 1 (SOFA central nervous system): score 0 "15", score 1 "13–14" |

`WBC` and `Platelets` are in cells per nanolitre, which is thousands per cubic millimetre and
the table's ×10 3 /µL, so 12,000 and 4,000 per cubic millimetre read 12 and 4. `mechvent` is 1
when the stay holds a `MechVent` value of 1, else 0: no ventilation recorded.

**6. The rows.** One `Codes` row per abnormal state, to HPO, labels the pinned release's names:
`hr_max` above 90 Tachycardia; `resprate_max` above 20 Tachypnea; `temp_max` above 38 Fever;
`temp_min` below 36 Hypothermia; `wbc_max` above 12 Increased total leukocyte count; `wbc_min`
below 4 Decreased total leukocyte count; `map_min` below 70 Hypotension; `creatinine_max` 1.2 and
above Elevated circulating creatinine concentration; `bilirubin_max` 1.2 and above
Hyperbilirubinemia; `platelets_min` below 150 Thrombocytopenia; `gcs_min` below 15 Reduced
consciousness. Each row's source is `singer-2016`, its limits naming the original and saying the
cutoff was read in Singer et al.'s reproduction. `ICUType` and `mechvent` map nothing.

**7. What never enters.** `SAPS-I`, `SOFA`, `Length_of_stay` and `Survival`, which are outcome-
related, and every descriptor and time series the fields above do not name, `Height` and `Weight`
among them.

**8. The order.** The reduction rule and the rows are committed; then the training split
(`split(records, seed=0, edges=frame.edges).train`) is counted, its record count, its in-hospital
deaths and their rate; then the gap listing and the census are read. Consequences record each
step's commit. A training split below 5,000 records excludes the frame until ADR
`research/confirmatory-dataset-selection` is revised.

**9. Every reading.** The page and its licence hold no re-identification clause, as ADR
`research/confirmatory-dataset-selection` point 5 records, so every reading runs on this frame.
This frame fits and scores nothing; a later unit runs them.

## Rationale

Each cutoff is a criterion a stay meets when any of its readings crosses it: "Heart rate >90/min"
holds for a stay whose highest heart rate is above 90, and "MAP <70 mm Hg" for one whose lowest
pressure is below 70. The maximum and the minimum over the window are exactly the readings that
decide whether any reading crossed, so the reduction keeps each row's meaning the criterion's and
asks nothing of when or how often a variable was measured. Taking both over the whole 48 hours
uses every stay's full window, which the page guarantees, since stays under 48 hours were
excluded. Reading only values at or above 0 follows the page: -1 is missing, and no valid value is
negative, so `Temp`'s -17.8 is not a reading.

`map_min` prefers the invasive `MAP` because the page lists both readings of one quantity and the
arterial line is the direct one; a stay with no arterial line still shows its cuff pressure.
`mechvent` keeps the page's own variable as a free field: the files record only 1, so its absence
means no ventilation was recorded, not a reading of 0. The empty-`Parameter` lines name no
variable the page lists, so no field could hold them; counting them keeps the skip visible.

## Consequences

**Gained**: a second intensive-care frame beside SUPPORT2's, with the same SIRS and SOFA rows read
from a 48-hour window. **Accepted**: the minimum and the maximum follow the outliers the page warns
of, so a single spurious reading can show a finding; a recorded 0 is a valid value by the
page's own rule, so a `Temp` of 0 shows Hypothermia and a `MAP` of 0 Hypotension. A `GCS` below 15
under sedation reads as Reduced consciousness, as SOFA reads it.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The first, last, mean or median value per stay | Only the maximum and the minimum decide whether a reading crossed the criterion's cutoff | 2026-11-25 |
| The first 24 hours only | The page guarantees 48 hours per stay; a shorter window would drop readings without reason | 2026-11-25 |
| Imputing a variable a stay never measured | An imputed value would show a finding, or its absence, no one measured | 2026-11-25 |
| `SAPS-I` or `SOFA` as fields | Both are scores of the outcomes file, computed over the stay, so they would leak toward the label | 2026-11-25 |
| Set A alone, the only set with outcomes during the challenge | The page now serves every set's outcomes, and the selection ADR chose the 12,000 stays | 2026-11-25 |
| A row on `mechvent` | It records a treatment, not a phenotype, and no cited cutoff applies | 2026-11-25 |
| The SIRS alternatives PaCO2 below 32 and over 10% immature bands, and SOFA's PaO2/FiO2 | The frame names neither `PaCO2` nor a ratio; band counts are no variable of the files | 2026-11-25 |
| Refusing the empty-`Parameter` lines of set C | They are part of the published bytes; refusing them would refuse the set; they are skipped and counted | 2026-11-25 |

## Enforcement

- Owner: `src/bko/ontology/catalog.py`
- Pinned by: nothing -- convention only

## Reversal Conditions

**PhysioNet re-issues a file** with other bytes. **A read of Bone et al. or Vincent et al. differs
from Singer et al.'s reproduction** in a cutoff used here. **The training split falls below 5,000
records**, which ADR `research/confirmatory-dataset-selection` would then revise.

## Related

- [Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page](confirmatory-dataset-selection.md)
- [SUPPORT2's frame: in-hospital death as the label, physiology coded at SIRS and SOFA cutoffs](support2-frame.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
