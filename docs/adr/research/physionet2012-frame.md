---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-26
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
stay are refused. A stay must be ASCII text, end in a newline, start with the header, and hold
three fields per line, a time from `00:00` to `48:00`, and a finite decimal number: digits with an
optional fraction, or the exponent form the files hold, such as `1.213e+04`. `RecordID`, `Age`,
`Gender`, `Height` and `ICUType` each appear once, at `00:00`; `Age` must be whole, `Gender` 0,
1 or -1, `ICUType` 1 to 4, and each `MechVent` value 0, 1 or -1. A `Parameter` that is none of
the six descriptors and 37 time series is refused, except an empty one, which names no variable:
its line is skipped and counted.

**3. The label and the cohort.** `target` is `In-hospital_death`, 0 or 1. A stay whose `Age` or
`Gender` is -1 is left out and counted. `age` is `Age`; `sex` is `Gender`, 0 female and 1 male.
Age bands start at 50, 65 and 80.

**4. The reduction rule.** Per stay and variable, over the whole 48 hours and over the values at
or above 0: the minimum and the maximum. A value below 0, -1 or another, is not a reading; of the
variables the frame reads, the lines of -1 are counted as missing and the lines of another value
below 0 apart. Only `MechVent`'s values are checked against codes; no value is checked against a
range. A 0 of `HR`, `Temp`, `WBC`, `MAP`, `NIMAP`, `Creatinine`, `Bilirubin`, `Platelets` or
`GCS`, whose scale runs from 3, is physiologically implausible and is not a reading either:
`parse_stay` does not append it and counts it in `Stay.zeros`, and the loader logs one `zeros
dataset=physionet2012 field=<variable> readings=<n> records=<n>` line per listed variable before
its `cohort` line, a variable with no zero logging 0. These variables are
`bko.datasets.physionet2012.IMPLAUSIBLE_ZERO`. A 0 of `RespRate` stays a reading, since a patient
on a ventilator can record no breath of their own, and `MechVent` is a flag. A stay with no
reading of a variable did not measure it.

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
| `map_min` | `MAP` and `NIMAP` together, minimum | below 70 | Table 1 (SOFA cardiovascular, score 1): "MAP <70 mm Hg" |
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
negative, so `Temp`'s -17.8 is not a reading. No living patient has a heart rate, a temperature, a
white cell or platelet count, a mean arterial pressure, a creatinine or a bilirubin of 0, and the
Glasgow scale starts at 3, so a 0 there is no reading; read as one, a `Temp` of 0 would show
Hypothermia and a `MAP` of 0 Hypotension from no measurement.

`map_min` reads `MAP` and `NIMAP` together because the page lists both as readings of one
quantity, and "MAP <70 mm Hg" holds for a stay when any of its readings is below 70, as every
other field takes its minimum over all its readings. Reading `NIMAP` only where the stay has no
`MAP` would leave a stay without Hypotension though its cuff pressure fell below 70.
`mechvent` keeps the page's own variable as a free field: the files record only 1, so its absence
means no ventilation was recorded, not a reading of 0. The empty-`Parameter` lines name no
variable the page lists, so no field could hold them; counting them keeps the skip visible.

## Consequences

**The order, by commit.** The reduction rule and the rows are fixed in `ce0f090` (Record the
PhysioNet 2012 frame ADR and its reduction rule) and coded at `9aef4a3` (Add the PhysioNet 2012
loader, frame, field rows and catalog entry). The training split's count is recorded in the
commit after `9aef4a3`, before any gap listing or census ran: `CATALOG["physionet2012"].read` on
the data root, then `split(records, seed=0, edges=frame.edges).train`, as this command does:

    uv run --no-sync python -c 'from bko.datasets.fetch import default_data_dir, open_https; from bko.datasets.frames import digest; from bko.datasets.split import split; from bko.ontology.catalog import CATALOG; records, frame = CATALOG["physionet2012"].read(default_data_dir(), open_url=open_https); train = split(records, seed=0, edges=frame.edges).train; print(len(records), int(records["target"].sum()), len(train), int(train["target"].sum()), round(float(train["target"].mean()), 4), digest(train))' </dev/null

Run at `9aef4a3`, it printed `11988 1706 9590 1364 0.1422
b99bf9acdc084ad965060ed868b24db828fe216bc08260c45544536a203b72e2`. The loader's `cohort` line
reads `stays=12000 unknown_age=0 unknown_sex=12 left_out=12 records=11988 died=1706 unnamed=2403
below_zero=59`: the 12 stays of unknown `Gender` leave the cohort, the 2,403 empty-`Parameter`
lines of set C are skipped, and the 59 `Temp` values below 0 are no readings. The cohort holds
11,988 records, 1,706 of them in-hospital deaths. The training split holds 9,590 records, 1,364
of them in-hospital deaths, an in-hospital death rate of 0.1422, beside the challenge paper's
"The overall mortality rate was 14.2%." over all 12,000 stays (`silva-2012b`). The split is above
the 5,000 records condition 5 of ADR `research/confirmatory-dataset-selection` asks.

The frame these first readings read took a recorded 0 as a reading. Over its 11,988 records, 102
stays had a lowest `Temp` of 0, so showed Hypothermia; 108 a lowest mean arterial pressure of 0,
so showed Hypotension; one a lowest `WBC` of 0, so showed Decreased total leukocyte count; one a
highest `Bilirubin` of 0; and none a highest `HR` or `RespRate` of 0. This command counted them on
that frame's code, where `Stay.reduced` took one variable's name:

    uv run --no-sync python -c 'import re; from bko.datasets.archive import members, tar_members; from bko.datasets.fetch import default_data_dir, read_verified; from bko.datasets.physionet2012 import SETS, TARRED, parse_stay; from bko.datasets.sources import PHYSIONET2012; stays = [parse_stay(n, t) for label, s in zip(SETS, PHYSIONET2012) for n, t in (tar_members if label in TARRED else members)(read_verified(s, default_data_dir()), re.compile(rf"set-{label}/\d+\.txt"), s)]; kept = [s for s in stays if -1.0 not in (s.age, s.gender)]; [print(v, h, sum(s.reduced(v, h) == 0 for s in kept)) for v, h in (("Temp", "min"), ("MAP", "min"), ("HR", "max"), ("RespRate", "max"), ("WBC", "min"), ("Bilirubin", "max"))]' </dev/null

**The readings, at `258b3b6`, after the count.** `uv run --no-sync python -m bko.ontology check`
exited 0, its log ending `checked 821 rows: 0 disagree`. The gap listing is run `d1-10`, `uv run
--no-sync python -m bko.gaps physionet2012` into `runs/d1-10/gaps-physionet2012.csv` and
`runs/d1-10/gaps-physionet2012.log`, exit 0, closing `gaps rows=42
sha256=28756461746301bf969283d95d5a6e9cd24db5ccb110a64b961a19a38b4a4235`: per-subgroup none,
population-level none, frequency-aware 42 rows in 7 cells, and no cell below the count threshold
of 20. Its csv sha256 is `28756461746301bf969283d95d5a6e9cd24db5ccb110a64b961a19a38b4a4235` and
its log sha256 `81bc5fd530b77d90a940c4de6e226a3265d2c0e838bdeee1f85eb626b77fe0c2`; its `data`
line gives the 11,988 records' digest
`5ca824f645d6a37af0aa99c29d37556f8994ba3cde2941cdac61b807b0678f48` and its `split` line the test
fingerprint `0974dd102fc84ae567c22dda7061f5127ec5c0bb23a1ceb31db18734b402b4be`. The census is run
`d2-10`, `uv run --no-sync python -m bko.evaluate contradictions --dataset physionet2012` into
`runs/d2-10/contradictions.csv` and `runs/d2-10/contradictions.log`, exit 0, over 16 of 16 cells
and 11 field rows, none of whose concepts descends from another by `is_a`: every `census kind=`
count is 0 (`narrow-without-broad`, `child-share-above-parent`, `empty-cell`, `all-show`,
`code-overlap`). Its csv sha256 is
`b8fbd7f8c95b38fa93262f7e8f92dfd28ed305a3e784342d1b32db9707aa0056` (the header alone) and its
log sha256 `4f411770387c691063dc3f3f1ac2734898b88b4f3fdfd2b9833c57a7fe7732c0`. Both read the six
pinned files and HPO `hp/releases/2026-09-01`, and each log's `versions` line reads
`dirty=false`. `uv run --no-sync pytest tests/test_datasets_physionet2012_pinned.py --releases -q
-p no:cacheprovider --no-cov` passed. These are commits of pull request #140, kept at
`refs/pull/140/head` after its squash merge; [`EVIDENCE.md`](../../EVIDENCE.md) keeps the runs'
rows.

**The readings again, at `fbd6be9`**, after the loader checked each value as an ASCII decimal
and its `cohort` line came to count each set's stays, the read values of -1 apart from the others
below 0, and the records whose `map_min` reads `NIMAP`. The gap listing is run `d1-11` and the
census run `d2-11`, the commands of `d1-10` and `d2-10` into their own folders, each exit 0 from
a clean tree, each log's `versions` line reading `dirty=false`. Each csv is byte-identical to the
earlier run's: `28756461746301bf969283d95d5a6e9cd24db5ccb110a64b961a19a38b4a4235` and
`b8fbd7f8c95b38fa93262f7e8f92dfd28ed305a3e784342d1b32db9707aa0056`; the `data` line gives the same
records digest, so the records did not change. The logs' sha256s are
`782e98cf266cf6df88cd307b7f1d13da74ba4436ce5ad313e64d865b9fdf7371` and
`bc43ee1a38f76890b21c0c621bcf6a0e29eb81f77c351b8deca9d06eee9f6603`, each differing from the
earlier log in its `versions` and `cohort` lines only. The `cohort` line reads `stays=12000
stays_a=4000 stays_b=4000 stays_c=4000 unknown_age=0 unknown_sex=12 left_out=12 records=11988
died=1706 unnamed=2403 missing=0 below_zero=59`, then the `measured_` counts of `d1-10`, then
`map_from_nimap=3406 mechvent=7576`: no value of a read variable is -1, set B holds 4,000 stays
as A and C do, and 3,406 of the 11,988 records read `map_min` from `NIMAP`. At `fbd6be9`, `uv run
--no-sync python -m bko.ontology check` exited 0, its log ending `checked 821 rows: 0 disagree`,
and the pinned test passed with `--releases`.

**The zero rule and `map_min` over both readings change the frame under its own name**,
`physionet2012`, because no `bko.evaluate.named_runs.RUNS` entry reads it, so no scored run
folder is re-read under the frame's identity checks. The `d1` and `d2` readings above are of the
frame before the change, and the readings after it are recorded beside the `run/<id>` tag at
which it changed. The loader's `cohort` line counts, in place of `map_from_nimap`, the records
whose `map_min` is a `NIMAP` reading with no `MAP` reading (`map_min_nimap`), the records holding
both whose lowest `NIMAP` is the lower (`map_min_nimap_lower`) and the records with no `MAP` or
`NIMAP` reading (`map_unmeasured`).

**Gained**: a second intensive-care frame beside SUPPORT2's, with the same SIRS and SOFA rows read
from a 48-hour window. **Accepted**: the minimum and the maximum follow the outliers the page warns
of, so a single spurious reading can show a finding, and an artifact of either pressure line
above 0 can show Hypotension. A `GCS` below 15 under sedation reads as Reduced consciousness, as
SOFA reads it. Of the 11,988 records, 7,049 hold values of both `MAP` and `NIMAP` at or above 0:
653 of them have a lowest `MAP` of 70 or more but a lowest `NIMAP` below 70, which `map_min` over
both readings shows as Hypotension; 67 have a lowest `MAP` of 0, likely an arterial-line
artifact, which the zero rule reads as no reading. This command counted them over values at or
above 0:

    uv run --no-sync python -c 'import re; from bko.datasets.archive import members, tar_members; from bko.datasets.fetch import default_data_dir, read_verified; from bko.datasets.physionet2012 import SETS, TARRED, parse_stay; from bko.datasets.sources import PHYSIONET2012; stays = [parse_stay(n, t) for label, s in zip(SETS, PHYSIONET2012) for n, t in (tar_members if label in TARRED else members)(read_verified(s, default_data_dir()), re.compile(rf"set-{label}/[0-9]+\.txt"), s)]; both = [(min(s.values["MAP"]), min(s.values["NIMAP"])) for s in stays if -1.0 not in (s.age, s.gender) and s.values["MAP"] and s.values["NIMAP"]]; print(len(both), sum(m >= 70 > n for m, n in both), sum(m == 0 < n for m, n in both))' </dev/null

It printed `7049 653 67` at `4c0f6c5`.

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
| Reading a recorded 0 as a valid value | A 0 of a listed variable is no physiological reading, and it would show Hypothermia or Hypotension from no measurement (Rationale) | 2026-11-26 |
| Reading `NIMAP` only where the stay has no `MAP` | It leaves 653 stays without Hypotension though a cuff reading fell below 70; the criterion holds when any reading crosses it (Rationale) | 2026-11-26 |

## Enforcement

- Owner: `src/bko/ontology/physionet2012_rows.py`
- Pinned by: `tests/test_datasets_physionet2012.py::test_each_variable_is_reduced_to_its_minimum_and_maximum`,
  `tests/test_datasets_physionet2012.py::test_each_coded_field_is_at_its_cutoffs`,
  `tests/test_datasets_physionet2012.py::test_minus_1_is_missing`,
  `tests/test_datasets_physionet2012.py::test_map_min_reads_map_and_nimap_together`,
  `tests/test_datasets_physionet2012.py::test_a_zero_in_each_listed_field_codes_not_measured`,
  `tests/test_datasets_physionet2012.py::test_a_stay_whose_only_reading_is_zero_is_not_measured`,
  `tests/test_datasets_physionet2012.py::test_mechvent_is_any_1`,
  `tests/test_datasets_physionet2012.py::test_a_stay_with_unknown_age_or_gender_is_left_out_and_counted`,
  `tests/test_datasets_physionet2012.py::test_the_severity_scores_never_enter_the_frame`,
  `tests/test_datasets_physionet2012.py::test_the_frame_holds_exactly_the_named_fields`,
  `tests/test_datasets_physionet2012.py::test_a_missing_value_shows_no_concept`,
  `tests/test_evaluate_physionet2012.py::test_the_rows_map_these_states_to_these_concepts`,
  `tests/test_datasets_archive_members.py` and `tests/test_datasets_physionet2012_pinned.py` (with
  `--releases`)

## Reversal Conditions

**PhysioNet re-issues a file** with other bytes. **A read of Bone et al. or Vincent et al. differs
from Singer et al.'s reproduction** in a cutoff used here. **The training split falls below 5,000
records**, which ADR `research/confirmatory-dataset-selection` would then revise.

## Related

- [Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page](confirmatory-dataset-selection.md)
- [SUPPORT2's frame: in-hospital death as the label, physiology coded at SIRS and SOFA cutoffs](support2-frame.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
