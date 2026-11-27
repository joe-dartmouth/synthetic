---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-26
---
# SUPPORT2's frame: in-hospital death as the label, physiology coded at SIRS and SOFA cutoffs

## Context

ADR [`research/confirmatory-dataset-selection`](confirmatory-dataset-selection.md) chose SUPPORT2
as the first confirmatory health dataset and left its in-hospital death rate to be counted from
the training split. SUPPORT2 is the Vanderbilt Department of Biostatistics' file of all 9,105
SUPPORT patients ([`vanderbilt-2026`](../../literature.md#datasets)); the SUPPORT prognostic
model paper describes the study: seriously ill hospitalized adults at 5 tertiary care academic
centers in the United States, with diagnosis, age, days in hospital before study entry, cancer,
neurologic function and 11 physiologic measures recorded on day 3
([`knaus-1995`](../../literature.md#datasets), read from its PubMed abstract).

The file is one zip member, `support2.csv`. Its header holds 47 names and every data row 48
fields: the first, unnamed, is a row number. Values carry leading spaces, a missing value is an
empty field, and the categoricals are text. Many measured values are stored truncated in binary,
so a creatinine of 1.2 reads `1.19995117`.

The publisher's page (`hbiostat.org/data/repo/supportdesc.html`, read 2026-11-25) says: "To
develop models without using findings from previous models, be sure not to use aps, sps, surv2m,
surv6m as predictors. You also will probably not want to use prg2m, prg6m, dnr, dnrday."

A mapped lab or vital needs a cited cutoff. The SIRS criteria come from the ACCP/SCCM consensus
conference ([`bone-1992`](../../literature.md#clinical-definitions)) and SOFA from the ESICM
working group ([`vincent-1996`](../../literature.md#clinical-definitions)). Neither original could
be read: the Chest article's open copy sits behind a bot check that refused every route tried,
and the SOFA paper is paywalled. Both are reproduced in the Sepsis-3 consensus paper
([`singer-2016`](../../literature.md#clinical-definitions), read from PubMed Central on
2026-11-25): its Box 1 lists the SIRS criteria "From Bone et al." and its Table 1 the SOFA score,
"Adapted from Vincent et al.". The cutoffs below are read there. A quotation is the page's text
with markup dropped and whitespace runs collapsed to one space.

## Decision Drivers

* **Must keep the label free of leakage**: no column the publisher warns against and no column
  known only at or after the outcome.
* **Must map physiology only through a cited cutoff**, read in its source's text before the row
  is written.
* **Must let a missing value show no concept**, so absence of a measurement is never read as a
  normal or an abnormal finding.
* **Must fix the rows before the training split is counted**, and count the split before any gap
  is read.

## Decision

The pin is `bko.datasets.sources.SUPPORT2`, the loader `bko.datasets.support2`, the frame
`bko.datasets.support2_frame` and the rows `bko.ontology.support2_rows`.

**1. The cohort.** One record per row of `support2.csv`, `record_id` the row number. The one
record whose `scoma` is missing is dropped, and the loader's `cohort` line counts it. The label
`target` is `hospdead`.

**2. The fields.** `age` floored to whole years and `sex` coded `{0: female, 1: male}`, then
`dzgroup` (the codebook's eight levels, in its order), `num.co`, `diabetes`, `dementia`, `ca`
(`no`, `yes`, `metastatic`), `hday`, `scoma`, `race` and `income`, each with code 0 for not
recorded and its codebook levels after it, then the labs and vitals below. `age`, `num.co`,
`hday` and `scoma` are numeric; every other feature is coded. The age bands start at 50, 65 and
80. These columns never enter: `aps`, `sps`, `surv2m`, `surv6m`, `prg2m`, `prg6m`, `dnr` and
`dnrday`, which the publisher warns against; `death`, `slos`, `d.time`, `charges`, `totcst`,
`totmcst`, `avtisst` and `sfdm2`, known only at or after the outcome; `dzclass`, a grouping of
`dzgroup`; and the activity scores `adlp`, `adls` and `adlsc`, partly imputed by the publisher.

**3. Coded states.** A lab or vital, and a numeric field with missing values and no cited cutoff,
is a coded field: `0` = not measured, else `1` plus the number of "at or above" cutoffs at or below
the value plus the number of "above" cutoffs strictly below it. Each value is read at 2 decimals
first, which undoes the file's binary truncation (`1.19995117` reads 1.2). The cited cutoffs, from
Singer et al.'s reproduction:

| Field | Cutoffs | Source as reproduced |
|---|---|---|
| `hrt` | above 90 | Box 1 (SIRS): "Heart rate >90/min" |
| `resp` | above 20 | Box 1 (SIRS): "Respiratory rate >20/min or PaCO2 <32" |
| `temp` | below 36, above 38 | Box 1 (SIRS): "Temperature >38°C or <36°C" |
| `wblc` | below 4, above 12 (thousands) | Box 1 (SIRS): "White blood cell count >12 000/mm3 or" |
| `meanbp` | below 70 | Table 1 (SOFA cardiovascular, score 1): "MAP <70 mm Hg" |
| `crea` | 1.2 and above | Table 1 (SOFA renal, score 1): "1.2–1.9 (110–170)" mg/dL |
| `bili` | 1.2 and above | Table 1 (SOFA liver, score 1): "1.2–1.9 (20–32)" mg/dL |
| `pafi` | below 400 | Table 1 (SOFA respiration, score 1): "<400 (53.3)" |

`wblc` is in thousands (the publisher's fill-in table gives "9 (thousands)"), so 12,000 and 4,000
per cubic millimetre read 12 and 4. `edu`, `sod`, `alb`, `ph`, `glucose`, `bun` and `urine` are
coded at the tertiles of their measured values over the records the frame keeps, once the zero
rule below has made each listed field's zeros not measured, each rounded to 2 decimals; they map
nothing. `bko.datasets.support2.tertile_base` returns each tertile field's kept, zero-masked
values, and the loader logs one `tertile dataset=support2 field=<f> low=<c> high=<c> base=<n>`
line per tertile field before its `cohort` line, `base` the kept records measured in it. Each
tertile is an "at or above" cutoff:

| Field | Lower tertile | Upper tertile | Base |
|---|---|---|---|
| `edu` | 11 | 12 | 7,470 |
| `sod` | 135 | 140 | 9,103 |
| `alb` | 2.5 | 3.3 | 5,733 |
| `ph` | 7.4 | 7.45 | 6,821 |
| `glucose` | 111 | 166 | 4,603 |
| `bun` | 17 | 33 | 4,752 |
| `urine` | 1428.33 | 2595 | 4,242 |

This command derived them from the pinned archive in the data root, printing each field, its base
count and its two tertiles:

    uv run --no-sync python -c 'import io, zipfile; from bko.datasets.fetch import default_data_dir, read_verified; from bko.datasets.sources import SUPPORT2; from bko.datasets.states import tertiles; from bko.datasets.support2 import DECIMALS, MEMBER, parse_support2, tertile_base; base = tertile_base(parse_support2(zipfile.ZipFile(io.BytesIO(read_verified(SUPPORT2, default_data_dir()))).read(MEMBER).decode("utf-8"))); [print(name, len(values), *(round(c, DECIMALS) for c in tertiles(values))) for name, values in base.items()]' </dev/null

At tag `run/d1-12` it printed:

    edu 7470 11.0 12.0
    sod 9103 135.0 140.0
    alb 5733 2.5 3.3
    ph 6821 7.4 7.45
    glucose 4603 111.0 166.0
    bun 4752 17.0 33.0
    urine 4242 1428.33 2595.0

A physiologically implausible recorded 0 is not measured: a 0 in `hrt`, `temp`, `wblc`, `meanbp`,
`crea`, `bili` or `pafi` of the cited fields, or in `sod`, `alb`, `ph`, `glucose` or `bun` of the
tertile fields, any value that reads as 0 at 2 decimals, is masked to missing before the field is
coded, so it codes `0` and shows no concept. These fields are
`bko.datasets.support2.IMPLAUSIBLE_ZERO`. A 0 stays a measured value where it is plausible: `resp`,
which a patient on a ventilator can record with no breath of their own; `urine`, an anuric day;
`edu`, no schooling; and `scoma`, whose 0 is the score's normal. The loader logs one `zeros
dataset=support2 field=<f> readings=<n> records=<n>` line per listed field before its `cohort` line,
a field with no zero logging 0. The zeros, over the published file's 9,105 rows and the 9,104
records the frame keeps:

| Field | Zeros in the file | Zeros in the kept records | Coded |
|---|---|---|---|
| `hrt` | 83 | 83 | not measured |
| `wblc` | 10 | 10 | not measured |
| `meanbp` | 51 | 51 | not measured |
| `glucose` | 1 | 1 | not measured |
| `temp`, `crea`, `bili`, `pafi`, `sod`, `alb`, `ph`, `bun` | 0 | 0 | not measured |
| `resp` | 65 | 65 | measured |
| `urine` | 93 | 93 | measured |
| `edu` | 43 | 43 | measured |

This command counted them, printing each field and its zeros in the file and in the kept records:

    uv run --no-sync python -c 'import io, zipfile; from bko.datasets.fetch import default_data_dir, read_verified; from bko.datasets.sources import SUPPORT2; from bko.datasets.support2 import IMPLAUSIBLE_ZERO, MEMBER, checked, measured, parse_support2; raw = checked(parse_support2(zipfile.ZipFile(io.BytesIO(read_verified(SUPPORT2, default_data_dir()))).read(MEMBER).decode("utf-8"))); kept = raw.loc[raw["scoma"] != ""]; [print(name, int((measured(raw[name]) == 0).sum()), int((measured(kept[name]) == 0).sum())) for name in (*IMPLAUSIBLE_ZERO, "resp", "urine", "edu")]' </dev/null

At tag `run/d1-12` it printed:

    hrt 83 83
    temp 0 0
    wblc 10 10
    meanbp 51 51
    crea 0 0
    bili 0 0
    pafi 0 0
    sod 0 0
    alb 0 0
    ph 0 0
    glucose 1 1
    bun 0 0
    resp 65 65
    urine 93 93
    edu 43 43

**4. The rows.** One `Codes` row per mapped state, labels the pinned releases' names:

- `dzgroup` COPD, CHF, Cirrhosis, Colon Cancer and Lung Cancer to MONDO's chronic obstructive
  pulmonary disease, congestive heart failure, cirrhosis of liver, malignant colon neoplasm and
  lung cancer; Coma to HPO's Coma. The two multi-organ levels map nothing.
- `diabetes` 1 to diabetes mellitus, `dementia` 1 to dementia, `ca` `yes` or `metastatic` to
  cancer, all MONDO.
- The abnormal states to HPO: `hrt` above 90 Tachycardia; `resp` above 20 Tachypnea; `temp`
  above 38 Fever and below 36 Hypothermia; `wblc` above 12 Increased total leukocyte count and
  below 4 Decreased total leukocyte count; `meanbp` below 70 Hypotension; `crea` 1.2 and above
  Elevated circulating creatinine concentration; `bili` 1.2 and above Hyperbilirubinemia; `pafi`
  below 400 Hypoxemia.
- `scoma`, SUPPORT's own coma score and not the Glasgow scale SOFA reads, maps nothing.

Each cutoff row's source is `singer-2016`, and its limits name the original (`bone-1992` or
`vincent-1996`) and say the cutoff was read in Singer et al.'s reproduction.

**5. The order.** The rows are committed; then the training split
(`split(records, seed=0, edges=frame.edges).train`) is counted; then the gap listing and the
census are read. Consequences record each step's commit.

**6. The pin.** The Vanderbilt archive, `agreement=None`, as ADR
`research/confirmatory-dataset-selection` point 4 fixes.

## Rationale

Bands at cited cutoffs keep each row's meaning the clinical definition's, and state 0 keeps a
missing measurement apart from both bands, so a record with no creatinine shows neither elevated
creatinine nor its absence. The tertile coding keeps unmapped measurements in the frame without
inventing a cutoff and without an imputed value. Reading at 2 decimals is needed because a cutoff
written as "1.2–1.9" would otherwise put the file's `1.19995117`, a recorded 1.2, below it.

No living patient has a heart rate, a mean arterial pressure, a temperature, a white cell count, a
creatinine, a bilirubin, a PaO2/FiO2 ratio, a sodium, an albumin, a pH, a glucose or a urea of 0,
so a 0 there is a placeholder for a value not recorded; read as measured, a `meanbp` of 0 would
show Hypotension and a `wblc` of 0 Decreased total leukocyte count from no reading at all. Taking
the tertiles over the records the frame keeps, with those zeros masked, cuts each field where the
coded records' values fall, so each tertile state holds about a third of the records it codes,
short of ties.

Singer et al.'s reproduction is a consensus paper co-authored by SOFA's first author, and it
quotes both tables as its sources give them; reading the cutoffs there is the closest readable
text to the originals. Hypoxemia is the HPO term for an abnormally low blood oxygen level, which a
PaO2/FiO2 below 400 reads as on room air.

## Consequences

**The order, by commit.** The rows are at `cb15592` (Add the SUPPORT2 loader, frame and mapping
rows). The training split's count is recorded in `7366752`, before any gap listing or census
ran: `CATALOG["support2"].read` on the data root, then `split(records, seed=0,
edges=frame.edges).train`. The cohort holds 9,104 records (9,105 rows, one dropped for a missing
`scoma`, 2,359 in-hospital deaths); the training split holds 7,282 records, 1,887 of them
in-hospital deaths, an in-hospital death rate of 0.2591, digest
`a9231c19529069a9bd0cc94a02209bac5c1242baf8da610f4dc93eca868347e7`. The split is above the 5,000
records condition 5 asks. The first readings are recorded in `de50f71` and the current ones in
`bbc698b`. These are commits of pull request #135, kept at `refs/pull/135/head` after its squash
merge.

**The readings, at `ea2c11a`, after the count.** `uv run --no-sync python -m bko.ontology check`
exited 0, its log ending `checked 89 rows: 0 disagree`. The gap listing is run `d1-4`, `uv run
--no-sync python -m bko.gaps support2` into `runs/d1-4/gaps-support2.csv` and
`runs/d1-4/gaps-support2.log`, exit 0, closing `gaps rows=298
sha256=6dbe1c132a94640225fd0d74666a390d0b9b3537707b80feac67143154003d20`: per-subgroup 57 gaps in
3 cells, population-level none, frequency-aware 241 in 15 cells, and no cell below the count
threshold. Its csv sha256 is `6dbe1c132a94640225fd0d74666a390d0b9b3537707b80feac67143154003d20`
and its log sha256 `0956ae66175462a3b798354c8485b4d8c720cadea4ca507089d66987987dc535`; its
`data` line gives the 9,104 records' digest
`69c36f36f1b63e083d4506c02c25ab87f5340d4d8a051d695773821f47310bcc` and its `split` line the test
fingerprint `c076b78222bd9fc1021a0f7a4af77732b2f5b3229b897ef026875d171d64cae6`. The census is
run `d2-4`, `uv run --no-sync python -m bko.evaluate contradictions --dataset support2` into
`runs/d2-4/contradictions.csv` and `runs/d2-4/contradictions.log`, exit 0: every `census kind=`
count is 0 (`narrow-without-broad`, `child-share-above-parent`, `empty-cell`, `all-show`,
`code-overlap`), over 16 of 16 cells, 19 rows and two `is_a` pairs, lung cancer and malignant
colon neoplasm under cancer, shown by 733 and 411 records, none of them without cancer. Its csv
sha256 is `b8fbd7f8c95b38fa93262f7e8f92dfd28ed305a3e784342d1b32db9707aa0056` (the header alone)
and its log sha256 `fcae4dfdf7a8062b7dccd80a8d695d63372f481bd8bd37971e9e85250e0b61c9`. Both read
the archive sha256 `8ed43980742a18e1847a8dfc5530bc4b30564ad9e4ad1b1b50bbc5d29d8c86fe`, HPO
`hp/releases/2026-09-01` and MONDO `releases/2026-10-06`. The first readings, runs `d1-3` and
`d2-3` at `7366752`, wrote the same csv bytes; their logs lack the `data` and `split` lines and
the `cohort` line's `dataset=`.

**Choices beside the rows.** The cutoffs are read in Singer et al.'s reproduction, because
neither original could be read (Context). The one record with a missing `scoma` is dropped
rather than coded, because coding it at its tertiles would make a missing flag (Ruled Out). Each
measured value is read at 2 decimals, and the tertile cutoffs are rounded to 2 decimals, because
the file stores values truncated in binary (Decision 3). The frame builder lives in its own
module beside the loader, which keeps the loader under the 500-line cap. The census reads
SUPPORT2 through `--dataset support2`, which keeps the default census's three frames and its
recorded bytes (Ruled Out). The zero rule and the kept-record tertiles change the frame under its
own name, `support2`, because no `bko.evaluate.named_runs.RUNS` entry reads it, so no scored run
folder is re-read under the frame's identity checks; the `d1-4` and `d2-4` readings above are of
the frame before the change.

**The readings at tag `run/d1-12`**, commit `633c0cf`, the frame under the zero rule and the
kept-record tertiles. The 9,104 records' digest is
`4a24967bdbd09cdae8950ed8762a947c2170f5b22cd6d2e2d2c6d53f2b4aaeae` (each run's `data` line), where
the frame before the change gave `69c36f36f1b63e083d4506c02c25ab87f5340d4d8a051d695773821f47310bcc`;
the training split's 7,282 records' digest is
`f82a8bc67f036f1580fdd99ce8dbbccdd63bfc633cd1ab4f52876e19ee8e2b6b` (the census's `census data`
line), where it was `a9231c19529069a9bd0cc94a02209bac5c1242baf8da610f4dc93eca868347e7`. The test
fingerprint stays `c076b78222bd9fc1021a0f7a4af77732b2f5b3229b897ef026875d171d64cae6` and the
`cohort` line stays `rows=9105 missing_scoma=1 records=9104 hospital_deaths=2359`. Each run ran
from a detached worktree at the tag as `env PYTHONPATH=<worktree>/src .venv/bin/python -m
<command> --data-dir data`, exit 0, each log's `versions` line reading `dirty=false`. The gap
listing is run `d1-12`, `bko.gaps support2` into `runs/d1-12/gaps-support2.csv` and
`runs/d1-12/gaps-support2.log`, closing `gaps rows=298
sha256=6dbe1c132a94640225fd0d74666a390d0b9b3537707b80feac67143154003d20`, byte for byte `d1-4`'s
csv; its log sha256 is `34b5cd8996cd881f91b3382cb9a77f2b95e92e18af34bf1f922ace35e498203b` and
holds the `zeros` and `tertile` lines of Decision 3. The census is run `d2-12`, `bko.evaluate
contradictions --dataset support2` into `runs/d2-12/contradictions.csv` and
`runs/d2-12/contradictions.log`: every `census kind=` count is 0, over 16 of 16 cells and the 19
rows of `d2-4` (digest `3c818f27943faaf09b58a1ef9bf151f1b40e5f472a63e93af847795e76d5cecf`), the
two `is_a` pairs shown by 733 and 411 records, none without cancer. Its csv sha256 is
`b8fbd7f8c95b38fa93262f7e8f92dfd28ed305a3e784342d1b32db9707aa0056` (the header alone) and its log
sha256 `b7f9f26f828cc0d9776b1e08204fe6d9dc3a3ceb85dbbb0fecfc8e9360b0d204`. At the tag `uv run
--no-sync python -m bko.ontology check` exited 0, its log ending `checked 821 rows: 0 disagree`,
and `tests/test_datasets_support2_pinned.py` passed with `--releases`, run from a worktree at the
tag.

**Gained**: a second positive dataset whose diagnoses and physiology reach MONDO and HPO through
fixed rows. **Accepted**: one record leaves the cohort for a missing `scoma`. The originals of the
two scores stay unread; a re-read that differs from Singer et al.'s reproduction would change a
row. Tertile bands can be uneven where values tie at a tertile (`edu`'s 11 and 12).

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Reading the SIRS and SOFA cutoffs in Bone et al. and Vincent et al. themselves | Neither could be read: the Chest copy refused every route behind a bot check, and the SOFA paper is paywalled; Singer et al. reproduce both | 2026-11-25 |
| Coding `scoma` at its tertiles, as for the other numeric fields with missing values | Both tertiles are 0, so the field would collapse into a missing flag; its one missing record is dropped instead | 2026-11-25 |
| Imputing missing physiology with the publisher's normal fill-in values | An imputed normal value would show the absence of a finding no one measured | 2026-11-25 |
| The SIRS alternatives PaCO2 below 32 and over 10% immature bands | Neither is a column of the file | 2026-11-25 |
| A colon cancer concept named `colon cancer` | The pinned MONDO has no term of that name; malignant colon neoplasm, its class for malignant neoplasms of the colon, maps instead | 2026-11-25 |
| A metastatic-cancer row on `ca` `metastatic` | `ca` `metastatic` shows cancer through the `ca` row, and the rows fixed before the count (Decision 5) hold no metastatic-cancer row | 2026-11-25 |
| `urine` at SOFA's renal cutoff, below 500 mL/d | The file states no period for `urine`, so it stays at tertiles and maps nothing | 2026-11-25 |
| SUPPORT2 in the default census | It would change the bytes of EVIDENCE D2's recorded run and make the bare command fetch another archive; the census takes `--dataset` | 2026-11-25 |
| Reading a recorded 0 as a measured value | A 0 in a listed field is no physiological reading, and it would show Hypotension or Decreased total leukocyte count from no measurement (Rationale) | 2026-11-26 |
| Tertiles over every record of the published file | They take in the record the frame drops and the implausible zeros, so the cuts are not those of the records they code (Rationale) | 2026-11-26 |

## Enforcement

- Owner: `src/bko/ontology/support2_rows.py`
- Pinned by: `tests/test_datasets_support2.py::test_the_frame_holds_exactly_the_named_fields`,
  `tests/test_datasets_support2.py::test_the_excluded_columns_never_enter_the_frame`,
  `tests/test_datasets_support2.py::test_each_cited_field_is_coded_at_its_cutoffs`,
  `tests/test_datasets_support2.py::test_a_missing_value_shows_no_concept`,
  `tests/test_datasets_support2.py::test_a_value_that_reads_0_in_each_listed_field_codes_not_measured`,
  `tests/test_datasets_support2.py::test_a_zero_in_an_unlisted_field_still_codes_as_measured`,
  `tests/test_datasets_support2.py::test_tertile_base_drops_listed_zeros_and_unkept_records`,
  `tests/test_datasets_support2.py::test_an_unmasked_zero_would_move_the_tertiles`,
  `tests/test_datasets_support2.py::test_a_missing_scoma_drops_its_record_and_is_counted`,
  `tests/test_datasets_support2.py::test_each_loader_logs_a_zero_count_per_listed_field`,
  `tests/test_datasets_support2.py::test_each_loader_logs_each_tertile_fields_cuts_and_base`,
  `tests/test_datasets_states.py::test_log_tertile_logs_the_cuts_at_the_decimals_and_the_base`,
  `tests/test_ontology_support2_rows.py::test_the_rows_map_these_states_to_these_concepts`,
  `tests/test_ontology_support2_rows.py::test_every_cutoff_row_cites_a_registered_source`,
  `tests/test_datasets_support2_pinned.py` (with `--releases`)

## Reversal Conditions

**A read of Bone et al. or Vincent et al. differs from Singer et al.'s reproduction** in a cutoff
used here. **The publisher re-issues `support2.csv`** with other bytes. **The training split
falls below 5,000 records**, which ADR `research/confirmatory-dataset-selection` condition 5
excludes.

## Related

- [Confirmatory datasets are chosen by a fixed rule before any record is loaded](confirmatory-dataset-selection.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
- [Each stage keeps a data contradiction and reports it](data-contradiction-handling.md)
