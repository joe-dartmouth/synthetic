---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
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
coded at the tertiles of their measured values over every record of the published file, each
rounded to 2 decimals; they map nothing.

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

Singer et al.'s reproduction is a consensus paper co-authored by SOFA's first author, and it
quotes both tables as its sources give them; reading the cutoffs there is the closest readable
text to the originals. Hypoxemia is the HPO term for an abnormally low blood oxygen level, which a
PaO2/FiO2 below 400 reads as on room air; its row's limits say the ratio is not PaO2 alone.

## Consequences

**The order, by commit.** The rows were committed in `cb15592` (Add the SUPPORT2 loader, frame
and mapping rows). At `14be75a`, before any gap listing or census, the training split was counted
by a scratch script that runs `CATALOG["support2"].read` on the data root, then
`split(records, seed=0, edges=frame.edges).train`: the cohort holds 9,104 records (9,105 rows,
one dropped for a missing `scoma`, 2,359 in-hospital deaths); the training split holds 7,282
records, 1,887 of them in-hospital deaths, an in-hospital death rate of 0.2591, digest
`a9231c19529069a9bd0cc94a02209bac5c1242baf8da610f4dc93eca868347e7`. The split is above the 5,000
records condition 5 asks.

**The readings, at `7366752`, after the count.** `uv run --no-sync python -m bko.ontology check`
exited 0, its log ending `checked 89 rows: 0 disagree`. The gap listing is run `d1-3`, `uv run
--no-sync python -m bko.gaps support2` into `runs/d1-3/gaps-support2.csv` and
`runs/d1-3/gaps-support2.log`, exit 0, closing `gaps rows=298
sha256=6dbe1c132a94640225fd0d74666a390d0b9b3537707b80feac67143154003d20`: per-subgroup 57 gaps in
3 cells, population-level none, frequency-aware 241 in 15 cells, and no cell below the count
threshold. Its csv sha256 is `6dbe1c132a94640225fd0d74666a390d0b9b3537707b80feac67143154003d20`
and its log sha256 `c535f7992ec9ecc6eeb5f64b8dc8606d6729762511f3f290e9b321dc6ee82003`. The census
is run `d2-3`, `uv run --no-sync python -m bko.evaluate contradictions --dataset support2` into
`runs/d2-3/contradictions.csv` and `runs/d2-3/contradictions.log`, exit 0: every `census kind=`
count is 0 (`narrow-without-broad`, `child-share-above-parent`, `empty-cell`, `all-show`,
`code-overlap`), over 16 of 16 cells, 19 rows and two `is_a` pairs, lung cancer and malignant
colon neoplasm under cancer, shown by 733 and 411 records, none of them without cancer. Its csv
sha256 is `b8fbd7f8c95b38fa93262f7e8f92dfd28ed305a3e784342d1b32db9707aa0056` (the header alone)
and its log sha256 `7f29f287c6e5827631ebbf0cf7ba2908aceb8bf034d6b48db09e83ebba55eade`. Both read
the archive sha256 `8ed43980742a18e1847a8dfc5530bc4b30564ad9e4ad1b1b50bbc5d29d8c86fe`, HPO
`hp/releases/2026-09-01` and MONDO `releases/2026-10-06`.
`tests/test_datasets_support2_pinned.py` passes with `--releases`.

**Deviations from the plan.** The cutoffs are read in Singer et al.'s reproduction, not in Bone
et al. and Vincent et al. themselves (Context). The one record with a missing `scoma` is dropped
rather than coded (Ruled Out). Each measured value is read at 2 decimals, and the tertile cutoffs
are rounded to 2 decimals (Decision 3). The frame builder lives in its own module beside the
loader, which would otherwise pass the 500-line cap. The census reads SUPPORT2 through
`--dataset support2`; the default census keeps its three frames.

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
| A metastatic-cancer row on `ca` `metastatic` | Not among the rows fixed in the mapping commit; `ca` `metastatic` shows cancer | 2026-11-25 |
| `urine` at SOFA's renal cutoff, below 500 mL/d | The file states no period for `urine`, so it stays at tertiles and maps nothing | 2026-11-25 |
| SUPPORT2 in the default census | It would change the bytes of EVIDENCE D2's recorded run and make the bare command fetch another archive; the census takes `--dataset` | 2026-11-25 |

## Enforcement

- Owner: `src/bko/ontology/support2_rows.py`
- Pinned by: `tests/test_datasets_support2.py::test_the_frame_holds_exactly_the_named_fields`,
  `tests/test_datasets_support2.py::test_the_excluded_columns_never_enter_the_frame`,
  `tests/test_datasets_support2.py::test_each_cited_field_is_coded_at_its_cutoffs`,
  `tests/test_datasets_support2.py::test_a_missing_value_shows_no_concept`,
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
