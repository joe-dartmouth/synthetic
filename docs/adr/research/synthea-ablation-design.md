---
status: accepted
topic: research
created: 2026-10-15
updated: 2026-10-26
---
# The Synthea ablation: a pinned generated population, with whole patients removed

## Context

E2 tests gap detection on a Synthea population with selected cases removed, so the removed cases
are a known answer (`docs/RESEARCH.md` section 2). The published pre-generated samples carry no
seed, and the largest one with a fixed name, `synthea_sample_data_csv_apr2020.zip`, holds 1171
patients with 61 coronary heart disease condition rows, too few for any age-sex cell to hold 20
cases before anything is removed.

Synthea recreates a population only from the same release given the same `-s`, `-cs`, `-r` and
`-e` (its wiki, Recreating a Dataset); each defaults to the wall clock. Its CSV dates follow the
JVM time zone, multithreaded runs write rows in completion order, and `exporter.years_of_history`
drops conditions resolved before its window. v4.0.0 codes ischemic heart disease as SNOMED-CT
414545008 and starts it between ages 40 and 79; the code 53741008 is not emitted from v3.2.0 on.

## Decision Drivers

* **Must give a known set of removed cases** for every gap definition and the count-based detector.
* **Must record an identity that recreates the population** (`docs/EVIDENCE.md`).
* **Must leave no trace of a removed case** in the loaded tables.

## Decision

- **Generate** with the Synthea v4.0.0 jar whose size and sha256 `bko.datasets.sources` holds, on
  JDK 17 or newer, passing `-s`, `-cs` and `-p`, `-r` and `-e` both equal to the snapshot date (a
  past date), the state, CSV export of `patients.csv` and `conditions.csv` only,
  `exporter.years_of_history=0`, `generate.thread_pool_size=1`, and the machine's settings fixed:
  `TZ=UTC`, `LANG` and `LC_ALL` `C.UTF-8`, and the JVM's time zone, locale (`en`, `US`) and file
  encoding (UTF-8). Java gets no other environment variable.
- **Reuse** a population only while its `run.json` records these settings and each CSV still
  matches its recorded sha256; otherwise it is generated again.
- **The population is the patients alive on the snapshot date.**
- **A case is a patient**: the ablation removes every patient who has one of the spec's condition
  codes by the snapshot date, with all their condition rows. The spec narrows the cases to one age
  band, one sex, both, or neither; neither removes the condition from the whole population.
- **Removal is complete** within the spec: a spec removes whole patients, all the cases it selects
  or a seeded fraction of them (`AblationSpec.fraction`, drawn with `AblationSpec.seed`, rounded up
  so a positive fraction removes one case at least). Fractions of one spec at one seed share its
  draw, so each removes the cases of every smaller fraction and more.
- **Ground truth** is the removed patient ids and the cell counts before and after.

`bko.datasets.synthea_run`, `bko.datasets.synthea` and `bko.datasets.ablation` hold the values. The
command's default spec is a demonstration cell; E2's configuration fixes its own target cells and
whether they are cells or whole conditions.

The default size comes from a pilot of these settings at `-p 1000`, seed 1, on JDK 17: two runs
gave byte-identical CSVs, and its largest ischemic heart disease cell between ages 40 and 79 was men
aged 60 to 69, with 20 cases. The default population is that cell's size scaled to 27.9 expected
cases, the Poisson mean at which a cell holds 20 or more with probability 0.95, rounded up to a
thousand: 2000. That cell is the default spec.

## Rationale

Generating lets the population size put a target cell above 20 before ablation, so the count-based
detector's finding is the ablation and not natural sparsity, and the pinned parameters are the
identity. Removing condition rows alone would keep the patient as a non-case, which inflates the
cell of patients without the condition. Both grounding definitions of ADR
`research/bko-formal-gap-definition` ground a statement from one remaining record, of the cell or
of the population, so a partial removal is visible to them only where it removes every case of a
cell (per-subgroup) or of the population (population-level), which a fraction below 1 can do in a
cell with few cases; the frequency-aware score reads shares and could see it, and the count-based
detector reads the cell's fallen count; E2's fraction ablations measure all four.
Population-level grounding, over an ontology with no age- or sex-specific statement, sees only a
whole condition's absence, and per-subgroup grounding a whole cell's, which is why the spec can
narrow to a cell or drop the band and the sex.

## Consequences

**Gained**: a population recreated from its parameters; a ground truth every detector reads: the
count-based one by the cells whose count fell, each gap definition by the (cell, statement) pairs
whose statement the removed patients of the cell ground.
**Accepted**: a JDK on every machine that generates; a jar download of the size
`bko.datasets.sources` pins; one thread, so
generation is slower; cross-platform identity is not promised, so each run records its CSV
checksums.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The apr2020 pre-generated sample | No seed, and no cell can hold 20 cases | 2026-10-14 |
| The `latest` pre-generated sample | Its contents change under one name, and it has no seed | 2026-10-14 |
| Removing condition rows only | Turns a case into a non-case; see Rationale | 2026-10-14 |
| Removing part of a patient's rows | Turns a case into a non-case; a fraction removes whole patients instead | 2026-10-14 |
| Synthea's `-a` age range | Samples ages uniformly instead of from the census, distorting frequencies | 2026-10-14 |

## Enforcement

- Owner: `src/bko/datasets/`
- Pinned by: `tests/test_datasets_synthea_run.py`, `tests/test_datasets_synthea.py`,
  `tests/test_datasets_ablation.py`

## Reversal Conditions

**The jar moves to another release if E2's configuration names one**; populations are regenerated
under a new directory key.

## Related

- [The UCI evaluation frame](uci-evaluation-frame.md)
- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
