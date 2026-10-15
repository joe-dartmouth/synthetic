---
status: accepted
topic: research
created: 2026-10-15
updated: 2026-10-15
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

* **Must give a known set of removed cases** for both the BKO detector and the count-based one.
* **Must record an identity that recreates the population** (`docs/EVIDENCE.md`).
* **Must leave no trace of a removed case** in the loaded tables.

## Decision

- **Generate** with the Synthea v4.0.0 jar whose size and sha256 `bko.datasets.sources` holds, on
  JDK 17 or newer, passing `-s`, `-cs` and `-p`, `-r` and `-e` both equal to the snapshot date (a
  past date), the state, CSV export of `patients.csv` and `conditions.csv` only,
  `exporter.years_of_history=0`, `generate.thread_pool_size=1`, and `TZ=UTC` with
  `-Duser.timezone=UTC`.
- **The population is the patients alive on the snapshot date.**
- **A case is a patient**: the ablation removes every patient who has one of the spec's condition
  codes by the snapshot date, with all their condition rows. The spec narrows the cases to one age
  band, one sex, both, or neither; neither removes the condition from the whole population.
- **Removal is complete** within the spec.
- **Ground truth** is the removed patient ids and the cell counts before and after.

`bko.datasets.synthea_run`, `bko.datasets.synthea` and `bko.datasets.ablation` hold the values. The
command's default spec is a demonstration cell sized from a pilot run; E2's configuration fixes its
own target cells and whether they are cells or whole conditions.

## Rationale

Generating lets the population size put a target cell above 20 before ablation, so the count-based
detector's finding is the ablation and not natural sparsity, and the pinned parameters are the
identity. Removing condition rows alone would keep the patient as a non-case, which inflates the
cell of patients without the condition. A BKO statement is grounded by one remaining record, so
partial removal is invisible to the BKO detector. An ontology that holds no age- or sex-specific
statement may see only a whole condition's absence, which is why the spec can drop the band and
the sex.

## Consequences

**Gained**: a population recreated from its parameters; a ground truth at the cell level, where
both detectors report.
**Accepted**: a JDK on every machine that generates; a 201 MB jar download; one thread, so
generation is slower; cross-platform identity is not promised, so each run records its CSV
checksums.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The apr2020 pre-generated sample | No seed, and no cell can hold 20 cases | 2026-10-14 |
| The `latest` pre-generated sample | Its contents change under one name, and it has no seed | 2026-10-14 |
| Removing condition rows only | Turns a case into a non-case; see Rationale | 2026-10-14 |
| Partial removal | Invisible to the BKO detector; E2 can add it for the count-based arm | 2026-10-14 |
| Synthea's `-a` age range | Samples ages uniformly instead of from the census, distorting frequencies | 2026-10-14 |

## Enforcement

- Owner: `src/bko/datasets/`
- Pinned by: `tests/test_datasets_synthea_run.py`, `tests/test_datasets_synthea.py`,
  `tests/test_datasets_ablation.py`

## Reversal Conditions

**Partial removal is added if E2's configuration in `docs/EVIDENCE.md` asks for a count-based arm
with a fraction removed**; this record is then edited.

**The jar moves to another release if E2's configuration names one**; populations are regenerated
under a new directory key.

## Related

- [The UCI evaluation frame](uci-evaluation-frame.md)
- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
