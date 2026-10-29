---
id: evaluate-classical-baselines
title: Classical statistical baselines at small n
status: ready
depends_on: --
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: --
---

# Classical statistical baselines at small n

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

E1's comparison methods are two controls, two oversamplers and two deep generators (ADR
[`research/baseline-augmentation-design`](../docs/adr/research/baseline-augmentation-design.md)).
SDV's defaults give CTGAN and TVAE few optimizer steps on about 240 training records, so a weak
deep baseline can flatter `bko`. The statistical family the brief lists
([`docs/BACKGROUND.md`](../docs/BACKGROUND.md#tabular-generator-families)) holds the methods
built for small samples: a Gaussian copula, the synthesizer of the Synthetic Data Vault
([`patki-2016`](../docs/literature.md#tabular-generators)), and sequential CART synthesis as
synthpop runs it ([`nowok-2016`](../docs/literature.md#tabular-generators)). Neither calls an API,
so both can be read beside the judged runs for free.

Its E3 reading pairs with E3's recorded scores
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e3-synthea-downstream-augmentation)). E3's arms are
`METHOD_ARMS` plus the BKO arms and `real` (`bko.evaluate.experiment.E3`), so a method that joins
`METHOD_ARMS` would change E3's configuration; the seam below keeps E3 as its ADR fixes it.

## Scope

**In.**

- Two methods under `bko.baselines`, each filling a request exactly, stratum by stratum, on the
  training split only, as every method does: SDV's `GaussianCopulaSynthesizer` (already in the
  `synth` extra) and sequential CART synthesis built on scikit-learn (already in the `eval`
  extra), each seeded from the run's seed. ADR `research/baseline-augmentation-design` gains each
  method's parameters, how it meets a stratum's quota (unconditioned draws kept by their own
  values, as CTGAN's, or the stratum's columns fixed and the rest synthesized), its seeding, and
  their reasons.
- SDV, its `copulas` dependency and the `ctgan` package carry the BUSL-1.1 licence (each installed
  distribution's metadata), so it covers CTGAN and TVAE as well as the copula; the ADR states it.
- E1, E3 and every run built from them keep their arms and comparators. No seam allows that yet:
  `METHOD_ARMS` and `ADDS_ROWS` (`bko.evaluate.experiment`) derive from every `Method` value, as
  `bko.evaluate.run.ARMS` does through `bko.baselines.methods.METHODS`, so a new `Method` value
  joins E1's and E3's arms. The unit builds the seam first: `METHOD_ARMS` and `ADDS_ROWS` pinned to
  explicit tuples of today's six methods, with a test pinning E1's and E3's arms, comparators and
  replica arms; the new methods then join `ARMS` alone.
- Two free named runs (`bko.evaluate.experiment.RUNS`), with no criterion and no BKO arm, so
  `score` takes no `--log` or `--candidates` and spends nothing: one on E1's holdouts over UCI,
  one on E3's holdouts, removal and copy rule over the Synthea frame. Each holds `METHOD_ARMS`,
  the two new methods and, on Synthea, `real`, with one new method as the candidate.
- Each run read by `compare <run> <scores> --candidate <method> --beside <judged scores>
  --beside-arm bko`, once per new method: on UCI beside E1's judged run's scores, on Synthea
  beside E3's merged scores. `--beside` pairs a holdout only when both files hold the same
  held-out rows and flagged cells (`docs/DEVELOPMENT.md`, Evaluation), so no judged run is
  re-spent.
- A new `docs/EVIDENCE.md` entry: per method and dataset, validity and drops, replica fidelity and
  utility, each criterion's measures and each `beside` interval, with no verdict; on Synthea,
  recall on the removed cell's held-out cases against `real`.

**Out.** Adding either method to E1's comparison set or judging `bko` against it (that changes
[`docs/RESEARCH.md`](../docs/RESEARCH.md#4-success-criteria) section 4 first); tuning either
method inside the stage; the synthpop R package itself.

## Acceptance

- Each method fills a fixture request exactly, and its output is equal across two runs at one
  seed, in `tmp_path` tests.
- `E1`'s and `E3`'s arms and comparators are unchanged, pinned by a test.
- Both named runs are recorded in the new entry with their scores' sha256s, and each `beside`
  reading names its paired and excluded holdouts.
- `make check`.
