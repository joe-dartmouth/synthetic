---
id: evaluate-classical-baselines
title: Classical statistical baselines at small n
status: blocked
depends_on: evaluate-synthea-downstream
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

It depends on `evaluate-synthea-downstream` for two reasons. Its E3 reading pairs with E3's
recorded scores, which that unit writes. And E3's arms are `METHOD_ARMS` plus the BKO arms and
`real` (`bko.evaluate.experiment.E3`), so a method that joined `METHOD_ARMS` before E3's paid run
would change that run's configuration; landing after it keeps E3 as its ADR fixes it.

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
- E1, E3 and every run built from them keep their arms and comparators: the new methods join the
  harness's arms (`bko.evaluate.run.ARMS`) without entering `METHOD_ARMS`, which those runs read.
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
