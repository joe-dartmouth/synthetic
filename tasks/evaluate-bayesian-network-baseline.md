---
id: evaluate-bayesian-network-baseline
title: A learned Bayesian network against the expert structure
status: blocked
depends_on: evaluate-synthea-downstream
blocks: --
parallel_safe: true
owner: Joe
milestone: final-report
paid: false
adrs: --
---

# A learned Bayesian network against the expert structure

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

The BKO's claim is that medical knowledge adds what the sparse data cannot. No arm yet separates
the knowledge from the probabilistic sampling it rides on. A Bayesian network whose structure and
parameters are learned from the same training split, PrivBayes's method
([`zhang-2014`](../docs/literature.md#tabular-generators)) without its privacy noise, samples
from a structure the data chose. `bko-direct` (ADR
[`research/bkb-direct-generation`](../docs/adr/research/bkb-direct-generation.md)) samples
from the fused BKB, whose structure the ontologies give, also with no API call. Read against each
other, the two isolate the expert structure; read against `bko`, the learned network isolates
knowledge and the LLM together.

It depends on `evaluate-synthea-downstream` for the reasons `evaluate-classical-baselines` gives:
its E3 reading pairs with E3's recorded scores, and a method that joined `METHOD_ARMS` before
E3's paid run would change that run's arms.

## Scope

**In.**

- The learner's source, decided first and recorded in a new ADR under `docs/adr/research/`. The
  question: whether a library learner resolves, installs and imports on Python 3.14 under the lock
  rules (`AGENTS.md`, Gotchas; ADR [`tooling/uv-ruff-ty`](../docs/adr/tooling/uv-ruff-ty.md) says
  when a new extra joins `make setup`), or whether the project writes its own. pgmpy resolves
  under a scratch copy of the lock (checked 2026-10-28); its install, its import and the learners
  the unit needs are unchecked on Python 3.14. `bko.bkb` holds no structure or parameter learner,
  so an in-house learner is new code beside it.
- The same ADR fixes the structure score and search, the parameter prior, how measured predictors
  are discretized for the network and returned to the frame's precision, how a stratum's quota is
  met (sampling with the stratum's columns as evidence, or unconditioned draws kept by their own
  values), and the seeding, each with its reason.
- A method under `bko.baselines` that fills a request exactly, stratum by stratum, on the training
  split only, seeded from the run's seed. E1, E3 and every run built from them keep their arms and
  comparators: it joins the harness's arms (`bko.evaluate.run.ARMS`) alone, through the seam
  `evaluate-classical-baselines` builds (`METHOD_ARMS` and `ADDS_ROWS` pinned to explicit tuples of
  today's six methods, with a test pinning E1's and E3's arms). If this unit lands first, it builds
  that seam itself.
- Free named runs (`bko.evaluate.experiment.RUNS`), with no criterion and no BKO arm, the network
  as the candidate: on E1's holdouts over UCI, read by `compare --beside` E1's judged run's scores
  with `--beside-arm bko` and G2's scores with `--beside-arm bko-direct`; on E3's holdouts,
  removal and copy rule over the Synthea frame, read beside E3's merged scores with `--beside-arm
  bko`, with `real` and, where the direct arm runs on the Synthea frame, `bko-direct` in the same
  run. If it does not, the entry says why.
- A new `docs/EVIDENCE.md` entry: the learned structure per dataset at one seed, and per dataset
  validity and drops, replica fidelity and utility, each criterion's measures and each `beside`
  interval, with no verdict.

**Out.** Privacy noise (PrivBayes's differential privacy); adding the network to E1's comparison
set (that changes [`docs/RESEARCH.md`](../docs/RESEARCH.md#4-success-criteria) section 4 first);
changing the fused BKB or the direct arm.

## Acceptance

- The ADR records the learner's source with the resolve, install and import evidence behind it.
- A `tmp_path` test learns a network from a fixture, fills a fixture request exactly, and its
  output is equal across two runs at one seed.
- `E1`'s and `E3`'s arms and comparators are unchanged, pinned by a test.
- The named runs are recorded in the new entry with their scores' sha256s, and each `beside`
  reading names its paired and excluded holdouts.
- `make check`.
