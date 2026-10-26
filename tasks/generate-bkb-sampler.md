---
id: generate-bkb-sampler
title: Generate records from the BKO without an LLM
status: blocked
depends_on: evaluate-named-runs
blocks: generate-bkb-skeleton-fill, generate-bkb-curation
parallel_safe: true
owner: Partiban
milestone: final-report
paid: false
adrs: research/bkb-direct-generation
---

# Generate records from the BKO without an LLM

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

Every BKO arm of E1 generates through the Messages API, so its result mixes what the ontology
contributes with what the model's prior contributes, and C1's record notes that `cell`-level
candidates, whose prompt names no gap, closed the same per-subgroup gaps as `ontology`-level ones
([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#c1-gap-closure-after-fusion), Limits). An arm that
draws records from the fused BKB alone isolates the knowledge: it is free, seeded, reproducible
and sends nothing off the machine.

What the fused BKB holds fixes the design. Every class component has one state, the individual
(`bko.ontology.fragments.INDIVIDUAL`; ADR
[`research/bko-fragment-encoding`](../docs/adr/research/bko-fragment-encoding.md) rules out
member and non-member states), each dataset fragment is one prior per concept its records show, at
that concept's share (ADR
[`research/dataset-fragment-encoding`](../docs/adr/research/dataset-fragment-encoding.md)), and
every `is_a` S-node has probability 1. There is no absent state to draw, so a sampler cannot walk
the BKB choosing states; what it can draw is each concept's presence at its fused share, closed
upward under `is_a`.

## Scope

**In.**

- `bko.generate.direct`: for each requested cell, build the BKB fused from the ontology fragments
  and the cell's and the population's dataset fragments (`bko.gaps.observe`), read each mapped
  concept's fused share from its S-nodes, draw the concept present with that probability, close
  the drawn set upward under the ontology's `is_a` statements, and map the result to field values
  through `bko.ontology.mapping`: a codebook row gives its code, a cutoff row the side of its
  cutoff, and each numeric field is drawn inside its side from the training label group's
  empirical distribution, the cell's own records when the cell holds any. Draws come from a seeded
  numpy generator held by `bko.generate`, so `bko.bkb` stays standard-library only (ADR
  [`research/bkb-layer-source`](../docs/adr/research/bkb-layer-source.md)) and the sampler meets
  the repository's ruff rules.
- A new ADR under `docs/adr/research/` fixes the fused BKB's weighting of the cell's fragment
  against the population's (which decides whether a concept the cell never shows can be drawn,
  and so whether the arm can close a per-subgroup gap), how a row that names several codes or
  several fields is read back, how a label-only row is read in a cell of the other label, what a
  concept with no mapping row contributes, and the numeric fallback when the cell holds no record
  on the drawn side of a cutoff. The mapping covers every feature but `thalach`, so the ADR also
  states which fields the draw decides and which the fallback draws, reported per field.
- The output is a candidates file; `bko.generate.candidates`'s header gains a generator field so
  the file names the sampler rather than a guidance level, and `bko.validate`, the drop rule and
  `bko.fusion` read it unchanged. A `direct` subcommand of `python -m bko.generate` with `--seed`
  and `--log` writes it.
- A named run (`evaluate-named-runs`) with the baselines and an arm `bko-direct`, free, on E1's
  holdouts: validity, drops, replica fidelity and utility, test copies and each criterion's
  measures with no verdict, read beside the judged run's `bko` at the same holdouts.
- Closure under C1's reading over the default split (`python -m bko.fusion` over the sampler's
  candidates file), reported beside `c1-2`'s.
- A new `docs/EVIDENCE.md` entry, and [`docs/RESEARCH.md`](../docs/RESEARCH.md#generate-and-validate),
  which says records are generated through the Messages API, gains the arm.

**Out.** Any API call; a verdict on the arm (a criterion reading it means changing
[`docs/RESEARCH.md`](../docs/RESEARCH.md#4-success-criteria) first); exact belief updating, which
[`docs/IDEAS.md`](../docs/IDEAS.md#exact-belief-updating-over-a-bkb) keeps; a change to the
fragment encoding.

## Acceptance

- `uv run --no-sync python -m bko.generate direct --request completeness --seed 0 --log <new file>`
  writes a candidates file identical on a second run, and a test pins the draw on a fixture BKB
  whose shares are 0 and 1, so the expected record is known by hand.
- The arm's measures regenerate from `score <run>` and `compare <run>`, recorded in
  `docs/EVIDENCE.md`.
- `make check`.
