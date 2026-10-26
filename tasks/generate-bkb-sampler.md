---
id: generate-bkb-sampler
title: Generate records from the BKO without an LLM
status: ready
depends_on: --
blocks: generate-bkb-skeleton-fill
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
candidates, whose prompt names no gap, closed the same per-subgroup gaps as `ontology`-level ones.
An arm that samples records from the fused BKB alone isolates the knowledge: it is free, seeded,
reproducible, sends nothing off the machine, and closes per-subgroup gaps by construction, which is
itself a finding to report. `bko.bkb` holds structure, instantiation and the joint mass of a set of
S-nodes (`bko.bkb.inference.probability`), and no sampler.

## Scope

**In.**

- `bko.bkb.sample`: forward sampling of one complete set of I-node states from a BKB, each
  component's state drawn among the S-nodes whose antecedents the states drawn so far support, by
  their probabilities, in an order the structure fixes; seeded; refusing a BKB with a cycle it
  cannot order. A new ADR under `docs/adr/research/` fixes the algorithm, how a component with no
  supported S-node is read, and how the ontology and dataset fragments are weighted in the fused
  BKB the sampler reads.
- `bko.generate.direct`: for each requested cell, sample concept states from the BKB fused from the
  ontology fragments and the cell's and the population's dataset fragments (`bko.gaps.observe`),
  map each sampled state back to field values through `bko.ontology.mapping` (a codebook state
  gives the code; a guideline cutoff gives the side of the cutoff), and draw each numeric field
  inside its side from the training label group's empirical distribution, the cell's own records
  when the cell holds any. The output is a candidates file in `bko.generate.candidates`'s form, so
  `bko.validate`, the drop rule and `bko.fusion` read it unchanged.
- Per record, which fields the BKB decided and which the empirical fallback drew, reported as a
  count per field, since the mapping covers few fields.
- An arm `bko-direct` in `bko.evaluate.experiment.E1`, no criterion reading it, scored beside the
  other arms on every holdout from `e1-3`'s call log through `score --replay` (it makes no call,
  so the replay's call order is unchanged): validity, drops, replica fidelity and utility, test
  copies, and each criterion's measures with no verdict. If the replay refuses the new arm, the
  record states why and the arm is scored alone.
- Closure under C1's reading over the default split (`python -m bko.fusion` over the sampler's
  candidates file), reported beside `c1-2`'s.
- A new E1 Runs row and a Result addition in `docs/EVIDENCE.md`.

**Out.** Any API call; a verdict on the arm (a criterion reading it means changing
[`docs/RESEARCH.md`](../docs/RESEARCH.md#4-success-criteria) first); exact belief updating, which
[`docs/IDEAS.md`](../docs/IDEAS.md#exact-belief-updating-over-a-bkb) keeps.

## Acceptance

- `uv run --no-sync python -m bko.generate direct --request completeness` writes a candidates file
  identical on a second run with the same seed, and a test pins a hand-worked sample on a fixture
  BKB.
- The arm's measures regenerate from `score --replay` over `e1-3`'s call log, recorded in
  `docs/EVIDENCE.md`.
- `make check`.
