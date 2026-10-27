---
id: generate-bkb-curation
title: Curate LLM candidates by their likelihood under the BKO
status: blocked
depends_on: evaluate-e1-second-configuration
blocks: --
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: research/bkb-candidate-curation
---

# Curate LLM candidates by their likelihood under the BKO

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

The BKO guides generation only through the prompt. CLLM
([`docs/IDEAS.md`](../docs/IDEAS.md#cllm-and-great-as-alternative-generators)) curates generated
rows by a classifier's confidence; the fused BKB offers a knowledge likelihood instead. The joint
mass of the S-nodes instantiation reaches is not that likelihood: every dataset prior is reached
from any anchor and every `is_a` S-node has probability 1, so the mass is the same for every
record of a cell. With one state per class (the encoding
ADR `research/bkb-direct-generation` describes), the score the fused BKB can give a
record is the likelihood of its mapped concepts' presence and absence under their fused shares,
and whether that score separates records that match the real cell's distribution from records
that do not can be measured before any paid arm is built on it.

## Scope

**In.**

- `bko.gaps`, which owns the fused fragments, gains a score per candidate: the log-likelihood of
  the record's mapped concepts, present and absent, under each concept's fused share, with the
  shares and their smoothing read from the fused BKB the sampler's ADR fixes. A new ADR under
  `docs/adr/research/` fixes the score, what a concept with no mapping row contributes, and the
  tie-break among equal scores.
- Phase 1, free: one committed command scores every candidate in the per-holdout candidates file
  the judged run writes (the `holdout-candidates.jsonl` role), and reports per cell the score's
  spread, the share of records at the minimum, and the replica fidelity (quality, column
  shapes, column pair trends against the training rows) of the top half by score against the
  bottom half, paired over the holdouts.
- Phase 2, paid, only if phase 1's top half is closer to the training rows than its bottom half on
  each of the three measures over the paired holdouts: a named run with the baselines and an arm
  `bko-curated` that asks each cell for a multiple of its quota at `SETTINGS` and keeps the quota's
  highest-scoring records. Spend estimate: the multiple times one arm's share of the judged run's
  spend, stated in the Runs row before the run, under the named run's ceiling and the paid-run protocol
  (`docs/DEVELOPMENT.md`, Evaluation). If phase 1 shows no separation, the record says so and phase 2 does
  not run.
- A new `docs/EVIDENCE.md` entry holding phase 1's measures and, when it runs, phase 2's run with
  its spend; [`docs/IDEAS.md`](../docs/IDEAS.md#cllm-and-great-as-alternative-generators) and
  [`docs/RESEARCH.md`](../docs/RESEARCH.md#generate-and-validate) record that curation joined the
  Validate stage.

**Out.** A verdict on the arm; changing what `bko` sends; a change to the fragment encoding.

## Acceptance

- Phase 1's measures come from one committed command over the judged run's folder, no API call,
  and a test pins the score on a fixture BKB.
- Phase 2 is recorded with its spend, or the entry states the phase 1 measure that stopped it.
- `make check`.
