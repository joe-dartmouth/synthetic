---
status: accepted
topic: research
created: 2026-10-31
updated: 2026-10-31
---
# The BKB likelihood of a candidate, and the free gate that decides whether it curates

## Context

The BKO guides generation only through the prompt. CLLM curates generated rows by a classifier's
confidence (ADR [`research/cllm-arm-design`](cllm-arm-design.md)); the fused BKB could curate by a
knowledge likelihood instead, keeping the records the BKB finds most probable. The joint mass of the
S-nodes an instantiation reaches is not that likelihood: every dataset prior is reached from any
anchor and every `is_a` S-node has probability 1, so the mass is the same for every record of a
cell. With one state per class (ADR [`research/bkb-direct-generation`](bkb-direct-generation.md)),
the score the fused BKB can give a record is the probability that its draw makes the record's mapped
concepts, present and absent, at the cell's fused shares.

A curated arm costs a multiple of one arm's spend, since it asks for more records than it keeps.
Whether the score separates records that match the real cell from records that do not can be
measured first, for free, on the candidates the judged run already generated: E1's `e1-4` keeps
every `bko` candidate per holdout in its `holdout-candidates.jsonl`
([EVIDENCE.md](../../EVIDENCE.md#e1-uci-heart-disease-augmentation)).

## Decision Drivers

* **The score must come from the fused BKB the sampler uses**, so a curated arm and `bko-direct`
  read the same knowledge.
* **The gate is fixed before it runs**, with its failure reading, so its result cannot shape it.
* **Both halves of the split carry the training cell mix**, so the contrast reads the score and not
  which cells it favours.
* **The free measurement reads the judged run's own file**, not a rerun or another run.

## Decision

We will score each candidate by **its log-probability under the direct sampler's concept draw**,
and run a curated arm only if **a free per-cell median split by that score separates fidelity on
all three measures**.

- **Score.** At the cell's shares (`bko.gaps.shares.shares`, the cell's and its label group's
  dataset-fragment priors at `GROUP_WEIGHT`), each mapped concept that is not label-only adds the
  log of its share when the record shows it and of one less its share when not. A concept with two
  field rows is present when either row applies. The outer concept of a nested pair is drawn given
  its inner: certain when the inner is present or the inner's share is 1, else at
  `(s_o - s_i) / (1 - s_i)`. Label-only and unmapped concepts add 0. A record the draw cannot make
  scores minus infinity, and the score keeps it.
- **Pool.** Per holdout, the `bko` arm's replica records whose outcome is `kept` or
  `dropped-quota`: every record the arm generated that passed validation.
- **Split.** Each cell's pool sorted by score, highest first, ties ordered by
  `sha256(f"{holdout}:{call}:{position}")`; the first `n // 2` are its top half and the last
  `n // 2` its bottom half. Per-cell tops are pooled against per-cell bottoms.
- **Gate.** Quality, column shapes and column pair trends (`bko.evaluate.fidelity`) of the pooled
  tops and of the pooled bottoms against the holdout's training rows, paired top minus bottom over
  E1's 100 holdouts, each with the corrected interval of `bko.evaluate.criteria` at `E1.confidence`
  (family 3) and the largest held-out ratio, as the criteria take it. The gate is separated when all three intervals lie above 0. A failure reads as "the
  score does not separate fidelity at the median", and a curated arm does not run.
- **Diagnostic.** Each concept's share in the top half, the bottom half and the training rows,
  logged beside the gate.
- **Home.** The shares and the score live in `bko.gaps` (`shares.py`, `likelihood.py`), which owns
  the fused fragments; the command is `python -m bko.evaluate likelihood`, because it reads
  `holdout_candidates`, `split`, `prepared` and `fidelity`, and `bko.generate` already imports
  `bko.gaps`.
- **Input.** `runs/e1-4/holdout-candidates.jsonl` as written: `holdout_candidates.read` accepts
  schema 1 beside schema 2, checking a schema-1 file's `uci` digest against UCI Heart Disease's.

## Rationale

The score is the sampler's own draw probability, so it ranks records by how readily `bko-direct`
would make them; a score from any other model of the cell would curate by knowledge the BKB does not
hold. Splitting each cell at its own median keeps the cell mix equal in both halves: a whole-pool
ranking puts low-share cells' records low, and the two halves would then differ in which cells they
hold, which fidelity reads as a difference. Requiring all three measures keeps the gate from passing
on marginals alone; column pair trends is the measure a concept-level score is least likely to move,
and an arm that keeps marginal shapes but breaks pairs is not one worth paying for.

## Consequences

**Gained**: a free, reproducible answer to whether BKB curation is worth a paid arm; a score over
any candidates file in the schema-1 or schema-2 format; the shares in the stage that owns the fused
fragments, read by the sampler and the score alike.
**Accepted**: the score sees only mapped concepts, so two records with the same concepts score the
same whatever their numeric fields; the gate reads one generator's candidates, at `e1-4`'s level and
effort; a record at minus infinity sorts last and stays in its cell's bottom half.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| The joint mass of the reached S-nodes as the score | the same for every record of a cell, as Context says | 2026-10-31 |
| The score in `bko.generate` | makes the generation stage a scorer; the fused fragments are `bko.gaps`'s | 2026-10-31 |
| Ranking the whole pool rather than each cell | the halves would differ in cell mix, and fidelity would read that | 2026-10-31 |
| Excluding holdouts with tied scores at the median | shrinks the paired sample; the sha256 order breaks ties without favouring a call | 2026-10-31 |
| Breaking ties by `(call, position)` | favours early calls | 2026-10-31 |
| Replaying `e1-4` to a schema-2 file first | the gate would read a rewritten file, not the judged run's own | 2026-10-31 |
| `g5-1`'s candidates as the input | not the judged run | 2026-10-31 |

## Enforcement
- Owner: `src/bko/gaps/likelihood.py`, `src/bko/evaluate/likelihood.py`
- Pinned by: `tests/test_gaps_likelihood.py`, `tests/test_evaluate_likelihood.py`

## Reversal Conditions
A fused BKB with more than one state per class, or field-level knowledge in the fragments, gives a
record's numeric fields a likelihood; the score is then revisited. A generator whose candidates
separate on all three measures under this gate reopens the curated arm.

## Related
- [Direct generation draws each mapped concept at its fused share](bkb-direct-generation.md)
- [The CLLM arms and their curator](cllm-arm-design.md)
- [E1's evaluation protocol](e1-evaluation-protocol.md)
