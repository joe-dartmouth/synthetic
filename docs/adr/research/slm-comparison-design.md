---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# One serializer and one LoRA recipe fine-tune a language model per arm, scored on P(yes)

## Context

The arms' records are scored by tabular classifiers ([EVIDENCE.md](../../EVIDENCE.md#e6-diabetes-130-subgroup-removal)).
Whether the differences between generators survive in a fine-tuned language model is unknown. A
comparison of the arms through a language model is fair only if every arm shares one serializer,
one base model and one training budget. ADR [`tooling/slm-extra`](../tooling/slm-extra.md) holds
the stack and the base model, OLMo 3 7B, and measures one fine-tune over E6 holdout 0's full
training split at 17 to 28 hours on this machine.

The nearest prior study is HARMONIC's LLM Efficacy
([`harmonic-2024`](../../literature.md#tabular-generators)), which fine-tunes a language model on
synthetic records and scores it on the real test set. This design differs in three ways: it
trains on the training split plus an arm's fill, where HARMONIC trains on synthetic records alone;
its budget is fixed in machine time, where HARMONIC's is 5 epochs; and its base model is OLMo 3 7B,
whose training data is published, where HARMONIC's is LLaMA-2-7b-chat.

## Decision Drivers

* **Must give every arm the same serializer, base model and recipe**, so an arm difference is the
  records' difference.
* **Must score a probability that `bko.evaluate.utility` reads**, so the language model's measures
  sit beside the tabular classifiers'.
* **Must keep a fitted adapter on this machine**, since it is fitted on patient-derived records.
* **Must fit the whole comparison in under one day of machine time.**

## Decision

1. **Serializer.** One prompt template serves training and scoring: one line per feature in the
   frame's feature order, the feature's description, a colon and its value (a coded value as its
   codebook meaning, a whole value without a decimal point, a fractional value at the frame's
   decimal places), then the frame's question. The answer is ` yes` or ` no`, one token each. A
   training text minus its answer is the scoring prompt.
2. **Loss.** The loss falls on the answer token alone, over records of both labels.
3. **Probability.** P(yes) is the softmax over the two answer tokens' logits at the last prompt
   position.
4. **Training set per arm.** The training split plus the arm's stored fill. **Controls:** the base
   model with no fine-tune, `none` (the training split alone) and `real` (the un-ablated split).
5. **LoRA settings.** Rank 16, alpha 32, dropout 0.05, every linear layer; AdamW at torch's
   defaults and learning rate 2e-4, linear warmup over 3% of the steps then linear decay;
   micro-batch 4; one epoch; seed 0.
6. **Adapter path.** `data/models/slm-adapters/<run-id>/<holdout>-<arm>/`, under the gitignored
   `data/`. An adapter never leaves the machine: one fitted on patient-derived records can
   reproduce training rows.
7. **Budget.** A full-split fine-tune takes 17 to 28 hours (ADR `tooling/slm-extra`), so eight
   fine-tunes per holdout cost about 6 to 10 days; the team does not run that. The whole
   comparison fits in under one day of machine time. The holdouts, seeds, rows per fine-tune,
   base model size, power limit and share mixes are fixed by unit `slm-lora-day-budget`, which
   revises this record.

## Rationale

One template for both passes means the scoring prompt is exactly the prefix the model was trained
to complete, so the answer position is the same in both. Loss on the answer alone trains the
classifier and not a model of the records' text. Reading two logits rather than the whole
vocabulary gives a probability that sums to one over the two answers, so it feeds the cutoff and
the Brier score as a tabular classifier's does. Adding the fill to the training split matches how
the arms are scored by the tabular classifiers, so the two readings compare.

## Consequences

**Gained**: a language-model reading of each arm on the same footing as the tabular classifiers;
an adapter path that keeps fitted weights out of the repository and off the network.
**Accepted**: the full-split design does not fit the budget, so the comparison waits for unit
`slm-lora-day-budget` to choose a smaller one; P(yes) ignores probability mass on other tokens.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The full-split design, eight fine-tunes per holdout | 6 to 10 days of machine time per holdout | 2026-11-24 |
| Placing the trainer in `bko.evaluate` | `bko.evaluate` imports the `eval` extra only; the trainer brings its own extra and stage | 2026-11-24 |
| P(yes) over the full vocabulary | Mass on tokens other than the two answers would leave the two probabilities summing below one | 2026-11-24 |
| A `Source` pin per weight file | A base model is a set of files at one revision; one pin of the set keeps the revision and the files together | 2026-11-24 |

## Enforcement

- Owner: `docs/adr/tooling/slm-extra.md`
- Pinned by: `tests/test_repo_gate.py`

## Reversal Conditions

The serializer stops giving ` yes` and ` no` one token each under the base model's tokenizer; or a
measured design fits a full-split fine-tune per arm in under one day.

## Related

- [The `slm` extra](../tooling/slm-extra.md)
- [Diabetes 130 subgroup removal](diabetes-subgroup-removal-design.md)
- [Classifier panel](classifier-panel.md)
