---
status: accepted
topic: research
created: 2026-11-24
updated: 2026-11-24
---
# One serializer and one LoRA recipe fine-tune a language model per arm, scored on P(yes)

## Context

The arms' records are scored by tabular classifiers
([EVIDENCE.md](../../EVIDENCE.md#e6-diabetes-130-subgroup-removal)). Whether the differences
between generators survive in a fine-tuned language model is unknown. A comparison of the arms
through a language model is fair only if every arm shares one serializer, one base model and one
training budget. ADR [`tooling/slm-extra`](../tooling/slm-extra.md) holds the stack and the base
model, OLMo 3 7B, and measures one fine-tune over E6 holdout 0's full training split at 17 to 28
hours on this machine.

The nearest prior study is HARMONIC's LLM Efficacy
([`harmonic-2024`](../../literature.md#tabular-generators)), which fine-tunes a language model on
synthetic records and scores it on the real test set. This design differs in three ways: it
trains on the training split plus an arm's fill, where HARMONIC trains on synthetic records alone;
its budget is fixed in machine time, where HARMONIC's is 5 epochs; and its base model is OLMo 3 7B,
whose training data is published, where HARMONIC's is LLaMA-2-7b-chat.

**Determinism, 2026-11-24.** At commit `4f28c416ffa40d835f904e8683ee4114aae23bd0` (`dirty=false`),
on the Apple M5 Max on AC power under `caffeinate -i -s`, a scratch script called
`bko.slm.train.train` twice at `LORA` (seed 0) on MPS in BF16, each time on a fresh load of the
pinned base model, over the same first 400 rows of E6 holdout 0's `none` training split
(`runs/e6-25`, run `e6-thina`; 27 positive), 100 steps each. The two adapters' tensors were
`torch.equal` throughout (largest absolute difference 0.0), both `adapter_model.safetensors` files
had sha256 `c8085661aebd030009daa1a222412ae83d77435e63a467625ab67c91adc0b4b0`, both final losses
were 0.059293, and P(yes) over the holdout's first 200 test records differed by at most 0.0
(mean 0.0584). Training took 409.7 s and 434.4 s (1.024 and 1.086 s per row). The adapters were
deleted after the comparison.

## Decision Drivers

* **Must give every arm the same serializer, base model and recipe**, so an arm difference is the
  records' difference.
* **Must score a probability that `bko.evaluate.utility` reads**, so the language model's measures
  sit beside the tabular classifiers'.
* **Must keep a fitted adapter on this machine**, since it is fitted on patient-derived records.
* **Must fit the whole comparison in under one day of machine time.**

## Decision

1. **Serializer** (`bko.slm.serialize`). One prompt template serves training and scoring: one
   line per feature in the frame's feature order, the feature's description, a colon and its
   value (a coded value as its codebook meaning, a whole value without a decimal point, a
   fractional value at the frame's decimal places), then the frame's question. The answer is
   ` yes` or ` no`, one token each. A training text minus its answer is the scoring prompt.
2. **Loss.** The loss falls on the answer token alone, over records of both labels.
3. **Probability.** P(yes) is the softmax over the two answer tokens' logits at the last prompt
   position.
4. **Training set per arm.** The training split plus the arm's stored fill. **Controls:** the base
   model with no fine-tune, `none` (the training split alone) and `real` (the un-ablated split).
   The test split is scored whole, so a run whose experiment decontaminates is refused.
5. **LoRA settings** (`bko.slm.train.LORA`). Rank 16, alpha 32, dropout 0.05, every linear
   layer; AdamW at torch's defaults and learning rate 2e-4, linear warmup over 3% of the steps
   then linear decay; micro-batch 4; one epoch; seed 0.
6. **Adapter path.** `data/models/slm-adapters/<run-id>/<holdout>-<arm>/`, under the gitignored
   `data/`. An adapter never leaves the machine: one fitted on patient-derived records can
   reproduce training rows. The directory holds the adapter's configuration, its safetensors
   weights and `adapter_meta.json`, which names the run, folder, `rows` store sha256, holdout, arm
   and base revision; scoring refuses a directory holding any other file or a meta that differs
   from its arguments.
7. **Reproducibility.** A fine-tune is reproducible at one seed on this machine: two fits give
   bitwise-equal adapters (Context, Determinism), so one fine-tune per arm and seed stands for
   that arm, and a stored adapter's sha256 can be re-derived by fitting again.
8. **Budget.** A full-split fine-tune takes 17 to 28 hours (ADR `tooling/slm-extra`), so eight
   fine-tunes per holdout cost about 6 to 10 days. The comparison's budget is under one day of
   machine time. This record does not fix the comparison's holdouts, seeds, rows per fine-tune,
   base model size, power limit or share mixes; unit `slm-lora-day-budget` fixes them and revises
   this record.

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
- Pinned by: `tests/test_repo_gate.py`; `tests/test_slm_serialize.py` for the serializer's
  golden prompt and its training text; `tests/test_slm_train.py` for the answer-only loss, the
  two-logit P(yes) at each prompt's last token, the settings reaching the fit, and the adapter's
  safetensors file, and `test_the_comparison_lora_settings_are_pinned` for `LORA`'s values;
  `tests/test_slm_holdout.py` for each arm's training set and the refused decontaminating run;
  `tests/test_slm_cli.py` for the adapter directory's files and meta
- Code: `src/bko/slm/serialize.py`, `src/bko/slm/train.py`, `src/bko/slm/__main__.py`

## Reversal Conditions

The serializer stops giving ` yes` and ` no` one token each under the base model's tokenizer; or a
measured design fits a full-split fine-tune per arm in under one day.

## Related

- [The `slm` extra](../tooling/slm-extra.md)
- [Diabetes 130 subgroup removal](diabetes-subgroup-removal-design.md)
- [Classifier panel](classifier-panel.md)
