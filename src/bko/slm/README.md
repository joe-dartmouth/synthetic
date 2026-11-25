# `bko.slm`: LoRA language-model classifier

Fine-tunes a LoRA adapter on a pinned causal language model as a classifier of the records each
arm produces, and scores test records as a probability that feeds `bko.evaluate.utility`. One
serializer turns a record into the prompt that training and scoring share. ADR
[`research/slm-comparison-design`](../../../docs/adr/research/slm-comparison-design.md) holds the
design and ADR [`tooling/slm-extra`](../../../docs/adr/tooling/slm-extra.md) the stack and the base
model.

- **Dependencies:** The `slm` extra (`pyproject.toml`), and `bko.datasets` for the frame.
- **Status:** built.
