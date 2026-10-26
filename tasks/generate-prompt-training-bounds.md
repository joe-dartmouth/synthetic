---
id: generate-prompt-training-bounds
title: Bound each numeric field by the training split
status: blocked
depends_on: evaluate-second-classifier
blocks: evaluate-e1-second-configuration
parallel_safe: true
owner: Shirley
milestone: final-report
paid: false
adrs: --
---

# Bound each numeric field by the training split

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

The user turn bounds each numeric field by `RANGES`' plausible range, age clipped to the cell's
band, while the drop rule in `bko.baselines.records` keeps only rows inside the training split's
minimum and maximum. In `e1-3` the `bko` arm dropped 5840 of the 31643 records it generated as
outside that range (its `validity` line in E1's Result,
[`docs/EVIDENCE.md`](../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation)), each drop costing a
top-up call, and of the records its call log decodes 7.2% are younger than 29 and 9.3% older than
77, where UCI runs 29 to 77, so the kept records spread wider in age than the training rows (a
standard deviation of 9 to 11 years per cell against 3 to 6). Those figures come from
`env -u ANTHROPIC_API_KEY uv run --no-sync python -I scratchpad/quality.py scratchpad/e1-3.jsonl`
on the runner's machine (gitignored, sha256
`aa21abfedf484d8f23677ddfe5805bf055906aaf40aef928670403a06ec834fb`). It is the cheapest likely
gain in fidelity, and it changes what every call sends, so it is a new configuration of E1 that
[`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md) runs, and it lands
after the units that replay `e1-3`'s log.

## Scope

**In.**

- `bko.generate.guidance.intervals` takes the training split and reads each numeric field's
  interval from its minimum and maximum there; age is that interval intersected with the cell's
  band, which the drop rule's whole-split range and its off-cell reason together enforce. The
  system prompt's statement of each field's plausible range moves with it, so a call carries one
  range per field.
- A test pinning the rendered system prompt and user turn on a `tmp_path` fixture split, and one
  pinning that every field's interval is the intersection the drop rule enforces.
- ADR [`research/record-generation-design`](../docs/adr/research/record-generation-design.md)
  records the decision with `e1-3`'s drop count as its evidence and drops its ruled-out row for the
  training split's minimum and maximum; [`docs/RESEARCH.md`](../docs/RESEARCH.md#6-datasets) and
  [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md#generation), which state what leaves the machine,
  say that the bounds are two training values per field.

**Out.** Any paid run; changing `RANGES`, which validation still reads; the drop rule.

## Acceptance

- The rendered prompts on the fixture split are pinned, and `uv run --no-sync python -m bko.generate
  per-subgroup --guidance ontology --log <new file>` with a fake `Send` renders bounds equal to the
  default training split's intersection per field.
- `make check`.
