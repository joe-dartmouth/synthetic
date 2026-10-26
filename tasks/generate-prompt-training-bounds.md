---
id: generate-prompt-training-bounds
title: Bound each numeric field by the training split
status: ready
depends_on: --
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

`cleanup:generate/prompt-range-wider-than-training`: the user turn bounds each numeric field by
`RANGES`' plausible range, age clipped to the cell's band, while the drop rule in
`bko.baselines.records` keeps only rows inside the training split's minimum and maximum. In `e1-3`
the `bko` arm generated 31643 records and dropped 5840 of them as outside that range, every drop
costing a top-up call, and the kept records spread wider in age than the training rows. It is the
cheapest likely gain in fidelity, and it changes what every call sends, so it is a new
configuration of E1 that [`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md)
runs.

## Scope

**In.**

- `bko.generate.guidance` reads each numeric field's interval from the training split's minimum
  and maximum, age also clipped to the cell's band, and `prompt.render` states those bounds. The
  bounds the prompt states are the bounds the drop rule enforces, and nothing else about the prompt
  changes.
- A test pinning the rendered user turn on a fixture split, and one pinning that the interval of
  every numeric field equals the drop rule's.
- ADR [`research/record-generation-design`](../docs/adr/research/record-generation-design.md)
  records the decision in place of its reason for `RANGES`, with `e1-3`'s drop count as the
  evidence, and ADR
  [`research/record-validation-design`](../docs/adr/research/record-validation-design.md) is
  checked for a range statement that moves with it.
- The cleanup entry is deleted in the commit that lands this.

**Out.** Any paid run; changing `RANGES`, which validation still reads; the drop rule.

## Acceptance

- `python -m bko.generate per-subgroup --guidance ontology` with a fake `Send` renders bounds equal
  to the default training split's minimum and maximum per field, pinned by a test.
- No cited `cleanup:generate/prompt-range-wider-than-training` remains.
- `make check`.
