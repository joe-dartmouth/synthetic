---
id: evaluate-confirmatory-results
title: Run every confirmatory run at the tag and report every registered contrast
status: blocked
depends_on: evaluate-confirmatory-registration
blocks: evaluate-crc-acs-results
parallel_safe: true
milestone: direct-method-research
paid: true
adrs: --
---

# Run every confirmatory run at the tag and report every registered contrast

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The study's claim rests on these runs.

## Scope

**In.**

- Every confirmatory run unit
  [`evaluate-confirmatory-registration`](evaluate-confirmatory-registration.md) registers, at its
  tag.
- The `bko` ablation through `launch`, under its own ceiling, with the estimate in its Runs row
  first. These are the only paid calls.
- `rows`, `panel`, `privacy` and `contradictions --rows` (ADR
  [`research/data-contradiction-handling`](../docs/adr/research/data-contradiction-handling.md),
  Decision 2), the reading of unit
  [`evaluate-trace-attribution`](evaluate-trace-attribution.md), and `privacy`'s match attack and
  donor reuse (ADR [`research/membership-inference-reading`](../docs/adr/research/membership-inference-reading.md)).
- Every registered contrast reported, null or not.
- A deviations paragraph.

**Out.**

- Any contrast not registered, except one labelled exploratory.

## Acceptance

- Each [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry is `complete`, with its verdict and its
  runs' sha256s.
- `make check` passes.
