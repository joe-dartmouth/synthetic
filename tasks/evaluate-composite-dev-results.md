---
id: evaluate-composite-dev-results
title: Run bko-direct-2 on the development frames and name the confirmatory candidate
status: blocked
depends_on: evaluate-composite-arm
blocks: evaluate-confirmatory-registration
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run bko-direct-2 on the development frames and name the confirmatory candidate

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

Variants can interact. A CART leaf donor, for one, is not under the donor cap. The composite's
own runs say whether the composition holds.

## Scope

**In.**

- The composite's development runs registered by unit
  [`evaluate-composite-arm`](evaluate-composite-arm.md), with every reading of unit
  [`evaluate-variant-dev-results`](evaluate-variant-dev-results.md).
- The [`docs/EVIDENCE.md`](../docs/EVIDENCE.md) entry's Result.

**Out.**

- Changing the composition after the run. A composite that fails the non-inferiority margin is
  reported, and unit
  [`evaluate-confirmatory-registration`](evaluate-confirmatory-registration.md) registers
  `bko-direct` as candidate with `bko-direct-2` as an arm.

## Acceptance

- The entry is `complete` and names the confirmatory candidate.
- `make check` passes.
