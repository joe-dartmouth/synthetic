---
id: privacy-donor-attacks-results
title: Read the match attack and donor reuse over the E6 and E10 stores
status: ready
depends_on: --
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Read the match attack and donor reuse over the E6 and E10 stores

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

`privacy` reads a field-group match attack and each direct or perturbation fill's donor reuse
(ADR
[`research/membership-inference-reading`](../docs/adr/research/membership-inference-reading.md)).
Each reading over a run's stores takes hours, and a reviewed code PR is not held for them, so
this unit runs and records them.

## Scope

**In.**

- `privacy` over the E6 stores (`e6-thin`, `e6-empty`, `e6-curated`) and the E10 stores
  (`e10-pram-thin`, `e10-pram-empty`, and the pilots `e10-pilot-thin` and `e10-pilot-empty`,
  which hold the rewrite arm), each from a frozen worktree.
- Each reading recorded as a Runs row of the
  [E8](../docs/EVIDENCE.md#e8-membership-inference-nndr-and-dcr-on-e6) and
  [E10](../docs/EVIDENCE.md#e10-identifier-perturbation) entries.
- One sentence in E8's Configuration naming the match attack's units (the field groups plus a
  singleton per other column) and donor reuse.

**Out.**

- Any change to the attack, the donor measure or an arm.

## Acceptance

- The seven Runs rows exist, each with the sha256 of its `privacy.csv` and `privacy.log`.
- `make check` passes.
