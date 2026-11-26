---
id: evaluate-crc-acs-results
title: Run the CRC ACS excerpts through SDNist and read them beside the archive's methods
status: blocked
depends_on: evaluate-crc-acs-benchmark, evaluate-confirmatory-results
blocks: --
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: --
---

# Run the CRC ACS excerpts through SDNist and read them beside the archive's methods

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

The CRC ACS reading runs only after the confirmatory ACS run is recorded
([`docs/RESEARCH.md`](../docs/RESEARCH.md#published-comparisons)), so this unit waits for unit
[`evaluate-confirmatory-results`](evaluate-confirmatory-results.md) as well as for the code unit
[`evaluate-crc-acs-benchmark`](evaluate-crc-acs-benchmark.md).

## Scope

**In.**

- Run the named runs, write one `bko-direct` file per excerpt, and score each with SDNist in the
  environment the code unit builds.
- The EVIDENCE entry's Runs, Result and Read against, beside the archive's published methods.

**Out.**

- Any change to the frame, the environment or a reading.
- Differential-privacy claims.

**Findings from planning.**

- It re-scores a fixed list of the archive's submissions under the code unit's own SDNist pin,
  rather than reading the archive's published scores from another SDNist version.
- It pins the archive by its release tag and a sha256 measured at first fetch.

## Acceptance

- The entry's status is `complete`.
- The entry's Read against compares with the archive's published methods' SDNist scores.
- Each written file's sha256 is recorded.
- `make check` passes.
