---
id: evaluate-crc-acs-benchmark
title: Score bko-direct on the NIST CRC ACS excerpts with SDNist in an environment outside the lock
status: blocked
depends_on: datasets-acs-frame
blocks: evaluate-crc-acs-results
parallel_safe: true
milestone: direct-method-research
paid: false
adrs: research/crc-acs-comparison
---

# Score bko-direct on the NIST CRC ACS excerpts with SDNist in an environment outside the lock

The set's graph is [`direct-index.md`](direct-index.md).

## Why this unit exists

[`docs/RESEARCH.md`](../docs/RESEARCH.md#published-comparisons) runs four protocols beside
published numbers. NIST's Collaborative Research Cycle publishes ACS excerpts, deidentified
by many methods and scored with SDNist ([archive](https://doi.org/10.18434/mds2-3024)). The
reading runs only after the confirmatory ACS run is recorded, which is why this unit follows
[`evaluate-confirmatory-results`](evaluate-confirmatory-results.md) through its results unit.

## Scope

**In.**

- SDNist in a hashed environment outside the lock on Python 3.12 that reads the written CSV, as
  `video/` does. SDNist 2.4 on PyPI pins numpy 1.24.2 and scipy 1.10.1 with no Python
  requirement, and those have no Python 3.14 wheels. GitHub main is 2.5, not on PyPI, and states
  Python 3.9 to 3.12. The unit pins one of them and says which.
- The NIST CRC ACS excerpts pinned as a `Source`, their terms quoted from the archive.
- A frame over the excerpt's fields that reuses the ACS mapping of unit
  [`datasets-acs-frame`](datasets-acs-frame.md), with its label and the request: a replica of the
  training split's cells at the excerpt's row count.
- Named runs writing one `bko-direct` file per excerpt.
- The SDNist report command as its README gives it:
  `python -m sdnist.report <deidentified csv> <target dataset name>`, with targets `MA`, `TX` and
  `NATIONAL`.
- The EVIDENCE Configuration, `not yet run`.
- The environment's build command in [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md).
- A new ADR under `docs/adr/research/` that records the environment, the pin and the frame.

**Out.**

- Differential-privacy claims.

## Acceptance

- The environment's build command is in `docs/DEVELOPMENT.md`.
- `make check` passes.
