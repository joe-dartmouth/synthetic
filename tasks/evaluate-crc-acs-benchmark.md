---
id: evaluate-crc-acs-benchmark
title: Score bko-direct on the NIST CRC ACS excerpts with SDNist in an environment outside the lock
status: ready
depends_on: --
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
reading runs only after the confirmatory ACS run is recorded. The excerpts hold no occupation
code, so with the four disability rows alone the reading tests the sampler's fidelity and privacy
on a published benchmark more than the ontology's part. This unit lands its code early;
only unit [`evaluate-crc-acs-results`](evaluate-crc-acs-results.md) waits on
[`evaluate-confirmatory-results`](evaluate-confirmatory-results.md).

## Scope

**In.**

- SDNist in a hashed environment outside the lock, on the Python its pin needs, that reads the
  written CSV, laid out as `bench/sdnist/` under ADR
  [`tooling/benchmark-tools-outside-the-lock`](../docs/adr/tooling/benchmark-tools-outside-the-lock.md).
  SDNist 2.4 on PyPI pins numpy 1.24.2 and scipy 1.10.1 with no Python requirement, and those
  have no Python 3.14 wheels. SDNist 2.5, not on PyPI, states Python 3.9 to 3.12. The unit pins
  one of them and says which, by PyPI version or by commit SHA, every package hashed.
- The NIST CRC ACS excerpts pinned as a `Source`, their terms quoted from the archive.
- A frame over the excerpt's own 24 fields that shares the ACS frame's rows for the four
  disability items it holds, `DEAR`, `DEYE`, `DREM` and `DPHY` (ADR
  [`research/acs-frame`](../docs/adr/research/acs-frame.md)), with its label and the request: a
  replica of the training split's cells at the excerpt's row count.
- A check that each excerpt carries every field that frame reads, refusing one that does not.
- Named runs writing one `bko-direct` file per excerpt.
- The SDNist report command as its README gives it:
  `python -m sdnist.report <deidentified csv> <target dataset name>`, with targets `MA`, `TX` and
  `NATIONAL`.
- The EVIDENCE Configuration, `not yet run`.
- The environment's build command in [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md).
- A new ADR under `docs/adr/research/` that records the environment, the pin and the frame.

**Out.**

- Differential-privacy claims.

**Findings from planning.**

- PyPI holds SDNist 2.4 only, and its scipy 1.10.1 pin needs Python below 3.12. SDNist 2.5 is tag
  `v2.5.0`, commit `f048b1c`, with no wheel; its pins have Python 3.12 wheels
  ([PyPI metadata](https://pypi.org/pypi/sdnist/json) and
  [the tag](https://github.com/usnistgov/SDNist/tree/v2.5.0), read 2026-11-26). The unit's ADR picks 2.4 on Python 3.11, from hashed wheels, or 2.5
  with a dated exception to ADR
  [`tooling/benchmark-tools-outside-the-lock`](../docs/adr/tooling/benchmark-tools-outside-the-lock.md).
  The registry's `sdnist-2025` cites the data dictionary at commit `ac77a7f`.
- The excerpts are six Git LFS CSVs under SDNist's
  [`BenchmarkData/ACSDataExcerpts/`](https://github.com/usnistgov/SDNist/tree/v2.5.0/BenchmarkData/ACSDataExcerpts), whose LFS
  oids are their sha256: `ma2019.csv` `4f1e49c9...`, 562,838 bytes; `tx2019.csv` `33b17bc2...`,
  672,219 bytes; `national2019.csv` `b49ba83e...`, 1,990,721 bytes (read 2026-11-26). SDNist
  scores the 2019 files only. NIST's [open licence](https://www.nist.gov/open/license) needs no signature, so the `Source` has
  `agreement=None`.
- To settle in its ADR: a missing code for the literal `N` in numeric fields, and a writer back to
  the excerpt's values; the label, since the excerpts hold none; band edges or a flagged cell, so
  the gap request has a cell; field groups for the excerpt's consistency pairs; and which
  protocol is read against the archive.
- It re-points the sentence of ADR [`research/acs-frame`](../docs/adr/research/acs-frame.md) that
  names this unit, and registers the SDNist tool in [`docs/literature.md`](../docs/literature.md):
  NIST, SDNist, doi:10.18434/mds2-2943, its year read from that record.

## Acceptance

- The environment's build command is in `docs/DEVELOPMENT.md`.
- A test pins the excerpt `Source` and the frame's fields against the mapping.
- The named runs and the ADR exist.
- The `docs/EVIDENCE.md` entry's Configuration, status `not yet run`, exists.
- `make check` passes.
