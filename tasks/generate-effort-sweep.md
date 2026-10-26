---
id: generate-effort-sweep
title: The lowest effort that keeps validity and fidelity
status: ready
depends_on: --
blocks: evaluate-e1-second-configuration
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: --
---

# The lowest effort that keeps validity and fidelity

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

`cleanup:generate/haiku-thinks-every-call`: at effort `medium`, every one of `e1-3`'s 9340 replies
holds a `thinking` block, output tokens per record asked are about twice `e1-1`'s, and output is
about three quarters of the spend at `bko.generate.spend.PRICES`. Whether a lower effort keeps
validity and replica fidelity is unmeasured, and the answer sets `SETTINGS.effort` for every later
paid run.

## Scope

**In.**

- A way to run one BKO arm on a chosen subset of E1's holdouts: a `--holdouts` and `--arms`
  option on `bko.evaluate score`, or a committed script under `tools/` that calls the harness, so
  the sweep is one command per effort.
- The `bko` arm's gap and replica fills at effort `low` on the first 10 holdout seeds, at
  `SETTINGS` otherwise unchanged; the `medium` control is `e1-3`'s log replayed over the same
  seeds. Per effort: validity, candidates dropped as outside the range, replica fidelity (quality,
  column shapes, column pair trends), replica utility, tokens per record asked, seconds per call,
  and spend. Spend estimate: a tenth of one arm's share of `e1-3`'s spend, less at `low`.
- The decision the cleanup entry states: `SETTINGS.effort` moves to the lowest effort whose
  validity rate meets `E1.validity` and whose replica fidelity does not fall below `medium`'s, or
  the entry's record states that `medium` is that effort. The move is a new configuration of E1,
  run by [`evaluate-e1-second-configuration`](evaluate-e1-second-configuration.md).
- A new `docs/EVIDENCE.md` entry for the sweep, with each run's spend and log location in its
  Runs field, and the cleanup entry deleted or narrowed in the landing commit.

**Out.** A full E1 run; any other change to `SETTINGS`.

## Acceptance

- Each effort's measures come from one committed command, recorded in `docs/EVIDENCE.md` with the
  run's spend.
- `SETTINGS.effort` is set by the stated rule, and the record says which measure decided it.
- `make check`.
