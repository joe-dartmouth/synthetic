---
id: generate-bkb-curation-diabetes
title: The BKB likelihood gate on Diabetes 130, then the curated arm if it separates
status: blocked
depends_on: evaluate-diabetes-subgroup-removal
blocks: --
parallel_safe: true
owner: Shirley
milestone: final-report
paid: true
adrs: research/bkb-candidate-curation
---

# The BKB likelihood gate on Diabetes 130, then the curated arm if it separates

The set's graph is [`experiments-index.md`](experiments-index.md).

## Why this unit exists

G7 asked whether a candidate's log-probability under the fused BKB's concept draw separates
fidelity, so that keeping each cell's highest-scoring records would be worth a paid arm. On UCI
Heart Disease the gate did not separate: quality and column shapes did, column pair trends' interval
held 0 ([`docs/EVIDENCE.md`](../docs/EVIDENCE.md#g7-bkb-likelihood-curation)). UCI's small test
splits
([`docs/RESEARCH.md`](../docs/RESEARCH.md#limits-of-uci-heart-disease-as-the-evaluation-frame)) and
the few concepts its field rows map (one line per concept in G7's diagnostic) may be why. Diabetes
130 holds far more records per subgroup, and its mapping is a deliverable of its own unit, so the
same free gate can be read again there before any curated arm is built. The score, the gate command
and the arm's reserved spec fields (`Curation.LIKELIHOOD`, `over_ask`;
`cleanup:evaluate/likelihood-curation-unused`) already exist.

## Scope

**In.**

- Phase 1, free: `python -m bko.evaluate likelihood <run> --candidates <file>` over the `bko`
  arm's candidates file from the Diabetes 130 run of
  [`evaluate-diabetes-subgroup-removal`](evaluate-diabetes-subgroup-removal.md), with the gate
  ADR `research/bkb-candidate-curation` fixes unchanged, recorded as a new G7 run in
  `docs/EVIDENCE.md`. If the gate fails again, the entry reads the concept-share diagnostic (on UCI
  the score favoured records with fewer concepts present) and the unit records whether a score
  normalized per concept is the next question, as a `docs/IDEAS.md` entry.
- Phase 2, paid, only if phase 1 separates on all three measures: the arm `bko-curated`, which asks
  each cell for `over_ask` times its quota at `SETTINGS` and keeps the quota's highest-scoring
  records (ties by the sha256 order the ADR fixes), named in `bko.evaluate.named_runs.RUNS` beside
  the baselines and `bko` on the Diabetes 130 frame, its spend estimated in the Runs row before the
  run, under its ceiling and the paid-run protocol (`docs/DEVELOPMENT.md`, Evaluation).

**Out.** A change to the score or the gate before phase 1 reads them on the new frame; a curated arm
on UCI.

## Acceptance

- Phase 1's measures come from the committed command over the Diabetes 130 run's own candidates
  file, with no API call, and are recorded with the gate's verdict.
- Phase 2 is recorded with its spend and replays byte for byte, or the entry states the phase 1
  measure that stopped it, and `cleanup:evaluate/likelihood-curation-unused` is paid either way.
- `make check`.
