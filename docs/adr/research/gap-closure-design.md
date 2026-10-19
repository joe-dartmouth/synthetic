---
status: accepted
topic: research
created: 2026-10-19
updated: 2026-10-19
---
# Validated records join the training records of their asked cell, and closure is read per gap definition

## Context

The completeness criterion asks that at least half the detected gaps close after fusion
(`docs/RESEARCH.md` section 4). ADR `research/bko-formal-gap-definition` (Decision, Completeness)
fixes the request, each cell holding a per-subgroup gap asked for 20 rows, and what closing
means: the generated records that pass validation join their cell's records, and a gap closes
when that cell's fragment then grounds it. It leaves open where that runs, how a record reaches
the BKO, which records join, and how the three definitions read the result.
`bko.bkb.fusion.fuse` takes fragments with one reliability each and refuses a fused BKB as a
fragment (ADR `research/bkb-layer-source`), and `bko.gaps.detect.represent` is the gap
pipeline's only caller of it. The Messages API takes no seed, so a paid generation cannot be
drawn again.

## Decision Drivers

* **Must read closure from recorded candidates**, so a fix to the reading spends nothing and the
  figure comes again from one free command.
* **Must keep `represent` the gap pipeline's only `fuse` call**, so every definition reads the
  same records.
* **Should tie each closed gap to the records asked for it.**

## Decision

- **Two commands.** `python -m bko.generate per-subgroup --request completeness` asks each cell
  of `bko.gaps.definitions.completeness_request` for its rows and writes the candidates file; the
  request reads per-subgroup gaps, so another guidance definition is refused. `python -m
  bko.fusion <candidates>` reads that file, spends nothing, and writes the closure report. The
  file's format has one home, `bko.generate.candidates`, which writes and reads it.
- **A stage, `bko.fusion`**, downstream of `bko.validate`. It imports `bko.datasets`, `bko.gaps`,
  `bko.validate` and `bko.generate.candidates`, and no SDK.
- **Which records join.** The command refuses a file that names a cell the request does not ask
  for, or holds fewer lines for a requested cell than its rows. It validates each requested cell's
  lines in file order, a placeholder counting as generated. Of the accepted records, one whose own
  age band, sex and label name another cell is dropped, and so is one past its cell's rows, each
  drop counted; the rest follow the training split's records in request order, indexed -1, -2,
  and so on.
- **Fusion is the join.** Detection runs again over the joined records: `record_cells` builds
  each cell's dataset fragment from them and `represent` fuses it with the ontology fragments at
  weight 1, as for any dataset. `bko.fusion` calls no `fuse`, and a generated record is never a
  fragment of its own.
- **The reading.** Under each definition of ADR `research/bko-formal-gap-definition`, a (cell,
  statement) pair listed before the join and not after is closed, one listed in both is open,
  and one listed only after is new. The share is the closed pairs over the pairs listed before,
  undefined when none were. The command prints counts and pairs, never a verdict;
  `docs/EVIDENCE.md` reads the per-subgroup share against `docs/RESEARCH.md` section 4.

## Rationale

Generation is the only step that spends and the only one that cannot repeat its output, so it
writes a file and stops; every later step is a deterministic reading of that file, which a review
fix reruns for free over the same bytes. A stage of its own keeps the reading's imports light and
its direction acyclic: it reads Validate's output, which neither Detect nor Validate may import,
and `bko.evaluate`'s command line loads the synthesizers. Joining records rather than fusing a
fragment of generated records gives every definition the same input: the grounding definitions
read each cell's held concepts and the frequency-aware score reads counts, and a frame join
updates both, while a fragment's reliability would be a weight no gap list reads (ADR
`research/dataset-fragment-encoding`).

## Consequences

**Gained**: closure regenerates from the candidates file at no cost; the paid and free runs are
tied by the candidates file's sha256, which generate logs over what it writes and the closure
command logs over what it reads; `fuse`'s refusal of a fused BKB is never reached.
**Accepted**: the candidates file lives in the gitignored `scratchpad/`, so only the machine that
holds it regenerates a recorded figure, and the Runs row's digest is the record; per-subgroup and
population-level grounding only gain held concepts when records join, while frequency-aware
reference shares move with any same-label cell's records and open or close pairs in cells that
received none; the candidates file does not record its guidance level or definition, so the
recorded command names them. The join rule differs from the one E1's BKO arms apply (ADR
`research/e1-evaluation-protocol`): it keeps a value outside the training split's range, since
closure reads only which mapped concepts a record shows and validation already bounds each
value, and it drops a record its own values place in another requested cell, so each closed gap
stays tied to the records asked for it. The join numbers its rows as
`bko.baselines.records.number` does, a copy, since Fuse may not import an Evaluate stage.
`bko.gaps.report` is Detect's public report API, which Fuse imports downstream, not a helper two
stages share (ADR `tooling/src-layout`, Reversal Conditions).

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| One paid command that generates, validates and reads closure | A fix to the reading would spend again and draw different records, since nothing replays a call log | 2026-10-18 |
| A fragment of generated records fused at its own reliability | The weights change no gap list (ADR `research/dataset-fragment-encoding`), grounding reads one set of records per cell, and the frequency-aware score reads counts, not fragments | 2026-10-18 |
| Fusing the records into the fused BKB | `fuse` refuses a fused BKB as a fragment | 2026-10-18 |
| The command in `bko.evaluate` | Its command line loads the synthesizers for a free reading, and E1 reads no completeness | 2026-10-18 |
| The command in `bko.gaps` | Detect would import Validate, a downstream stage | 2026-10-18 |
| The command in `bko.validate` | It reads releases, and ADR `research/record-validation-design` keeps that stage off the `ontology` extra | 2026-10-18 |
| An off-cell record joining the cell its own values name | Closes gaps the request never asked about and moves frequency-aware reference shares with no request behind them | 2026-10-18 |
| A verdict printed by the command | A second home for the criterion's threshold, and one run carries no interval | 2026-10-18 |

## Enforcement

- Owner: `src/bko/fusion/`
- Pinned by: `tests/test_package_layout.py`, `tests/test_fusion_join.py`,
  `tests/test_fusion_closure.py`, `tests/test_fusion_cli.py`, `tests/test_fusion_pipeline.py`,
  `tests/test_generate_candidates.py`

## Reversal Conditions

**Generated records become a fragment of their own when a gap definition reads fragment
weights**; the check is a search of `src/bko/gaps/definitions.py` for a reliability. **The
closure command moves into `bko.evaluate` when E1 reads completeness**; the check is a search of
`src/bko/evaluate/experiment.py` for `completeness_request`.

## Related

- [A dataset gap is an ungrounded BKO statement](bko-formal-gap-definition.md)
- [A dataset becomes a fragment of one prior per mapped concept its records show](dataset-fragment-encoding.md)
- [Record generation: one request per cell at a chosen guidance level, every asked record handed on](record-generation-design.md)
- [E1's evaluation: repeated holdouts, one classifier, separation judged on paired intervals](e1-evaluation-protocol.md)
- [src layout, one subpackage per pipeline stage](../tooling/src-layout.md)
