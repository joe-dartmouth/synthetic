---
status: accepted
topic: research
created: 2026-11-05
updated: 2026-11-05
---
# A recorded run's rows are rebuilt once, checked against its log, and stored with their hash

## Context

A reading that scores a recorded run's arms again, or measures its added rows, needs the rows each
arm added on every holdout. A Messages arm's rows cannot be drawn again (ADR
`research/call-log-replay`), and a seeded baseline's rows cost compute to regenerate on every
holdout. A reading that rebuilt them itself would repeat that cost per reading and could drift from
the run's rows without notice. Each folder's `scores.log` already names every arm's rows by
sha256, so one rebuild can be checked against it and kept.

## Decision Drivers

* **Must rebuild each arm's rows once**, so later readings read a file and regenerate nothing.
* **Must make no API call**, reading only what the folder holds.
* **Must bind the rows to the run's log**, so a store written from other code or data is refused.
* **Must let a reader verify a fill by hand**, with no code of the repository.

## Decision

- `python -m bko.evaluate rows <run> --folder <f> --rows <path>` rescores a recorded folder
  through `bko.evaluate.shared.rescore`, at the folder's own models. It makes no API call.
- After a holdout passes identity checks 3 and 4 and the strict check 7, the command appends every
  fill of that holdout to the store. The strict check 7, with `rows` admitting held-out keys only,
  and its 50,000-row exception are stated in ADR `research/shared-test-split-reading`.
- The store is one `rows.jsonl` per output folder: a header line, then one line per fill of every
  arm that has a runner, gap and replica. Keys are sorted, and the file is flushed and synced to
  disk after each holdout. The header is
  `{"kind":"header","schema":1,"run":<run>,"folder":<folder name>,"log_sha256":<sha256 of the
  folder's scores.log>,"data":<digest>,"frame":<name>}`. A fill line is
  `{"kind":"rows","holdout":h,"arm":<the arm line's arm, "<a>" or "<a>-replica">,"attempt":k,"added":n,"sha256":s,"csv":<digest bytes>}`.
  The `csv` is the text that `bko.evaluate.data.digest` hashes, and its sha256 equals the line's
  `sha256` and the folder's `arm` line's.
- The writer refuses a non-finite value, a fill whose `arm` line is not the folder's, and a
  holdout and arm already stored. It creates the file with mode 0600 and `O_EXCL`, and writes a
  holdout's lines only after its checks pass.
- A reader binds on `log_sha256`, the folder name and the header's `frame`, and requires one fill
  line for each of the folder's `arm` lines, so a store whose run stopped early is refused. It
  refuses a line that is not a JSON object, and checks that `sha256(csv)` equals the line's
  `sha256` and the folder's `arm` line's, that `added` is the row count, that the CSV reads at
  `frame.dtypes` with exactly `frame.columns`, and that
  `digest(pd.read_csv(io.StringIO(csv), index_col="record_id", dtype=frame.dtypes,
  float_precision="round_trip"))` equals `sha256`.
- A hand check needs only `jq` and `shasum`:
  `jq -j 'select(.kind=="rows" and .holdout==0 and .arm=="ctgan") | .csv' rows.jsonl | shasum -a 256`
  prints the sha256 of the folder's `arm holdout=0 arm=ctgan` line.
- The panel's shortfall WARNING lines are not reproduced, because a stored fill does not run the
  generator again.

## Rationale

`rescore` already rebuilds each arm's rows through the code that scored the run and refuses a
folder whose lines disagree, so a store written inside it inherits those checks. Writing only after
a holdout's strict check 7 passes keeps a fill from a rescore that no longer reproduces the folder out of the store. Holding
the CSV text, whose hash the log already records, lets a reader verify a fill without the
repository, and the round-trip read check shows the text parses back to the same frame.
Binding on the log's sha256 ties the store to one run of record.

## Consequences

**Gained**: every later reading of the run's added rows reads one file, with no regeneration and no
API call, and a fill is checkable by hand.
**Accepted**: the store holds every holdout's rows as text, so it is large; it is rebuilt only at a
commit whose rescore reproduces each `arm` line; a run whose log lacks an `arm` line for a fill
cannot be stored.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| One file per holdout in a directory role | `tooling/run-folders` keeps a role to one regular file except `pytest` and `rehearsal`; one file whose lines name their holdout serves every reader, the reason of that ADR's candidates row | 2026-11-05 |
| Gzip on the store | The hand check needs plain `jq` and `shasum` | 2026-11-05 |
| Pickle or parquet | Neither is checkable by hand, and a pickle runs code on load | 2026-11-05 |
| Privacy measures (MI, NNDR) as harness measures | They would move every recorded run's bytes | 2026-11-05 |
| One pass that scores the panel while regenerating | A panel failure would repeat the regeneration, and a failed reading would couple the stage-1 checks to stage 2 | 2026-11-05 |

## Enforcement

- Owner: `src/bko/evaluate/rows.py` (the command and the store's writer),
  `src/bko/evaluate/rows_read.py` (the store's reader)
- Pinned by: `tests/test_evaluate_rows.py`

## Reversal Conditions

**An E6 slice with no added held-out measure, rescored with `rows` at the commit its folder's
`versions` line names, reads `matches_folder=false` or exits 3 on the strict check 7**: the scoring
path moved, and the store waits until the rescore reproduces the folder again.

## Related

- [A run is read beside another on the test split neither copies](shared-test-split-reading.md)
- [Call-log replay](call-log-replay.md)
