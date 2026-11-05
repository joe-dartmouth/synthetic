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
  fill of that holdout to the store. The strict check 7 differs from the one `shared` keeps (ADR
  `research/shared-test-split-reading`): for keys the folder's `scores.csv` holds, rendered values
  must be byte-equal, NaN included, and a key the folder lacks must be a held-out measure. A
  replica's `quality` and `column-pair-trends` may differ only when the holdout's training split
  exceeds 50,000 rows. Any other difference exits 3 before rows are written.
- The store is one `rows.jsonl` per output folder: a header line, then one line per fill of every
  arm, gap and replica. Keys are sorted, and the file is flushed after each holdout. The header is
  `{"kind":"header","schema":1,"run":<run>,"folder":<folder name>,"log_sha256":<sha256 of the
  folder's scores.log>,"data":<digest>,"frame":<name>}`. A fill line is
  `{"kind":"rows","holdout":h,"arm":<the arm line's arm, "<a>" or "<a>-replica">,"attempt":k,"added":n,"sha256":s,"csv":<digest bytes>}`.
  The `csv` is the text that `bko.evaluate.data.digest` hashes, and its sha256 equals the line's
  `sha256` and the folder's `arm` line's.
- The writer refuses a non-finite value. It creates the file with mode 0600 and `O_EXCL`, and
  writes a holdout's lines only after its checks pass.
- A reader binds on `log_sha256` and the folder name, and checks that `sha256(csv)` equals the
  line's `sha256` and the folder's `arm` line's, that `added` is the row count, and that
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
a holdout's strict check 7 passes keeps a fill from a rescore that moved out of the store. Holding
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
| One file per holdout in a directory role | A directory role adds a layout the run-folder ADR (`tooling/run-folders`) does not have; one file serves every reader | 2026-11-05 |
| Gzip on the store | The hand check needs plain `jq` and `shasum` | 2026-11-05 |
| Pickle or parquet | Neither is checkable by hand, and a pickle runs code on load | 2026-11-05 |
| Privacy measures (MI, NNDR) as harness measures | They would move every recorded run's bytes | 2026-11-05 |
| One pass that scores the panel while regenerating | A panel failure would repeat the regeneration, and the rows would stay unchecked | 2026-11-05 |

## Enforcement

- Owner: `bko.evaluate.rows` (the command and the store's reader and writer)
- Pinned by: the rows test module `test_evaluate_rows`

## Reversal Conditions

**An E6 slice rescored with `rows` reads `matches_folder=false`** on a commit that is meant to
reproduce it: the scoring path moved, and the store waits until the rescore reproduces the folder
again.

## Related

- [A run is read beside another on the test split neither copies](shared-test-split-reading.md)
- [Call-log replay](call-log-replay.md)
