---
status: accepted
topic: research
created: 2026-11-01
updated: 2026-11-05
---
# A run is read beside another on the test split neither copies, rescored from both runs' own files

## Context

`compare --beside` pairs a holdout only when no arm in either file copied a test record there,
because each run removes from its test split every record any of its own arms copied, so two runs
that copy different records score on different splits (ADR `research/e1-evaluation-protocol`). G6
reads each generator run beside `e1-4`'s `bko`. A run whose arms copy test records on many
holdouts leaves few holdouts to pair, and the comparison then rests on the holdouts where the
generator happened not to copy.

Each run's folder already holds what its scores were built from: the `scores.log` names the data,
the experiment, each holdout's split and each arm's rows by sha256, the candidates file holds every
BKO row the run kept, and the baselines are seeded. Rerunning a run repeats a run whose results
exist, and a Messages run's rows cannot be drawn again (ADR `research/call-log-replay`).

## Decision Drivers

* **Must pair holdouts on one test split**, so a paired difference compares arms, not splits.
* **Must make no new run and no API call**, reading only what each run's folder holds.
* **Must score through the code that scored the run**, so a rescore with nothing removed reproduces
  the run's scores.
* **Must refuse two runs that differ in anything but their BKO rows**, so a pairing never mixes
  data, holdouts or baselines.

## Decision

- `python -m bko.evaluate shared <run> --folder runs/<id> [--with runs/<id>]...`
  (`bko.evaluate.shared`) rescores the folder's run on each holdout's test split less every record
  an arm of the folder or of a `--with` folder copies. It rebuilds each BKO arm's rows from its
  `holdout-candidates.jsonl` (the last returned attempt's `kept` lines, grouped by each row's own
  cell in grid order), regenerates each baseline with its runner, and passes the `--with` folders'
  BKO rows, rebuilt the same way, to `bko.evaluate.run.score_holdout` as `exclude`; their baselines'
  rows are the folder's own (check 5), so they would exclude nothing more. It writes a
  `scores.csv` to stdout and a `scores.log` to stderr, which names each input folder and the
  sha256s of its `scores.log` and candidates file on a `shared folder=` line.
- A holdout scored with `exclude` writes a held-out `shared-split` row (`Measure.SHARED_SPLIT`)
  whose value is the first 12 hex digits of the sha256 of its kept test record ids, sorted, and
  logs `shared holdout=%d records=%d copied=%d foreign=%d fingerprint=%s`. With no `exclude`, a
  holdout's rows are those `score` writes.
- `bko.evaluate.beside.pairable` pairs a holdout an arm copied when both files hold bitwise-equal
  `shared-split` rows there; a file with the row never pairs a copied holdout with a file without
  it.
- The split is read per pair: a run is rescored with one compared run, and the compared run with
  it, so one run's copies never shrink another pair's split.
- `--with` on a run that keeps copied test records (`Experiment.decontaminate` false) exits 3: its
  arms score on the whole split, copies included, so no other run's copies can leave it one split
  with that run. A folder given twice, as `--folder` and `--with` or as two `--with`, exits 3.
- Seven identity checks hold; the command runs checks 1 to 5 and exits 3 when one fails:
  1. The data digest is equal across the folder and every `--with` folder (from the `data` line or
     a schema-1 `uci` header).
  2. The `experiment` lines are equal: the folder's line names the rescored run and equals that
     run's line now on every key, and each `--with` folder's equals the folder's on every key but
     the run's own (`run`, `arms`, `replica`, `spend_ceiling`, `criteria`). A line written before
     a key existed may lack it only if the key has an assumed value
     (`bko.evaluate.run_folder.ASSUMED`: `frame` `uci`, `removal` `none`, `training_fraction` `1`,
     `decontaminate` `true`, `threshold` `20`, and `seeds` `0-<holdouts - 1>`); the other line must
     hold that value. Any other missing key exits 3.
  3. Each folder logs a `split seed= train= test= fingerprint=` line per holdout its `experiment`
     line counts, and each holdout's line is equal across the folders and to the rescore's. The
     rescore's line is read from a `Logged` collector that `run.score_holdout` fills with the
     lines it logs.
  4. Each folder logs, on each holdout, an `arm` line for each arm and replica fill its
     `experiment` line names and no other, so a line format that drifts is refused rather than
     skipped. Each rebuilt BKO arm row set, and each regenerated baseline row set, reproduces its
     logged `arm ... sha256=`, attempt, seed and params (a BKO arm's params are read from its
     line); each `--with` folder's rebuilt BKO gap and replica row sets reproduce their logged
     attempt, row count and sha256. The rescore's `arm` lines are read from the same `Logged`
     collector.
  5. The `--with` folders' non-BKO `arm` lines equal the folder's own.
  6. Scoring goes through `run.score_holdout`.
  7. Self-reproduction holds: with no `--with`, the output equals the folder's `scores.csv`, and
     the `scores` line logs `matches_folder=true`. The one exception is a held-out measure added to
     the harness after the run was scored: the output then holds every original row and that
     measure's rows besides, and the line logs `matches_folder=false`. `shared` keeps this log-only
     check: the tests pin it, and the command logs it and does not refuse. `rows` and `panel`
     (ADR `research/rebuilt-rows-store`) apply a strict check 7 on each holdout: for keys the
     folder's `scores.csv` holds, rendered values must be byte-equal, NaN included, and a key the
     folder lacks must be one the command admits (`rows` admits held-out keys only; `panel` admits
     its panel-model suffixes and the calibration measures). A replica's `quality` and
     `column-pair-trends` may differ only when the holdout's training split exceeds 50,000 rows,
     because a folder scored before `bko.evaluate.fidelity` read every row holds one unseeded
     SDMetrics subsample draw. Any other difference exits 3 before rows are written.
- A missing input, or an arm with no runner, exits 2.

## Rationale

Removing the union of both runs' copies from each holdout's test split gives both runs one split,
so every holdout where both runs' other rows match can pair. The fingerprint over the kept record
ids lets `beside` check that the two files were scored on that one split rather than trust the
command line. Rebuilding rows from the folder, checked against the logged sha256 of each arm's
rows, reads the run as it ran; scoring through `run.score_holdout` keeps one scoring path, and
self-reproduction with nothing removed shows that path still writes the run's bytes. A held-out
measure added after a run was scored is new output, not a moved score, so self-reproduction admits
its rows and no other difference. An assumed value reads an older `experiment` line at the setting
its run used, so the line is checked key by key rather than on the keys both lines happen to
name. Reading per pair keeps each pair's split as large as its two runs allow.

## Consequences

**Gained**: G6's generators read beside `e1-4` on every holdout whose other rows match, at no
cost, from files already on disk; the plain `--beside` reading stays beside it.
**Accepted**: each pair's test split differs from the next pair's, so readings of two pairs are
not on one split; the rescore holds on one machine at the run's data, releases and a commit whose
scoring writes the run's bytes, as replay does; a run whose candidates file lacks its kept rows
cannot be rescored; nor can a run with a shortfall's partial fill, whose lines hold no returned
attempt, so the file does not mark the rows the run kept.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Rerunning each run on a common split | Repeats runs whose results exist, and a Messages run draws new rows | 2026-11-01 |
| One split for every G6 run, less every run's copies | One run's copies would shrink every other pair's split | 2026-11-01 |
| Letting `beside` pair copied holdouts with no marker | Two files scored on different splits would pair | 2026-11-01 |
| A scoring path of its own for the rescore | Self-reproduction would no longer show the run's scoring | 2026-11-01 |
| Capturing the harness's lines with a logger handler | A handler's level and propagation depend on the process's logging setup; `score_holdout` fills a `Logged` collector instead, so checks 3 and 4 read what it logged | 2026-11-05 |

## Enforcement

- Owner: `src/bko/evaluate/shared.py` (the command), `src/bko/evaluate/run_folder.py` (the
  folder's reading and checks 1, 2, 3 and 5), `src/bko/evaluate/shared_split.py` (the marker)
- Pinned by: `tests/test_evaluate_shared.py`, `tests/test_evaluate_beside.py`,
  `tests/test_evaluate_run.py`

## Reversal Conditions

**A rescore with no `--with` stops reproducing its folder's `scores.csv`** at the commit the
folder's `versions` line names: the `scores` line logs `matches_folder=false` for a run with no added
held-out measure, or `rows` exits 3 on the strict check 7. The scoring path moved, and the rescore
waits until it reproduces again.

**`score` stops removing copied test records** (`Experiment.decontaminate` false for every run):
every run then scores on the whole split, and the marker has nothing to pair.

## Related

- [E1's evaluation](e1-evaluation-protocol.md)
- [Call-log replay](call-log-replay.md)
- [On-device generators](local-generator-backend.md)
