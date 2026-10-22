# Cleanup backlog

Small owed work that no task unit carries: review findings judged outside the boundary of the
change that found them, costs that do not bite at today's sizes, and gaps a later reader will trip
over. Each entry names its blocker when it has one.

The research queue lives elsewhere: planned work is a task unit
([`proposal-index.md`](proposal-index.md)), and unscheduled research is in
[`docs/IDEAS.md`](../docs/IDEAS.md). An entry here is not on any plan. It waits for a change that
passes close enough to pay it.

## The contract

- **A review that defers a finding files it here.** A finding fixed in its own diff needs no
  record. One judged out of boundary gets an entry, or it is lost.
- **Read this file before scoping a change.** Pay any entry in the surface you are about to touch.
- **Keys are `cleanup:<topic>/<slug>`.** The topic is the surface whose code must change, not the
  place the symptom shows: a stage under `src/bko/` (`ontology`, `datasets`, ...) or one of `docs`,
  `tests`, `tools`, `tasks`. The slug says what is wrong. Cite an entry by its key in a code span.
  Two filings of one finding write the same key and collide, which is the point. A key is never
  reused.
- **An entry carries its evidence**: a reproduction, a measurement, or the quoted output that
  shows the problem. It also says why it was deferred, what discharges it, and its trigger: the
  change that should pay it.
- **`Blocked on` marks an entry that cannot be paid yet.** It names the missing thing and links to
  where the unblocking work is tracked. A blocked entry is still owed; it is not a softer deletion.
- **An entry leaves when it stops describing owed work, and is deleted, not archived.** Git is the
  history; this file is the queue. Delete it in the commit that pays it, or in a commit that says
  why it no longer holds (it did not reproduce, its subject was removed, or another document now
  owns the work and states the argument itself). An entry paid in part stays, narrowed to what is
  still owed. Deleting an entry also deletes every citation of its key.
- **One heading per topic**, and a heading goes with its last entry.
- **No counts.** Derive them from the file when you need them.

`tests/test_cleanup_backlog.py` enforces the key form, the topic set, unique keys, that every cited
key resolves, that each entry names what discharges it and its trigger, and that a blocked entry
links to where its blocker is tracked.

## evaluate

- **`cleanup:evaluate/score-cannot-resume`: an interrupted `score` run cannot continue, so a stop
  late in a paid run repays every call before it.** `score` refuses an existing call log and
  `--replay` refuses a log with calls left over or missing (`src/bko/evaluate/__main__.py:172-173`,
  `:226-250`), so neither picks up where a run stopped. `e1-2` stopped at holdout 83 of 100 when the
  runner's machine hibernated on battery (E1's Runs row); `e1-3` finished it through
  `scratchpad/resume-run.py`, a gitignored script that replays the stopped log's answered calls
  through `Context.send` and then sends live, for $2.15 of new spend against about $12.82 for a
  fresh run. *Filed 2026-10-21 by `proposal-e1-replica-run`.*
  **Why deferred.** It is a code change, and the evidence unit that found it changed no code.
  **What discharges it:** `score --resume LOG` answers a stopped call log's answered calls from it
  and sends the rest live, writing one call log that `score --replay` regenerates byte for byte,
  with a test that resumes a stopped fake run; ADR `research/e1-evaluation-protocol`'s "a run
  cannot resume" and its consequence that a call with no response forfeits the run's spend change
  with it.
  **Trigger:** `proposal-guidance-ablation`, whose paid runs each take as long as `e1-3`.

## fusion

- **`cleanup:fusion/completeness-counts-lines-not-calls`: the closure command checks each cell's
  line count, not that each of its calls answered.** `bko.fusion.__main__._answer` refuses a cell
  holding fewer lines than its rows. `bko.generate.decode` keeps every decoded item past the asked
  count (`src/bko/generate/decode.py:69-70`), so one call can write more lines than it asked for.
  With `SETTINGS.per_call` below a cell's rows, an early call that over-delivers can make up for a
  later call that never ran, and a failed run's file then passes as an answer. Today
  `per_call=20` equals `UNDERREPRESENTED_BELOW`, so each requested cell is one call and the hole
  cannot open. Found by static trace in PR #18's review. *Filed 2026-10-19 by
  `proposal-fusion-repair`.*
  **Why deferred.** The candidates file records no call's asked count, so the fix needs a format
  change in `bko.generate.candidates` or a call-log reader. Neither is warranted while each cell is
  one call.
  **What discharges it:** the closure command refuses a cell where any call wrote fewer lines than
  it asked for.
  **Trigger:** a change that sets `SETTINGS.per_call` below `UNDERREPRESENTED_BELOW`.

## gaps

- **`cleanup:gaps/uci-load-sequence-copied`: the UCI load sequence is written once per command and
  once in the app.** `ensure`, `load_uci`, `split`, `load_scope` and `record_cells` run in sequence
  in `src/bko/gaps/__main__.py:177-188`, `src/bko/fusion/__main__.py:152-167`,
  `src/bko/generate/__main__.py:202-217`, `src/bko/evaluate/__main__.py:175-176,217-224` and
  `bko.app.steps.analyze`. *Filed 2026-10-21 by `proposal-streamlit-app`.*
  **Why deferred.** Extracting it edits four commands owned by others, and
  `proposal-guidance-ablation` edits `bko.evaluate`'s command.
  **What discharges it:** one public loader in `bko.gaps` that every command and `bko.app` call.
  **Trigger:** `proposal-guidance-ablation` landing, or the next change to any copy.

## generate

- **`cleanup:generate/usd-log-precision`: stderr rounds a call's spend to four decimals, a seventh
  of a typical call at Haiku 5.5's prices.** `bko.generate.calls` and `bko.generate.log` write
  `usd=%.4f` (`src/bko/generate/calls.py:141`, `src/bko/generate/log.py:166`). A call of 1000
  input and 500 output tokens costs 0.00035 USD and prints `usd=0.0003`. The call log's `usd`
  fields keep full precision, so no recorded figure is lost. Raised in PR #24's review. *Filed
  2026-10-19 by `proposal-bko-replicas`.*
  **Why deferred.** Only the stderr lines round, and every Runs row reads spend from the call log.
  **What discharges it:** both lines print `usd` with enough decimals to show a single Haiku 5.5
  call, with their test pins updated.
  **Trigger:** the next change to either line's format.

- **`cleanup:generate/candidates-carry-no-run-identity`: a candidates file does not name the split,
  releases or guidance level it was generated against.** `python -m bko.fusion` re-derives the
  split and the request and logs the split fingerprint. It refuses only a file whose cells or line
  counts do not answer the request, so a file generated against another split or release with the
  same requested cells is read as if it answered this one. The run's record ties the two through
  the candidates sha256 that both commands log (ADR `research/gap-closure-design`, Consequences).
  Raised in PR #18's review. *Filed 2026-10-19 by `proposal-fusion-repair`.*
  **Why deferred.** It changes the candidates format, which `bko.generate.candidates` writes and
  reads, and each completeness run is checked by digest.
  **What discharges it:** generate writes the split fingerprint, release versions and guidance
  level where the closure command reads them, and the closure command refuses a mismatch.
  **Trigger:** `proposal-guidance-ablation`, which reads several completeness runs.

- **`cleanup:generate/prompt-range-wider-than-training`: the user turn asks for ages the training
  split never holds, so about a fifth of BKO records are dropped.** `prompt.render` states each
  numeric field's bounds from `guidance.intervals` (`src/bko/generate/prompt.py:191-204`), which are
  `RANGES`' plausible bounds with age clipped to the cell's band
  (`src/bko/generate/guidance.py:140-165`): age 18 to 54 and 55 to 120, trestbps 10 to 250. The drop
  rule keeps only rows inside the training split's minimum and maximum
  (`src/bko/baselines/records.py:43-44`, `:150-155`), and UCI's ages run 29 to 77. In `e1-3`,
  `bko`'s fills dropped 5840 of 31643 generated records as outside the range (its `validity` line in
  E1's Result), and of the 94753 records its call log decodes, 7.2% are younger than 29 and 9.3%
  older than 77. Each drop costs a top-up call, and the kept records spread wider in age than the
  training rows (a standard deviation of 9 to 11 years per cell against 3 to 6). The figures come
  from `env -u ANTHROPIC_API_KEY uv run --no-sync python -I scratchpad/quality.py
  scratchpad/e1-3.jsonl` (gitignored, sha256
  `aa21abfedf484d8f23677ddfe5805bf055906aaf40aef928670403a06ec834fb`). *Filed 2026-10-21 by
  `proposal-e1-replica-run`.*
  **Why deferred.** It changes what every call sends, so it is a code change and a new
  configuration of `E1`, and the evidence unit that found it changed no code.
  **What discharges it:** the user turn bounds each numeric field by the training split's minimum
  and maximum, the bounds the drop rule enforces, with a test pinning the rendered text.
  **Trigger:** the next change to `prompt.render`'s numeric bounds, or a new configuration of
  `E1`.

- **`cleanup:generate/replica-marginals-drift`: at level `ontology`, BKO records match their cell's
  sex, target and age band but not its field distributions.** Over the 94753 records `e1-3`'s call
  log decodes, against the real records of each requested cell (pooled, weighted to the synthetic
  cells): `cp` 4 at 0.28 against 0.50, with `cp` near uniform; `restecg` 1 at 0.26 and 2 at 0.13
  against 0.01 and 0.49; `thal` 6 at 0.25 against 0.06; `ca` 0 at 0.42 against 0.61; `chol`'s
  standard deviation narrower in every cell (28 to 37 against 35 to 80); `thalach` 14 to 24 bpm
  lower in the 55-and-over cells; and `trestbps` on a multiple of 10 in 14.2% of records against
  55.2% of real ones. E1's Result reads the effect: trained on its replica alone, `bko` scores a
  fidelity quality of 0.7766, below every non-BKO arm's, and a flagged-subgroup recall of 0.3260.
  The prompt at level `ontology` gives no frequencies or spreads; levels `profile` and `summary` add
  them (`src/bko/generate/prompt.py:129-153`, `src/bko/generate/messages.py:22-30`). The figures
  come from `env -u ANTHROPIC_API_KEY uv run --no-sync python -I scratchpad/quality.py
  scratchpad/e1-3.jsonl` (gitignored, sha256
  `aa21abfedf484d8f23677ddfe5805bf055906aaf40aef928670403a06ec834fb`). *Filed 2026-10-21 by
  `proposal-e1-replica-run`.*
  **Why deferred.** Which level E1 runs is a result of `proposal-guidance-ablation`, not a fix in
  this unit.
  **What discharges it:** E1 runs its BKO arms at the level whose replica column shapes and column
  pair trends the guidance ablation measures closest to the training rows, or the ablation's
  record states that no level closes the drift.
  **Trigger:** `proposal-guidance-ablation`'s result.

- **`cleanup:generate/haiku-thinks-every-call`: at effort `medium`, Haiku 5.5 thinks on every call
  and writes about twice Opus 5.5's output tokens per record.** `SETTINGS` sends
  `output_config.effort` `medium` (`src/bko/generate/messages.py:76`). All 9340 replies in `e1-3`'s
  call log hold a `thinking` block before their text, against 13 of 603 in `e1-1`'s; output tokens
  per record asked are 197.046 against 94.163, and output is about three quarters of the spend at
  `bko.generate.spend.PRICES`. Seconds per call are unchanged (8.1 against 8.2 over the first 682
  calls of `e1-2` and all of `e1-1`), since Haiku streams about twice as fast. The block counts come
  from `jq -r 'select(.kind=="reply") | .response.content[0].type'` over each call log, and the
  seconds per call from `scratchpad/latency.py` (gitignored, sha256
  `e0a1d87e242f6e43b489f0ada80f9b9a07b588fc93b8bb98ae3f15b8f1c05e25`) over `scratchpad/e1-1.jsonl`
  and `scratchpad/e1-2.jsonl`. *Filed 2026-10-21 by `proposal-e1-replica-run`.*
  **Why deferred.** Changing effort is a new configuration of `E1`, and whether a lower effort
  keeps validity and fidelity is unmeasured.
  **What discharges it:** `SETTINGS` uses the lowest effort whose validity rate meets
  `E1.validity` and whose replica fidelity does not fall below effort `medium`'s on a measured
  run, or this entry records that `medium` is that effort.
  **Trigger:** the next change to `SETTINGS`.
