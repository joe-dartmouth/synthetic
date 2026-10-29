---
status: accepted
topic: research
created: 2026-10-28
updated: 2026-10-28
---
# The Synthea downstream design: a 40-and-over frame, one thinned cell and an in-run reference

## Context

UCI gives E1 two underrepresented cells and about five positive test records in the flagged
subgroups per holdout, and no arrangement of the archive adds more (ADR
[`research/uci-evaluation-frame`](uci-evaluation-frame.md)). E3 runs E1's arms on the pinned
Synthea population of ADR [`research/synthea-ablation-design`](synthea-ablation-design.md)
instead, where the truth is fixed by construction: a share of one cell's cases leaves the training
rows, and the recall each method recovers on that cell's held-out cases measures the augmentation.

`score` reads each named run's frame from one `bko.evaluate.data.Data` value (ADR
[`research/uci-evaluation-frame`](uci-evaluation-frame.md)), so a Synthea frame enters as a
dataset. The population names no feature set, label, age bands, removed cell, copy rule or spend,
and every E3 number depends on all of them. Its records repeat far more than UCI's: a patient is
an age, a sex and a few 0/1 conditions, so many patients share every field.

The pinned population is
`data/synthea/ed43c20ad40ba5c3-s1-cs1-p2000-d20260101-massachusetts`, snapshot 2026-01-01. Every
figure below is what the command beside it in Rationale prints against that population.

## Decision Drivers

* **Must keep the removed cell's held-out cases**, so recovery is measured on real cases the
  training rows lost.
* **Must leave every other training cell at 20 rows or more**, so the removed cell is the only gap
  and the headline reads one subgroup.
* **Must pair every arm, the reference included, on identical splits**, as E1 does.
* **Must reproduce byte for byte from the pinned population**, with no JDK.
* **Should bound the replica spend**, since three BKO arms fill a replica of the training split
  on every holdout.

## Decision

- **Population**: the patients alive on 2026-01-01 and aged 40 and over: 985 patients ([C9]). No
  case is lost, since the youngest case is 44 ([C7]).
- **Label**: `target` is 1 for a patient with ischemic heart disease, SNOMED-CT `414545008`, by the
  snapshot date, the case rule of `bko.datasets.synthea.subjects`, which
  `bko.gaps.observe.observe_population` reads too.
- **Fields**: `age` in whole years, `sex`, and one 0/1 indicator per `CODE_MAPPINGS` row other than
  ischemic heart disease, in that table's order: hypertension, type 2 diabetes, hyperlipidemia,
  myocardial infarction, ST-elevation myocardial infarction and stroke. An indicator is 1 when the
  patient holds the code by the snapshot date.
  - Non-ST-elevation myocardial infarction is in `UNMAPPED_CODES`, so it has no column.
  - No mapped code is held by cases alone ([C7]), so no feature stands in for the label.
  - Myocardial infarction and ST-elevation myocardial infarction stay, read cross-sectionally at
    the snapshot: P(case | MI) is 35/65, and 20 of the 35 cases with one had it begin after their
    ischemic heart disease began ([C7]). Every ST-elevation holder also holds myocardial
    infarction ([C11]).
- **Bands**: one edge at 65, `(65,)`, so 8 cells. Among the single edges 55, 60, 65 and 70, only 65
  leaves no training cell under 20 ([C8], [C9]).
- **Split**: E1's cell-stratified split at E1's holdout seeds 0 to 99 (ADR
  [`research/uci-evaluation-frame`](uci-evaluation-frame.md), The split).
- **Removal**: 0.9 of the training rows of cell `(>=65, female, 1)` leave each holdout's training
  split, drawn by `removal_order` at the holdout's seed and counted as the ablation counts a
  fraction (ADR [`research/synthea-ablation-design`](synthea-ablation-design.md)): 47 of 52
  removed, 5 kept, and the cell's 13 test cases untouched ([C8], [C10]). The test split is never
  thinned.
- **Fill**: E1's `fill_to` of 20, so each arm is asked for 15 rows of the removed cell.
- **Headline**: recall at scope `flagged-subgroups`. With every other training cell at 23 or more
  ([C9]), `(>=65, female)` is the only flagged subgroup, and its positive test records are the
  removed cell's test cases, so the scope's recall is the recall on those cases. Each holdout
  writes a `flagged-fingerprint` held-out score of the cells it flagged, and `compare --beside`
  pairs a holdout only when both files' fingerprints match or either file lacks one.
- **Reference**: arm `real` adds the removed rows back, so its augmented training set is the
  un-ablated training split, with no runner and no `check`. Every other arm is read against it on
  the same holdout.
- **Held-out ratio**: the corrected interval's n2/n1 is read from the un-ablated split: 0.261
  ([C9]). The ablated cell's 13/5 = 2.6 is not used.
- **Copies**: counted, not removed (`Experiment.decontaminate` is false for E3): a test record an
  arm copies stays in the test split every arm is scored on, and the `test-copies` scores and the
  `test` log line still count it. Each holdout of such a run writes a `copies-kept` held-out row
  (value 1.0), and a decontaminating run writes none. `compare` calls
  `bko.evaluate.beside.pairable` with `allow_copies` for such a run, so a copy excludes no holdout
  where both files hold `copies-kept`; beside a file that lacks it there, a copy in either file
  excludes the holdout, since that file scored its arms on a split without the copied records.
- **Privacy**: the distance-to-closest-record rows are reported with no verdict.
- **Egress**: each BKO call sends the frame's study and cohort text, its codebook and indicator
  labels, age's training-split bounds within the cell's band, the cell and its training record
  count, the gap findings with their ontology chains and, at `summary`, the reference group's
  record count, each mapped concept's share of it and its age quartiles, the reference group being
  the training records outside the cell that share its label; no record's values are sent, and the
  rows are generated, so none describes a person
  ([`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#generation), What leaves the machine).
- **Arms**: E1's arms (`METHOD_ARMS` and `BKO_ARMS`) plus `real`. Replica arms: the row-adding
  methods (`ADDS_ROWS`) and the three BKO arms. Neither `bko-direct` nor `bko-skeleton` runs.
- **Settings**: `SETTINGS` itself (`claude-haiku-5-5`, effort `medium`, `max_tokens` 16000,
  `per_call` 20, level `summary`), the effort G4 chose
  ([G4](../../EVIDENCE.md#g4-effort-at-summary)).
- **Criteria**: none; E3 is reported with no verdict, and its headline reads each arm beside
  `real` and `none`.
- **Named runs and slices**: `e3` in `bko.evaluate.experiment.RUNS` holds the whole run over
  Synthea; `e3-a` holds holdouts 0 to 49 and `e3-b` holdouts 50 to 99, each with a ceiling of
  $32.50. The two slices run concurrently at 8 calls in flight each. `merge` joins their
  `scores.csv` files in seed order into the bytes one `e3` run writes, and `compare e3` reads the
  merged file.
- **Spend**: a total ceiling of $65.00, 2 x $32.50. The estimate is about $30.76; see Rationale.
- **Duration**: about 7.3 h per slice by scaling `e1-4`'s wall time per record asked, with the two
  slices concurrent; see Rationale.
- **Holdout count**: 100, E1's.

`bko.datasets.synthea` builds the frame's records, `bko.evaluate.data` holds the dataset, the frame
and its field rows, `bko.datasets.ablation` the training-only removal, and
`bko.evaluate.experiment` the runs. The frame's identity is the `digest` of the
`data dataset=synthea frame=synthea records=<n> digest=<sha256>` line `score` logs: the sha256
of its records, which two builds in separate processes match.

## Rationale

**The population.** All ages would add 1015 patients under 40, none of them a case ([C8]: every
`<40` case cell is 0), 391 of them under 18 ([C9]), and roughly double the replica rows each BKO
arm writes per holdout (1600 training rows against 788). From 40 the cells of interest are
unchanged ([C9]: every case cell and every `>=65` cell holds the same counts at 0, 18 and 40), and
the frame keeps every case, since the youngest is 44 ([C7]).

**The fields.** The indicators are the conditions the ontology mapping already grounds, so the BKO
arms' findings name the frame's own columns. A field only cases carried would leak the label; no
mapped code does ([C7]: every row but the label's has non-cases holding it). Myocardial infarction
began after the ischemic heart disease onset in 20 of its 35 cases, so for those cases it is a
later event the snapshot records, not a precursor. It stays because the frame is
cross-sectional, as UCI's readings are, and both facts are stated wherever E3 is read. Whole-year
age matches what `subjects` computes.

**The bands.** Edge 55 leaves `<55` case cells of 4 and 14 training rows, 60 leaves `<60/f/1` at 10,
and 70 leaves `>=70/m/0` at 10 ([C8]); E2's five edges leave cells of 0, 2, 4, 8 and 10 ([C8]).
At 65 the smallest training cell is 23, so before the removal no cell is underrepresented and an
un-ablated split flags nothing; only the removal makes a gap.

**The removal.** The removed cell is the female case cell over 65 (52 training rows). At 0.9, 5
rows remain, which gives SMOTE 4 neighbours and makes the 15-row fill most of the cell. At 0.75, 13
remain and the planning dry run's duplication gain does not resolve (+0.0519, half-width 0.0770,
[C10]); at 0.95, 2 remain, SMOTE's floor of one neighbour and the size of UCI's `<55 female 1`
cell, where default TVAE refused the gap request on 2 of 100 holdouts (ADR
[`research/e1-evaluation-protocol`](e1-evaluation-protocol.md), Rationale).
Removing 0.9 of the `<65` female cases too would leave about 2 of their 23 rows, with the same
floor, and would split the headline across two subgroups. [C10] ran over all ages; the 40-and-over
frame holds the same removed cell, since every case is 44 or older.

**The reference inside the run.** A separate un-ablated run cannot pair with E3: its training
rows, and so its held-out distances, differ on every holdout, and its split flags no subgroup, so
its flagged scope is empty. Adding the removed rows back on the same holdout gives the un-ablated
training set beside every other arm.

**The held-out ratio.** The corrected variance is (1/J + n2/n1) S^2. The ablated cell's 13/5 = 2.6
against the un-ablated 0.261 widens every interval sqrt((1/100 + 2.6)/(1/100 + 0.261)) = 3.10
times at 100 holdouts. The removal changes the training rows only to create the gap; the split's
own overlap, which the correction accounts for, is the un-ablated one.

**Copies.** The population has 572 distinct records among 2000, and the frame 467 among 985
([C11]); 66.7% of the frame's test records (60.9% to 73.6% over 20 holdouts) have a training twin
by chance. Removing every test record an arm copies would strip most of the test split for any arm
that writes plausible records. E1 removes copies because paid generation could reproduce public
Cleveland records; Synthea's population is generated locally from a pinned seed and is in no
public archive, so the reason does not apply, and copies are counted for every arm instead.

**Privacy.** With at least 60.9% of test records holding an exact training twin on each holdout
[C11] reads, the test rows' 5th-percentile distance to closest record is 0, so E1's privacy bar,
a share of it, is 0 and any arm with a positive 5th percentile would clear it.

**The holdout count.** Corrected resampled tests replicate their verdicts across seed sets at about
100 runs (ADR [`research/e1-evaluation-protocol`](e1-evaluation-protocol.md), Rationale). At ratio
0.261, 30 holdouts would widen the headline interval only about 7% (t at 29 degrees of freedom
times sqrt(1/30 + 0.261), over t at 99 times sqrt(1/100 + 0.261), is 1.074), but replicability
favours 100.

**The slices.** Two concurrent slices halve the wall time, and `merge` makes their union the bytes
one run writes, so `compare e3` reads E3 as one run.

**The spend.** The basis is the judged run `e1-4` at `SETTINGS`: its call log's `end` line reads
`"asked": 77487` and `"usd": 10.509915100000024` (`runs/e1-4/calls.jsonl`; E1 in
[`docs/EVIDENCE.md`](../../EVIDENCE.md#e1-uci-heart-disease-augmentation)), so a record asked
costs 10.509915100000024 / 77487 = $1.35635e-4. Equivalently, its 300.535 input and 211.162 output
tokens per record asked at Haiku 5.5's $0.10 and $0.50 per million tokens. Per holdout each BKO arm
asks a replica of the thinned training split, 788 - 47 = 741 rows, plus the 15-row fill: 756. Over
3 arms and 100 holdouts that is 3 x 756 x 100 = 226800 records, and 226800 x $1.35635e-4 = $30.76.
`e1-4` asked 77487 records against a base quota of 77400 (ADR
[`research/record-generation-design`](record-generation-design.md), Rationale), so top-ups add
little. The basis likely overstates on the input side: a Synthea record has 9 fields against
UCI's 14, so its prompts are shorter per record asked; output need not shrink with the fields
([E3](../../EVIDENCE.md#e3-synthea-downstream-augmentation), Runs). The $65.00 ceiling leaves about twice the estimate.

**The duration.** `e1-4` ran from 08:08:12 to 13:08:18 UTC on 2026-10-28, 5.00 h for 77487 records
asked at 8 in flight. Each slice asks 113400 records at 8 in flight, so 113400 / 77487 x 5.00 h is
about 7.3 h. The local arms fit about three times UCI's training rows, which this scaling does not
count. E3's rehearsal `e3-1` (below) scored all 100 holdouts with canned replies in 14.7 min, so the
local arms add about 9 s per holdout to each slice's API wait.

**The rehearsal.** `launch e3-1 e3 --rehearse` at commit
`44aaa9e7e9877163433c698a77adefff1762b366`, from a detached worktree, wrote `runs/e3-1/rehearsal/`
from 21:20:10 to 21:34:54 EDT on 2026-10-28, one segment ending `outcome=complete exit=0`. Its
`scores.log` reads `data dataset=synthea frame=synthea records=985
digest=f9c6a23e30036a12a8647903783cac6cd47eac3f58e40d82b09b7d572898b94f`, and two further
processes building the frame print the same digest, each running [C12] at the repository root. No
arm refused: `tvae` retried 4 times and
`tvae-replica` 12 times, each recovering. The call log ends `complete` with 12900 calls and 226800
records asked, the quota above. The paired `none` minus `real` recall at scope
`flagged-subgroups` over its 100 holdouts is -0.3515 [-0.4819, -0.2212] at confidence 0.95, so the
removal hurts the headline (Reversal Conditions). The figure is [C13]'s
`none vs real augmented recall flagged-subgroups -0.351538 [-0.481875, -0.221202] n=100` line,
run over `runs/e3-1/rehearsal/scores.csv` at that commit's code as
`env -u ANTHROPIC_API_KEY PYTHONPATH=<worktree>/src .venv/bin/python - runs/e3-1/rehearsal/scores.csv`,
`<worktree>` being a detached worktree of `44aaa9e` (kept in PR #56's history, on
`feat/synthea-frame`). The rehearsal's `scores.csv` holds no `copies-kept` row, so at later code
`pairable` excludes every holdout with a copy and [C13] pairs none; there the same figure, negated,
is the `unjudged criterion=secondary scope=flagged-subgroups arm=real against=none measure=recall
mean=0.351538 low=0.221202 high=0.481875` line of
`compare e3 runs/e3-1/rehearsal/scores.csv --candidate real`, whose comparators include `none`.
sha256: scores
`2a06450bca2687a41b6e1ba58710f78a73e516d9ddbc25763f58e4af4f074129`, scores.log
`ba63681bb61d2461d705a80844052452025c48c2009cd1e65943be9296ee7419`, calls
`9cffa9e201aca7ebf4bab66652e8b18534eac02c6ad84c242c83e99e09a7d824`.

The evidence commands, each run read-only at the repository root:

[C7] Cases and mapped-code prevalence:

```
uv run --no-sync python -I - <<'EOF' 2>&1 | grep -v INFO | tail -60
from pathlib import Path
from datetime import date
import pandas as pd
from bko.datasets.synthea import load_population, subjects, ISCHEMIC_HEART_DISEASE
from bko.ontology.mapping import CODE_MAPPINGS, UNMAPPED_CODES
d = Path("data/synthea/ed43c20ad40ba5c3-s1-cs1-p2000-d20260101-massachusetts")
pop = load_population(d, snapshot_date=date(2026, 1, 1))
s = subjects(pop, codes=frozenset({ISCHEMIC_HEART_DISEASE}))
print("alive", len(s), "cases", int(s.target.sum()), "women cases", int(s[(s.sex==0)].target.sum()), "men cases", int(s[(s.sex==1)].target.sum()))
print("case age min/max", s[s.target==1].age.min(), s[s.target==1].age.max(), " all age min/max", s.age.min(), s.age.max())
snap = pd.Timestamp(pop.snapshot_date)
on = pop.conditions[pop.conditions.start <= snap]
ihd_on = on[on.code == ISCHEMIC_HEART_DISEASE].groupby("patient").start.min()
for row in (*CODE_MAPPINGS, *UNMAPPED_CODES):
    holders = set(on.loc[on.code == row.snomed, "patient"])
    h = s.index.isin(holders)
    cases_with = int((h & (s.target == 1)).sum()); non_with = int((h & (s.target == 0)).sum())
    first = on[on.code == row.snomed].groupby("patient").start.min()
    both = first.index.intersection(ihd_on.index)
    before = int((first[both] < ihd_on[both]).sum()); same = int((first[both] == ihd_on[both]).sum())
    print(f"{row.snomed:>10} {row.label[:40]:40s} cases_with={cases_with:4d} noncases_with={non_with:4d} onset_before_ihd={before} same_day={same}")
EOF
```

```
alive 2000 cases 261 women cases 94 men cases 167
case age min/max 44 110  all age min/max 0 110
 414545008 myocardial ischemia                      cases_with= 261 noncases_with=   0 onset_before_ihd=0 same_day=261
  59621000 essential hypertension                   cases_with= 136 noncases_with= 264 onset_before_ihd=113 same_day=6
  44054006 type 2 diabetes mellitus                 cases_with=  58 noncases_with=  53 onset_before_ihd=43 same_day=1
  55822004 hyperlipidemia                           cases_with=  87 noncases_with=  98 onset_before_ihd=52 same_day=5
  22298006 myocardial infarction                    cases_with=  35 noncases_with=  30 onset_before_ihd=15 same_day=0
 401303003 ST-elevation myocardial infarction       cases_with=  15 noncases_with=   9 onset_before_ihd=5 same_day=0
 230690007 stroke disorder                          cases_with=   5 noncases_with=   7 onset_before_ihd=3 same_day=0
 401314000 Acute non-ST segment elevation myocardia cases_with=  20 noncases_with=  20 onset_before_ihd=10 same_day=0
```

[C8] Cell grids, train and test sizes and held-out ratios by edges, over all ages:

```
uv run --no-sync python -I - <<'EOF' 2>&1 | grep -v INFO | tail -70
from pathlib import Path
from datetime import date
from bko.datasets.synthea import load_population, subjects, ISCHEMIC_HEART_DISEASE, SYNTHEA_AGE_BAND_EDGES
from bko.datasets.cells import cell_counts
from bko.datasets.split import held_out_count
d = Path("data/synthea/ed43c20ad40ba5c3-s1-cs1-p2000-d20260101-massachusetts")
pop = load_population(d, snapshot_date=date(2026, 1, 1))
s = subjects(pop, codes=frozenset({ISCHEMIC_HEART_DISEASE}))
for edges in (SYNTHEA_AGE_BAND_EDGES, (55,), (60,), (65,), (70,)):
    cc = cell_counts(s, edges=edges)
    cc["train"] = cc["count"] - cc["count"].map(held_out_count); cc["test"] = cc["count"].map(held_out_count)
    cc["ratio"] = (cc.test / cc.train).round(3)
    print("edges", edges, "max ratio", cc.ratio.max(), " train total", cc.train.sum(), "test total", cc.test.sum(), "test cases", cc[cc.target==1].test.sum())
    print(cc.to_string(index=False))
EOF
```

```
edges (40, 50, 60, 70, 80) max ratio 0.3  train total 1600 test total 400 test cases 52
age_band    sex  target  count  underrepresented  train  test  ratio
     <40 female       0    512             False    410   102  0.249
     <40 female       1      0              True      0     0    NaN
     <40   male       0    503             False    402   101  0.251
     <40   male       1      0              True      0     0    NaN
   40-49 female       0    121             False     97    24  0.247
   40-49 female       1      0              True      0     0    NaN
   40-49   male       0    124             False     99    25  0.253
   40-49   male       1      5              True      4     1  0.250
   50-59 female       0    146             False    117    29  0.248
   50-59 female       1     13              True     10     3  0.300
   50-59   male       0    139             False    111    28  0.252
   50-59   male       1     27             False     22     5  0.227
   60-69 female       0     84             False     67    17  0.254
   60-69 female       1     25             False     20     5  0.250
   60-69   male       0     61             False     49    12  0.245
   60-69   male       1     51             False     41    10  0.244
   70-79 female       0     24             False     19     5  0.263
   70-79 female       1     33             False     26     7  0.269
   70-79   male       0     10              True      8     2  0.250
   70-79   male       1     47             False     38     9  0.237
    >=80 female       0     13              True     10     3  0.300
    >=80 female       1     23             False     18     5  0.278
    >=80   male       0      2              True      2     0  0.000
    >=80   male       1     37             False     30     7  0.233
edges (55,) max ratio 0.254  train total 1601 test total 399 test cases 52
age_band    sex  target  count  underrepresented  train  test  ratio
     <55 female       0    707             False    566   141  0.249
     <55 female       1      5              True      4     1  0.250
     <55   male       0    702             False    562   140  0.249
     <55   male       1     17              True     14     3  0.214
    >=55 female       0    193             False    154    39  0.253
    >=55 female       1     89             False     71    18  0.254
    >=55   male       0    137             False    110    27  0.245
    >=55   male       1    150             False    120    30  0.250
edges (60,) max ratio 0.3  train total 1600 test total 400 test cases 52
age_band    sex  target  count  underrepresented  train  test  ratio
     <60 female       0    779             False    623   156  0.250
     <60 female       1     13              True     10     3  0.300
     <60   male       0    766             False    613   153  0.250
     <60   male       1     32             False     26     6  0.231
    >=60 female       0    121             False     97    24  0.247
    >=60 female       1     81             False     65    16  0.246
    >=60   male       0     73             False     58    15  0.259
    >=60   male       1    135             False    108    27  0.250
edges (65,) max ratio 0.261  train total 1600 test total 400 test cases 52
age_band    sex  target  count  underrepresented  train  test  ratio
     <65 female       0    831             False    665   166  0.250
     <65 female       1     29             False     23     6  0.261
     <65   male       0    805             False    644   161  0.250
     <65   male       1     56             False     45    11  0.244
    >=65 female       0     69             False     55    14  0.255
    >=65 female       1     65             False     52    13  0.250
    >=65   male       0     34             False     27     7  0.259
    >=65   male       1    111             False     89    22  0.247
edges (70,) max ratio 0.267  train total 1600 test total 400 test cases 53
age_band    sex  target  count  underrepresented  train  test  ratio
     <70 female       0    863             False    690   173  0.251
     <70 female       1     38             False     30     8  0.267
     <70   male       0    827             False    662   165  0.249
     <70   male       1     83             False     66    17  0.258
    >=70 female       0     37             False     30     7  0.233
    >=70 female       1     56             False     45    11  0.244
    >=70   male       0     12              True     10     2  0.200
    >=70   male       1     84             False     67    17  0.254
```

[C9] Grids for the all-ages, 18-and-over and 40-and-over frames:

```
uv run --no-sync python -I - <<'EOF' 2>&1 | grep -v INFO | tail -24
from pathlib import Path
from datetime import date
from bko.datasets.synthea import load_population, subjects, ISCHEMIC_HEART_DISEASE
from bko.datasets.cells import cell_counts
from bko.datasets.split import held_out_count
d = Path("data/synthea/ed43c20ad40ba5c3-s1-cs1-p2000-d20260101-massachusetts")
pop = load_population(d, snapshot_date=date(2026, 1, 1))
s = subjects(pop, codes=frozenset({ISCHEMIC_HEART_DISEASE}))
for lo in (0, 18, 40):
    f = s[s.age >= lo]
    for edges in ((65,), (70,)):
        cc = cell_counts(f, edges=edges); cc["train"] = cc["count"] - cc["count"].map(held_out_count); cc["test"] = cc["count"].map(held_out_count)
        cc["ratio"] = (cc.test / cc.train)
        line = " ".join(f"{r.age_band}/{r.sex[0]}/{r.target}:{r.train}+{r.test}" for _, r in cc.iterrows())
        print(f"age>={lo} edges={edges} patients={len(f)} train={cc.train.sum()} test={cc.test.sum()} max_ratio={cc.ratio.max():.3f} | {line}")
EOF
```

```
age>=0 edges=(65,) patients=2000 train=1600 test=400 max_ratio=0.261 | <65/f/0:665+166 <65/f/1:23+6 <65/m/0:644+161 <65/m/1:45+11 >=65/f/0:55+14 >=65/f/1:52+13 >=65/m/0:27+7 >=65/m/1:89+22
age>=0 edges=(70,) patients=2000 train=1600 test=400 max_ratio=0.267 | <70/f/0:690+173 <70/f/1:30+8 <70/m/0:662+165 <70/m/1:66+17 >=70/f/0:30+7 >=70/f/1:45+11 >=70/m/0:10+2 >=70/m/1:67+17
age>=18 edges=(65,) patients=1609 train=1287 test=322 max_ratio=0.261 | <65/f/0:504+126 <65/f/1:23+6 <65/m/0:492+123 <65/m/1:45+11 >=65/f/0:55+14 >=65/f/1:52+13 >=65/m/0:27+7 >=65/m/1:89+22
age>=18 edges=(70,) patients=1609 train=1288 test=321 max_ratio=0.267 | <70/f/0:530+132 <70/f/1:30+8 <70/m/0:510+127 <70/m/1:66+17 >=70/f/0:30+7 >=70/f/1:45+11 >=70/m/0:10+2 >=70/m/1:67+17
age>=40 edges=(65,) patients=985 train=788 test=197 max_ratio=0.261 | <65/f/0:255+64 <65/f/1:23+6 <65/m/0:242+60 <65/m/1:45+11 >=65/f/0:55+14 >=65/f/1:52+13 >=65/m/0:27+7 >=65/m/1:89+22
age>=40 edges=(70,) patients=985 train=788 test=197 max_ratio=0.267 | <70/f/0:281+70 <70/f/1:30+8 <70/m/0:259+65 <70/m/1:66+17 >=70/f/0:30+7 >=70/f/1:45+11 >=70/m/0:10+2 >=70/m/1:67+17
```

[C10] A planning dry run of the removed cell's recall: a stand-in logistic regression over 40
holdouts and all ages, planning evidence rather than a gate:

```
uv run --no-sync python -I - <<'EOF' 2>&1 | grep -v INFO | tail -40
import math, warnings
from pathlib import Path
from datetime import date
from fractions import Fraction
import numpy as np, pandas as pd
from scipy.stats import t as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from bko.datasets.synthea import load_population, subjects, ISCHEMIC_HEART_DISEASE
from bko.datasets.split import split
from bko.datasets.cells import cell_keys
from bko.ontology.mapping import CODE_MAPPINGS
d = Path("data/synthea/ed43c20ad40ba5c3-s1-cs1-p2000-d20260101-massachusetts")
pop = load_population(d, snapshot_date=date(2026, 1, 1))
s = subjects(pop, codes=frozenset({ISCHEMIC_HEART_DISEASE}))
on = pop.conditions[pop.conditions.start <= pd.Timestamp(pop.snapshot_date)]
feats = [m.snomed for m in CODE_MAPPINGS[1:]]
for code in feats: s[code] = s.index.isin(set(on.loc[on.code == code, "patient"])).astype(int)
EDGES = (65,)
def fit_recall(train, test, mask):
    X = lambda f: np.column_stack([(f.age - train.age.mean()) / train.age.std(), f.sex, *(f[c] for c in feats)])
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(X(train), train.target)
    p = m.predict_proba(X(test))[:, 1]
    t = test.target.to_numpy()[mask]; return recall_score(t, (p[mask] >= 0.5).astype(int)), float(((p >= 0.5).astype(int)[test.target.to_numpy() == 1]).mean())
rows = []
for seed in range(40):
    parts = split(s, seed=seed, edges=EDGES)
    tr, te = parts.train, parts.test
    kt = cell_keys(tr, edges=EDGES); ke = cell_keys(te, edges=EDGES)
    in_cell_tr = ((kt.age_band == ">=65") & (kt.sex == "female") & (kt.target == 1)).to_numpy()
    mask = ((ke.age_band == ">=65") & (ke.sex == "female") & (ke.target == 1)).to_numpy()
    sel = tr.index[in_cell_tr].sort_values()
    rec = {"seed": seed, "cell_test": int(mask.sum()), "cell_train": len(sel)}
    rec["unablated"], rec["unablated_all"] = fit_recall(tr, te, mask)
    for f in (0.75, 0.9, 0.95):
        n = math.ceil(Fraction(str(f)) * len(sel)); removed = sel[np.random.default_rng(seed).permutation(len(sel))][:n]
        ab = tr.drop(index=removed); rest = ab.loc[sel.difference(removed)]
        rec[f"abl{f}"], rec[f"abl{f}_all"] = fit_recall(ab, te, mask)
        dup = pd.concat([ab, rest.sample(20 - len(rest), replace=True, random_state=seed)])
        rec[f"dup{f}"], _ = fit_recall(dup, te, mask)
        rec[f"left{f}"] = len(rest)
    rows.append(rec)
r = pd.DataFrame(rows)
print(r[["cell_test", "cell_train", "left0.75", "left0.9", "left0.95"]].drop_duplicates().to_string(index=False))
print(r.drop(columns=["seed", "cell_test", "cell_train"]).mean().round(4).to_string())
def ci(x, ratio):
    J = len(x); hw = T.ppf(0.975, J - 1) * math.sqrt((1 / J + ratio) * np.var(x, ddof=1)); return f"{np.mean(x):+.4f} hw={hw:.4f}"
for f in (0.75, 0.9, 0.95):
    left = r[f"left{f}"].iloc[0]
    print(f"f={f}: ablated-unablated {ci(r[f'abl{f}'] - r.unablated, 13/52)} | at ablated ratio {13/left:.2f}: {ci(r[f'abl{f}'] - r.unablated, 13/left)} | dup-ablated {ci(r[f'dup{f}'] - r[f'abl{f}'], 13/52)}")
EOF
```

```
 cell_test  cell_train  left0.75  left0.9  left0.95
        13          52        13        5         2
unablated         0.6019
unablated_all     0.5548
abl0.75           0.3481
abl0.75_all       0.4798
dup0.75           0.4000
left0.75         13.0000
abl0.9            0.2923
abl0.9_all        0.4606
dup0.9            0.3923
left0.9           5.0000
abl0.95           0.2750
abl0.95_all       0.4538
dup0.95           0.3846
left0.95          2.0000
f=0.75: ablated-unablated -0.2538 hw=0.1283 | at ablated ratio 1.00: -0.2538 hw=0.2477 | dup-ablated +0.0519 hw=0.0770
f=0.9: ablated-unablated -0.3096 hw=0.1389 | at ablated ratio 2.60: -0.3096 hw=0.4291 | dup-ablated +0.1000 hw=0.0720
f=0.95: ablated-unablated -0.3269 hw=0.1290 | at ablated ratio 6.50: -0.3269 hw=0.6284 | dup-ablated +0.1096 hw=0.0976
```

[C11] Chance exact twins, and ST-elevation holders within myocardial infarction:

```
uv run --no-sync python -I - <<'EOF' 2>&1 | grep -v INFO | tail -12
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd
from bko.datasets.synthea import load_population, subjects, ISCHEMIC_HEART_DISEASE
from bko.datasets.split import split
from bko.ontology.mapping import CODE_MAPPINGS
d = Path("data/synthea/ed43c20ad40ba5c3-s1-cs1-p2000-d20260101-massachusetts")
pop = load_population(d, snapshot_date=date(2026, 1, 1))
s = subjects(pop, codes=frozenset({ISCHEMIC_HEART_DISEASE}))
on = pop.conditions[pop.conditions.start <= pd.Timestamp(pop.snapshot_date)]
feats = [m.snomed for m in CODE_MAPPINGS[1:]]
for code in feats: s[code] = s.index.isin(set(on.loc[on.code == code, "patient"])).astype(int)
cols = ["age", "sex", "target", *feats]
print("STEMI holders", int(s["401303003"].sum()), "of whom hold MI", int((s["401303003"] & s["22298006"]).sum()))
for lo in (0, 40):
    f = s[s.age >= lo]
    tw = []
    for seed in range(20):
        p = split(f, seed=seed, edges=(65,))
        keys = set(map(tuple, p.train[cols].to_numpy()))
        tw.append(np.array([tuple(r) in keys for r in p.test[cols].to_numpy()]).mean())
    print(f"age>={lo} distinct {len(f[cols].drop_duplicates())} of {len(f)}; test records with an exact training twin: mean {np.mean(tw):.3f} (min {min(tw):.3f}, max {max(tw):.3f})")
EOF
```

```
STEMI holders 24 of whom hold MI 24
age>=0 distinct 572 of 2000; test records with an exact training twin: mean 0.823 (min 0.790, max 0.865)
age>=40 distinct 467 of 985; test records with an exact training twin: mean 0.667 (min 0.609, max 0.736)
```

[C12] The frame's digest, built in a fresh process:

```
env -u ANTHROPIC_API_KEY uv run --no-sync python -I -c "from pathlib import Path; from bko.evaluate.data import load, Dataset; from bko.evaluate.context import Context; print(load(Dataset.SYNTHEA, Path('data'), Context()).digest)"
```

[C13] E3's paired readings: every arm against `real` and against `none`, in-run on the same
holdouts, over every series `bko.evaluate.beside.pair` pairs, copies counted (`allow_copies`). E3
reads it over `runs/e3-4/scores.csv`:

```
uv run --no-sync python -I - runs/e3-4/scores.csv <<'EOF'
import sys
from pathlib import Path

from bko.evaluate.beside import Skipped, pair, pairable
from bko.evaluate.criteria import held_out_ratio_of, series_of
from bko.evaluate.experiment import E3
from bko.evaluate.scores import parse

s = parse(Path(sys.argv[1]).read_text())
pg = pairable(s, s, allow_copies=True)
ratio = held_out_ratio_of(series_of(s))
for arm in E3.arms:
    for ref in ("real", "none"):
        if arm == ref:
            continue
        for b in pair(s, s, arm=arm, against=ref, holdouts=pg.holdouts, ratio=ratio, confidence=0.95):
            if not isinstance(b, Skipped):
                i = b.interval
                print(
                    f"{arm} vs {ref} {b.source.value} {b.measure.value} {b.scope.value} "
                    f"{i.mean:+.6f} [{i.low:+.6f}, {i.high:+.6f}] n={len(pg.holdouts)}"
                )
EOF
```

[C14] Each arm's mean headline recall over its holdouts, `none`, `class-weight` and `real`
included:

```
uv run --no-sync python -I - runs/e3-4/scores.csv <<'EOF'
import sys
from pathlib import Path
from statistics import mean

from bko.evaluate.scores import Measure, Scope, Source, parse

rows = [
    s
    for s in parse(Path(sys.argv[1]).read_text())
    if s.source == Source.AUGMENTED and s.measure == Measure.RECALL and s.scope == Scope.FLAGGED
]
for arm in sorted({s.arm for s in rows}):
    values = [s.value for s in rows if s.arm == arm]
    print(f"{arm} holdouts={len(values)} mean_recall_removed_cell={mean(values):.4f}")
EOF
```

## Consequences

**Gained**: a gap whose truth is fixed by construction; a reference arm paired with every other
arm; intervals at the un-ablated ratio; a frame built without a JDK.
**Accepted**: 13 removed-cell test cases per holdout; a frame whose records repeat, so copies are
counted rather than removed and the privacy rows carry no verdict; myocardial infarction read as a
co-occurring condition; an estimate scaled from UCI's prompts.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| E2's age bands | Leave training cells of 0 to 10 rows, so the removed cell is not the only gap ([C8]) | 2026-10-28 |
| All ages | 1015 more patients with no case, 391 of them under 18, and double the replica spend; see Rationale | 2026-10-28 |
| Removing test copies | Most test records have a chance training twin; see Rationale | 2026-10-28 |
| The ablated held-out ratio | Widens every interval 3.10 times; see Rationale | 2026-10-28 |
| A separate un-ablated run as the reference | Cannot pair: its held-out distances differ and it flags nothing; see Rationale | 2026-10-28 |
| Removing 0.75 of the cell | The duplication gain does not resolve ([C10]) | 2026-10-28 |
| Removing 0.95 of the cell | Leaves 2 rows, SMOTE's floor and the size of the UCI cell default TVAE refused; see Rationale | 2026-10-28 |
| Removing both female case cells | Leaves about 2 `<65` rows and splits the headline; see Rationale | 2026-10-28 |
| Fractional age | `subjects` computes whole years, and a fractional field adds decimals the prompt must hold | 2026-10-28 |
| A richer frame from Synthea's observations | Moves E2's population, which exports conditions only, and needs a JDK to regenerate | 2026-10-28 |

## Enforcement

- Owner: `src/bko/evaluate/data.py`
- Pinned by: `tests/test_datasets_synthea_frame.py`, `tests/test_evaluate_synthea.py`,
  `tests/test_evaluate_removal.py`, `tests/test_evaluate_beside.py`

## Reversal Conditions

**The bands and the removal are re-derived with [C8] and [C10] if the pinned population's
`run.json` identity changes**; the check is the population's directory key and `run.json` against
the path in Context.

**The removal is re-chosen before the paid run if the rehearsal's paired `none` minus `real`
recall at scope `flagged-subgroups` has an interval that includes 0**; the removal would then not
hurt the headline, and E3 could show no recovery.

## Related

- [The Synthea ablation: a pinned generated population, with whole patients removed](synthea-ablation-design.md)
- [The UCI evaluation frame: Cleveland records, a binary label and a cell-stratified split](uci-evaluation-frame.md)
- [E1's evaluation: repeated holdouts, one classifier, separation judged on paired intervals](e1-evaluation-protocol.md)
- [Record generation: one request per cell at a chosen guidance level, every asked record handed on](record-generation-design.md)
- [Record validation: each field's number, bound and decimals, counted per check](record-validation-design.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
