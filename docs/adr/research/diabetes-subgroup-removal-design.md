---
status: accepted
topic: research
created: 2026-11-02
updated: 2026-11-02
---
# The Diabetes 130 subgroup-removal design: the full adult frame, one thinned cell at the prevalence cutoff, every arm

## Context

UCI Heart Disease cannot separate the arms, and E3 reads the removal design on Synthea, whose
patients are generated (ADR [`research/synthea-downstream-design`](synthea-downstream-design.md)).
E6 runs the same design on real encounters: UCI Diabetes 130-US Hospitals
([`clore-2014`](../../literature.md#datasets)), pinned as `uci-diabetes-130` in
`bko.datasets.sources`, whose frame `bko.datasets.diabetes` builds and `bko.evaluate.diabetes`
describes ([`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#data), The Diabetes frame).

Every E6 number depends on the cohort, the label, the cutoff, the age bands, the removed cell, the
removal fraction, the fill target, the arms, the holdout count and the copy rule. The task unit
[`evaluate-diabetes-subgroup-removal`](../../../tasks/evaluate-diabetes-subgroup-removal.md) asks
for a power calculation to fix them before any paid run. Two free measurements did that: a probe
over a frame reimplemented from the raw archive, then a dry run over the committed frame code. Each
is a throwaway script outside the repository, whose header gives its invocation with
`VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1`. Blocks [C1] to [C5] below are excerpts of the dry
run's output, every line kept byte for byte, with lines left out: [C1] omits G0's label counts per
filter order, the drug shares beside the committed `DRUGS`, and the `change` and `diabetesMed`
shares; [C2] the tables under edges (8,) and
(4, 8); [C3] the arms' definitions, every other cell's section, the 0.5-cutoff readings, and the
AUC-PR, precision, `mix-0.5` and `class-weight` lines; [C4] the other cells' replicate rows. The
probe block in Rationale is an excerpt of the probe's output in the same way. [C6] is not from
either script: it was measured separately, by the command it shows.

The dry run's gates, in order:

1. **G1.** If the mean un-ablated `real` recall on the cell at a 0.5 cutoff lies in [0.15, 0.85],
   the cutoff is 0.5; otherwise it is the un-ablated training prevalence to 4 decimals.
2. **G2.** f is the smallest 3-decimal fraction of at least 0.900 that leaves R, the cell's kept
   training rows, in [5, 19]; every other cell holds at least 20 training rows and 33 records; and
   the flagged set is the cell alone on all 100 seeds.
3. **G3.** D, the paired `none - real` recall on the removed cell (scope `flagged-subgroups`), has
   an upper bound below 0, and |D| >= 2 x kappa x MDE(arm vs `none`) for both stand-ins.
4. **G4.** kappa x MDE(arm vs arm) <= |D| / 4.
5. **G5.** D's sign and G3 agree across seed sets 0-99, 100-199 and 200-299.
6. **Cell.** Among cells passing G2 to G5, the smallest max over stand-ins of kappa x MDE(arm vs
   `none`) / |D|.

MDE is `bko.evaluate.power.mde(S, holdouts=100, ratio=r)` for a contrast's standard deviation S
over holdouts. kappa scales a stand-in's S to a paid arm's: E3's paid S over its stand-in S, 1.15
for SMOTE and 1.42 for duplication ([C3]).

## Decision Drivers

* **Must read a difference on the removed cell**: D must lie clear of zero, and each arm's contrast
  with `none` must resolve well inside it.
* **Must leave the removed cell the only flagged one** on every holdout, so the headline reads one
  subgroup.
* **Must pair every arm, `real` included, on identical splits**, as E3 does.
* **Must keep the spend inside the unit's cap**, so the replica fill is bounded.
* **Must report compute in system-independent units**: no reading, rule or design value uses wall
  time.

## Decision

- **Frame**: `bko.datasets.diabetes.frame_records` over the pinned archive. Each patient's first
  encounter (the lowest `encounter_id`), then drop discharges expired or to hospice
  (`discharge_disposition_id` 11, 13, 14, 19, 20, 21), `Unknown/Invalid` gender and age brackets
  under 20: 69,283 records, 6,248 of them label 1 ([C1]). The fields are the age decade index
  (2 to 9), sex, race, the eight count fields, `A1Cresult`, `max_glu_serum`, the eight drugs most
  often given, `change`, `diabetesMed`, and one indicator per `ICD9_MAPPINGS` row. Every indicator
  lies above 1% prevalence (the least is 0.0154, [C1]), and `bko.gaps.shares.nested` finds no
  nesting among the field rows (`tests/test_evaluate_diabetes.py` asserts it), so the table keeps
  all ten.
- **G0, approximate**: after the first-encounter and discharge filters the cohort holds 69,973
  encounters against Strack et al.'s 69,984 ([`strack-2014`](../../literature.md#datasets)), 11
  fewer ([C1]). At that step 6,277 are readmitted in under 30 days against the 6,459 (3,462 +
  2,997, the readmitted counts of the female and male rows of Strack et al.'s Table 1), and 37,229
  are female against that table's 37,234 ([C6]). The
  plan set no tolerance, so G0 is reported, not gated.
- **Label**: 1 for readmission in under 30 days (`readmitted` `<30`), 0 otherwise.
- **Cutoff**: the un-ablated training prevalence, 0.0902 (`Experiment.cutoff`). G1 failed on all
  18 cells: real recall at 0.5 is 0.0020 to 0.0157 ([C3]).
- **Bands and cell**: edges (4, 6, 8) (`DIABETES_FRAME.edges`); the removed cell is
  `(6-7, male, 1)`, men in their sixties and seventies readmitted within 30 days.
- **Removal**: f = 0.985 of the cell's 1,269 training rows, 1,250 removed and R = 19 kept, drawn
  by `removal_order` at the holdout's seed; and a second run at f = 1.0, the zero-shot case. The
  test split is never thinned: 317 test positives in the cell per holdout ([C4]).
- **Fill**: E1's threshold of 20 flags the cell; `fill_to` is the cell's un-ablated training
  count, 1,269, so each row-adding arm restores the cell's size.
- **Held-out ratio**: 0.250450, the largest held-out-to-training ratio of any cell over the seeds,
  read before the removal ([C2]).
- **Copies**: kept and counted (`decontaminate` false), since `real`'s rows are real training rows
  and the harness refuses `real` in an experiment that removes copies. The archive is public, so
  a test copy from a Messages arm reads as memorization of the archive, not as chance.
- **Arms, in each fraction's run**: every arm: `none`, `class-weight`, `smote`, `adasyn`, `ctgan`,
  `tvae`, `gaussian-copula`, `cart`, `bayes-net`, `bko`, `bko-population`, `bko-frequency`,
  `bko-direct`, `bko-skeleton`, `cllm`, `cllm-uncurated` and `real`. At 1.0 the cell keeps no
  training row, so SMOTE and ADASYN meet a structural shortfall and fill 0 rows; they stay in the
  run and read as `none` there. The scale smoke run (holdouts 0 and 1, canned replies,
  `scratchpad/e6-smoke/summary.txt`) measured the rejection samplers on the cell at full N: at
  0.985, TVAE filled 0 of 1,250 rows and CART and the Bayesian network about 30% (412 and 430; 355
  and 329), each after five attempts of 1,000,000 draws; at 1.0, TVAE and CART filled 0 of 1,269
  and the Bayesian network 1 and 3. Each completes as a shortfall, not a refusal.
- **Candidate and comparators**: `bko` is the candidate; the comparators are every arm except
  `real` and `bko`, the 15 others.
- **Replica**: `bko` alone fills a replica, of `replica_rows` 2000 rows, in the f = 0.985 run only.
  E1's rule that every row-adding arm fills a replica of the training split's cell counts (ADR
  [`research/e1-evaluation-protocol`](e1-evaluation-protocol.md)) is E1's, not E6's.
- **Holdouts and the family**: 100 holdouts, E1's seeds 0 to 99. The contrasts read together are
  `bko` minus each comparator on two measures, recall on the removed cell (scope
  `flagged-subgroups`) and AUC-PR over the whole test split (scope `all`). The family is the
  comparators times the measures: 15 x 2 = 30 intervals, each refuting reading at confidence
  1 - (1 - c)/30, 0.99833 at E1's c = 0.95.
- **N**: the full eligible frame, every one of the 69,283 records.
- **Settings**: `SETTINGS` (Haiku 5.5) for every Messages arm.
- **Slices**: each fraction's run is K = 10 holdout slices of 10 holdouts; the f = 0.985 run's
  slices finish before the 1.0 run's start. `merge` joins a fraction's slices into one run's bytes.
  K is the memory cap's: a free smoke run of every arm on holdouts 0 and 1 at full N, one process
  per fraction under the Threads environment with canned replies (`score --rehearse`), peaked at
  2,146,516,992 bytes maximum resident set size at f = 0.985 and 2,184,560,640 at 1.0
  (`/usr/bin/time -l`). 20 such processes, 43.7 GB, fit in 100 GB, and `projection_after`, 5, is
  under the 10 holdouts per slice.
- **Replays**: one slice per fraction, plus every slice that resumed after a stop.
- **Compute cost**: reported as model fits, rows drawn, tokens and USD per run and per record
  asked. No wall time is read, ruled on or reported.
- **Threads**: every run, every replay and the smoke run behind K set
  `VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1`, so a replay's numerics match its run's and K's
  memory reading holds for the slices.
- **Egress**: each Messages call sends what [`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md#generation)
  (What leaves the machine, Diabetes 130) lists, under the archive's CC BY 4.0 license: training
  bounds, quartiles and shares of de-identified encounters through the BKO arms, and frame records
  whole through the CLLM arms. `encounter_id`, `patient_nbr` and every column the frame drops
  never leave the machine.

The frame's identity is the `digest` of the data line `score` logs ([C5]):
`38b599c950d6f7bf70620bb8e0fbe90888de56fedf55d5dbb93aa1a06e9ab0cb`, records 69283.

## Rationale

**The probe.** Over a frame reimplemented from the raw CSV, with fewer fields than the committed
frame (no `change` or `diabetesMed`), the probe found 12 cells passing G2 to G4 and chose
`(4-7, female, 1)` under edges (4, 8) at score 0.1719, with `(6-7, male, 1)` second at 0.1734:

```
GO: edges (4, 8) cell ('4-7', 'female', 1) cutoff prevalence 0.0902 f=0.990 R=18 fill_to=1870 D=-0.4388 (upper -0.4184) score=0.1719
  candidate: edges (4, 8) cell ('4-7', 'female', 1) cut prev score=0.1719 D=-0.4388
  candidate: edges (4, 6, 8) cell ('6-7', 'male', 1) cut prev score=0.1734 D=-0.4723
```

On the committed frame that cell scores 0.1846, fifth of the nine passing cells, and fails G4 on
both replicate seed sets ([C4]). The chosen cell passes G4 on all three. Its lead is small: 0.1650
against the runner-up's 0.1812, and on seeds 100-199 alone `(6-7, female, 1)` would win (0.1764
against 0.1810). The choice follows rule 5 on seeds 0-99, as fixed before the dry run.

**The cohort ([C1]).** The committed order (first encounter, then discharges) is Strack et al.'s
and lands 11 encounters from their 69,984; discharges first lands 6 over. The cutoff of 0.0902 is
the label's share of the whole frame, which the split's un-ablated training prevalence matches to
4 decimals ([C2]).

**The cutoff ([C3]).** At 0.5 the logistic regression calls almost no record positive: real recall
on every cell is under 0.016, so every D is a few thousandths and no cell passes G3. At the
prevalence cutoff D is -0.4878 on the chosen cell and every contrast resolves.

**The cell ([C2], [C3], [C4]).** Nine cells pass G2 to G5; `(6-7, male, 1)` has the smallest
score, 0.1650: D = -0.4878 [-0.5132, -0.4625]; kappa x MDE is 0.0762 for SMOTE and 0.0805 for
duplication against `none`, both under |D| / 2, and 0.0996 for SMOTE against duplication, under
|D| / 4 = 0.1220. The `<4` cells fail G3 (D near -0.09, too small to resolve), and the `4-5` cells
fail G4.

**The fill.** A 20-row fill at about 56,000 training rows is null: [C3]'s `dup20 - none` recall is
+0.0004 on the chosen cell. Restoring the cell to its un-ablated count is the dose that lets an arm
recover what the removal took.

**The replica.** A Messages replica of a 56,000-row training split per holdout per arm would cost
far more than the unit's cap; 2,000 rows for `bko` alone reads replica fidelity on the candidate
at a bounded cost.

**The family.** E1 reads a not-met verdict at 1 - (1 - c)/k over its k intervals (ADR
[`research/e1-evaluation-protocol`](e1-evaluation-protocol.md), Rationale); E6 reads the same
intersection over its own k: 15 comparators on two measures.

**The holdout count.** At ratio 0.250450, `mde` at 30 holdouts is 1.0696 times its value at 100,
and corrected resampled tests replicate their verdicts across seed sets at about 100 runs
([`bouckaert-2004`](../../literature.md#evaluation-and-failure-modes)).

**Compute.** The dry run fit 12,600 models in [C3] and 19,600 in [C4]; model fits and rows drawn
measure local compute on any machine, as tokens and USD measure the Messages arms.

[C1] Cohort, prevalences and cells:

```
[C1] Cohort
archive uci-diabetes-130 sha256=f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a rows=101,766 columns=50
excluded discharge_disposition_id (IDS_mapping.csv Expired/Hospice) = [11, 13, 14, 19, 20, 21]; committed EXCLUDED_DISCHARGES = [11, 13, 14, 19, 20, 21]
committed load_diabetes log: bko.datasets.diabetes cohort encounters=101766 repeat_encounters=30248 excluded_discharges=1545 unknown_gender=3 under_20=687 records=69283 readmitted_under_30=6248
counts per filter step
Order A (the committed frame_records order): first encounter, discharges, gender, age
  all encounters                     rows=101,766 patients= 71,518 <30=11,357 >30=35,545 NO=54,864
  first encounter per patient        rows= 71,518 patients= 71,518 <30= 6,293 >30=22,240 NO=42,985
  minus expired/hospice discharges   rows= 69,973 patients= 69,973 <30= 6,277 >30=22,222 NO=41,474
  minus Unknown/Invalid gender       rows= 69,970 patients= 69,970 <30= 6,277 >30=22,222 NO=41,471
  minus ages under 20                rows= 69,283 patients= 69,283 <30= 6,248 >30=22,056 NO=40,979
Order B: discharges, first encounter, gender, age
  minus expired/hospice discharges   rows= 99,343 patients= 69,990 <30=11,314 >30=35,502 NO=52,527
  first encounter per patient        rows= 69,990 patients= 69,990 <30= 6,285 >30=22,226 NO=41,479
  minus Unknown/Invalid gender       rows= 69,987 patients= 69,987 <30= 6,285 >30=22,226 NO=41,476
  minus ages under 20                rows= 69,300 patients= 69,300 <30= 6,256 >30=22,060 NO=40,984
order A final == committed records (69,283 rows, 6,248 label 1): True
G0 (report, no gate): Strack 2014's 69,984 encounters after first encounter + expired/hospice removal: order A 69,973 (-11), order B 69,990 (+6)
[C1] Indicator prevalences (DIABETES_INDICATORS, any of diag_1..3 in the category):
  diabetic_kidney_disease                  ICD-9 250.4  0.0154
  essential_hypertension                   ICD-9 401    0.1424
  acute_myocardial_infarction              ICD-9 410    0.0469
  heart_failure                            ICD-9 428    0.1512
  cerebral_artery_occlusion                ICD-9 434    0.0239
  pneumonia                                ICD-9 486    0.0507
  chronic_bronchitis                       ICD-9 491    0.0366
  chronic_obstructive_pulmonary_disease    ICD-9 496    0.0552
  chronic_kidney_disease                   ICD-9 585    0.0270
  chronic_ulcer_of_skin                    ICD-9 707    0.0296
[C1] HPO lab-state prevalences (LAB_MAPPINGS):
  HP:0040217 Elevated hemoglobin A1c: A1Cresult in ['>7', '>8'] = 0.1269
  HP:0003074 Hyperglycemia: max_glu_serum in ['>200', '>300'] = 0.0236
  label <30 prevalence = 0.0902 (6,248 of 69,283)
[C1] Decade x sex x label counts (age = decade index, 2 = [20-30) ... 9 = [90-100)):
         F0   F1    M0   M1
  age                      
  2     672   55   366   28
  3    1459  100  1045   88
  4    3097  250  3225  256
  5    5556  444  5914  435
  6    6970  681  7301  731
  7    8557  962  7374  855
  8    6058  737  3849  458
  9    1135  115   457   53
```

[C2] Structure under the chosen edges, seeds 0-99:

```
G2: smallest 3-decimal f >= 0.900 leaving R in [5, 19]; every other cell >= 20 train and >= 33 total;
    flagged set == (cell,) on all 100 seeds at f (and, for reference, with the whole cell removed)
cell                    | train test | R    f     | other cells min train / min total | flagged==cell @f @1.0 | G2
edges (4, 6, 8): largest held-out ratio over seeds = 0.250450; un-ablated training prevalence (4 dp) = [0.0902]; smallest cell overall: ('<4', 'male', 1) train=93
  (4, 6, 8)     <4 female 1|   124   31 |   12 0.900 |    93 /   116 ok          | True  True          | PASS
  (4, 6, 8)     <4 male   1|    93   23 |    9 0.900 |   124 /   155 ok          | True  True          | PASS
  (4, 6, 8)    4-5 female 1|   555  139 |   19 0.964 |    93 /   116 ok          | True  True          | PASS
  (4, 6, 8)    4-5 male   1|   553  138 |   19 0.964 |    93 /   116 ok          | True  True          | PASS
  (4, 6, 8)    6-7 female 1|  1314  329 |   19 0.985 |    93 /   116 ok          | True  True          | PASS
  (4, 6, 8)    6-7 male   1|  1269  317 |   19 0.985 |    93 /   116 ok          | True  True          | PASS
  (4, 6, 8)    >=8 female 1|   682  170 |   19 0.971 |    93 /   116 ok          | True  True          | PASS
  (4, 6, 8)    >=8 male   1|   409  102 |   19 0.952 |    93 /   116 ok          | True  True          | PASS
```

[C3] Power at the full frame, seeds 0-99, the chosen cell at the prevalence cutoff, then the
verdict over all 18 cells:

```
  kappa (E3 paid S over E3 stand-in S): smote 1.15 = 0.0961 / 0.0834 (= 1.1523); duplication 1.42 = 0.0961 / 0.0679 (= 1.4153); arm-vs-arm (smote - dupfull) read at kappa 1.42, the larger

model fits: 12,600 (LR, bko.evaluate.classifier.build, one per arm per seed per cell)

=== edges (4, 6, 8) cell ('6-7', 'male', 1) f=0.985 R=19 fill_to=1269 test positives=[317] ratio r=0.250450 seeds with flagged!=cell: 0
    rows added per arm: none=[0], real=[1250], class-weight=[0], smote=[1250], dupfull=[1250], dup20=[1], mix-0.5=[1250]
  cutoff 0.0902:
    mean recall    flagged-subgroups none=0.0686 real=0.5565 class-weight=0.9999 smote=0.4264 dupfull=0.4757 dup20=0.0691 mix-0.5=0.5284
    mean recall    all               none=0.3084 real=0.5299 class-weight=1.0000 smote=0.4548 dupfull=0.4720 dup20=0.3086 mix-0.5=0.5075
    D = none - real (recall, flagged) = -0.4878  95% corrected interval [-0.5132, -0.4625]  S=0.0250
    smote - none: mean=0.3578 S=0.0459 MDE=0.0662 kappa*MDE=0.0762 2*kappa*MDE=0.1523 kappa*MDE/|D|=0.1561
    dupfull - none: mean=0.4071 S=0.0392 MDE=0.0567 kappa*MDE=0.0805 2*kappa*MDE=0.1609 kappa*MDE/|D|=0.1650
    smote - dupfull: mean=-0.0493 S=0.0486 MDE=0.0701 kappa*MDE=0.0996 |D|/4=0.1220
    dup20 - none: mean=0.0004 S=0.0022 MDE=0.0032
    G3 per stand-in: {'smote': True, 'dupfull': True} -> PASS; G4: PASS; rule-5 score max(kappa*MDE)/|D| = 0.1650

[C3] Verdict at seeds 0-99
G1: cells keeping 0.5: 0 of 18; mean real recall at 0.5 ranges 0.0020..0.0157 -> cutoff prevalence 0.0902 (Phase 20 runs)
edges      cell                 cut   f     R  fill_to  r        D       [low, high]          kMDE smote kMDE dupfull kMDE(s-d)  G3    G4    score
(8,)       ('<8', 'female', 1)  prev  0.990 19    1994  0.250261 -0.4522 [-0.4730, -0.4314]  0.0956     0.0802       0.1172     PASS  fail  0.2113
(8,)       ('<8', 'male', 1)    prev  0.990 19    1914  0.250261 -0.4648 [-0.4864, -0.4431]  0.1064     0.0796       0.1315     PASS  fail  0.2290
(8,)       ('>=8', 'female', 1) prev  0.971 19     682  0.250261 -0.5583 [-0.5941, -0.5225]  0.1048     0.1018       0.1003     PASS  PASS  0.1878
(8,)       ('>=8', 'male', 1)   prev  0.952 19     409  0.250261 -0.4159 [-0.4615, -0.3703]  0.0859     0.0991       0.0897     PASS  PASS  0.2382
(4, 8)     ('<4', 'female', 1)  prev  0.900 12     124  0.250087 -0.0813 [-0.1273, -0.0352]  0.0829     0.0965       0.0500     fail  fail  1.1868
(4, 8)     ('<4', 'male', 1)    prev  0.900  9      93  0.250087 -0.0987 [-0.1586, -0.0388]  0.1020     0.1238       0.0714     fail  fail  1.2546
(4, 8)     ('4-7', 'female', 1) prev  0.990 18    1870  0.250087 -0.4531 [-0.4733, -0.4329]  0.0836     0.0704       0.1119     PASS  PASS  0.1846
(4, 8)     ('4-7', 'male', 1)   prev  0.990 18    1822  0.250087 -0.4712 [-0.4919, -0.4505]  0.0914     0.0826       0.1207     PASS  fail  0.1939
(4, 8)     ('>=8', 'female', 1) prev  0.971 19     682  0.250087 -0.5579 [-0.5921, -0.5236]  0.1011     0.0975       0.1002     PASS  PASS  0.1812
(4, 8)     ('>=8', 'male', 1)   prev  0.952 19     409  0.250087 -0.4146 [-0.4604, -0.3688]  0.0824     0.1010       0.0930     PASS  PASS  0.2435
(4, 6, 8)  ('<4', 'female', 1)  prev  0.900 12     124  0.250450 -0.0826 [-0.1302, -0.0349]  0.0781     0.0979       0.0482     fail  fail  1.1852
(4, 6, 8)  ('<4', 'male', 1)    prev  0.900  9      93  0.250450 -0.1030 [-0.1626, -0.0435]  0.0931     0.1242       0.0722     fail  fail  1.2054
(4, 6, 8)  ('4-5', 'female', 1) prev  0.964 19     555  0.250450 -0.2321 [-0.2631, -0.2011]  0.0636     0.0711       0.0818     PASS  fail  0.3062
(4, 6, 8)  ('4-5', 'male', 1)   prev  0.964 19     553  0.250450 -0.2149 [-0.2452, -0.1846]  0.0649     0.0649       0.0717     PASS  fail  0.3020
(4, 6, 8)  ('6-7', 'female', 1) prev  0.985 19    1314  0.250450 -0.4632 [-0.4870, -0.4394]  0.0778     0.0851       0.0996     PASS  PASS  0.1838
(4, 6, 8)  ('6-7', 'male', 1)   prev  0.985 19    1269  0.250450 -0.4878 [-0.5132, -0.4625]  0.0762     0.0805       0.0996     PASS  PASS  0.1650
(4, 6, 8)  ('>=8', 'female', 1) prev  0.971 19     682  0.250450 -0.5586 [-0.5919, -0.5253]  0.1030     0.0961       0.1006     PASS  PASS  0.1844
(4, 6, 8)  ('>=8', 'male', 1)   prev  0.952 19     409  0.250450 -0.4136 [-0.4609, -0.3663]  0.0826     0.0977       0.0916     PASS  PASS  0.2361
```

[C4] Replicability for the chosen cell, the probe's cell and the seeds-100-199 leader, then the
verdict:

```
[C4] Replicability: seeds 100-199 and 200-299, G3-passing cells, at each cell's G1 cutoff
model fits: 19,600
edges      cell                 set      D       [low, high]          kMDE smote kMDE dupfull kMDE(s-d)  G3    G4    score
(4, 8)     ('4-7', 'female', 1) 0-99     -0.4531 [-0.4733, -0.4329]  0.0836     0.0704       0.1119     PASS  PASS  0.1846
(4, 8)     ('4-7', 'female', 1) 100-199  -0.4510 [-0.4723, -0.4298]  0.0890     0.0701       0.1189     PASS  fail  0.1973
(4, 8)     ('4-7', 'female', 1) 200-299  -0.4529 [-0.4767, -0.4292]  0.0883     0.0750       0.1142     PASS  fail  0.1949
  G5 (D sign and G3 agree across the three sets): PASS
(4, 6, 8)  ('6-7', 'female', 1) 0-99     -0.4632 [-0.4870, -0.4394]  0.0778     0.0851       0.0996     PASS  PASS  0.1838
(4, 6, 8)  ('6-7', 'female', 1) 100-199  -0.4613 [-0.4863, -0.4363]  0.0726     0.0814       0.0939     PASS  PASS  0.1764
(4, 6, 8)  ('6-7', 'female', 1) 200-299  -0.4613 [-0.4905, -0.4321]  0.0685     0.0759       0.0827     PASS  PASS  0.1646
  G5 (D sign and G3 agree across the three sets): PASS
(4, 6, 8)  ('6-7', 'male', 1)   0-99     -0.4878 [-0.5132, -0.4625]  0.0762     0.0805       0.0996     PASS  PASS  0.1650
(4, 6, 8)  ('6-7', 'male', 1)   100-199  -0.4844 [-0.5098, -0.4590]  0.0826     0.0877       0.1134     PASS  PASS  0.1810
(4, 6, 8)  ('6-7', 'male', 1)   200-299  -0.4879 [-0.5140, -0.4617]  0.0816     0.0779       0.1059     PASS  PASS  0.1673
  G5 (D sign and G3 agree across the three sets): PASS

[C4] Verdict
cells passing G2+G3+G4+G5 at the G1 cutoff: 9
CHOSEN (rule 5): edges (4, 6, 8) cell ('6-7', 'male', 1) cutoff prevalence 0.0902 f=0.985 R=19 fill_to=1269 held-out ratio r=0.250450 test positives=317
  D=-0.4878 [-0.5132, -0.4625]; smote - none MDE=0.0662 kappa*MDE=0.0762; dupfull - none MDE=0.0567 kappa*MDE=0.0805; smote - dupfull MDE=0.0701 kappa*MDE=0.0996; score=0.1650
  candidate: edges (4, 6, 8) cell ('6-7', 'male', 1) cut prev score=0.1650 D=-0.4878
  candidate: edges (4, 8) cell ('>=8', 'female', 1) cut prev score=0.1812 D=-0.5579
  candidate: edges (4, 6, 8) cell ('6-7', 'female', 1) cut prev score=0.1838 D=-0.4632
  candidate: edges (4, 6, 8) cell ('>=8', 'female', 1) cut prev score=0.1844 D=-0.5586
  candidate: edges (4, 8) cell ('4-7', 'female', 1) cut prev score=0.1846 D=-0.4531
  candidate: edges (8,) cell ('>=8', 'female', 1) cut prev score=0.1878 D=-0.5583
  candidate: edges (4, 6, 8) cell ('>=8', 'male', 1) cut prev score=0.2361 D=-0.4136
  candidate: edges (8,) cell ('>=8', 'male', 1) cut prev score=0.2382 D=-0.4159
  candidate: edges (4, 8) cell ('>=8', 'male', 1) cut prev score=0.2435 D=-0.4146
harness parity (score_holdout, seed 0, edges (4, 6, 8), cell ('6-7', 'male', 1), cutoff 0.0902): identical on recall/precision/auc-pr x both scopes x none/class-weight/smote/real
```

[C5] The frame's data line:

```
[C5] load(Dataset.DIABETES, data/, Context()) data line
bko.evaluate data dataset=diabetes frame=diabetes records=69283 digest=38b599c950d6f7bf70620bb8e0fbe90888de56fedf55d5dbb93aa1a06e9ab0cb agreement=none
```

[C6] Gender and label at Strack et al.'s step, read-only at the repository root:

```
uv run --no-sync python - </dev/null <<'EOF'
import zipfile, io, pandas as pd, glob
p = glob.glob("data/**/diabetes-130*.zip", recursive=True)[0]
z = zipfile.ZipFile(p)
m = [n for n in z.namelist() if n.endswith("diabetic_data.csv")][0]
d = pd.read_csv(io.BytesIO(z.read(m)), dtype=str, keep_default_na=False)
d["e"] = d.encounter_id.astype("int64")
d = d.sort_values("e", kind="stable").drop_duplicates("patient_nbr")
d = d[~d.discharge_disposition_id.isin(["11","13","14","19","20","21"])]
print(p, len(d), d.gender.value_counts().to_dict(), d.readmitted.value_counts().to_dict())
EOF
```

```
data/uci-diabetes-130/f82ac129da2ddd22/diabetes-130-us-hospitals-for-years-1999-2008.zip 69973 {'Female': 37229, 'Male': 32741, 'Unknown/Invalid': 3} {'NO': 41474, '>30': 22222, '<30': 6277}
```

## Consequences

**Gained**: a removal on real encounters with 317 removed-cell test positives per holdout; a
removal that cuts the cell's recall from 0.5565 to 0.0686 ([C3]), a deficit every stand-in
contrast resolves inside; every arm
read on one split with `real` beside it; compute stated in units any machine reproduces.
**Accepted**: G0 is approximate (11 encounters from Strack et al.'s cohort, 182 fewer readmitted
within 30 days); the chosen cell leads the runner-up narrowly and loses to `(6-7, female, 1)` on
one replicate seed set; at the prevalence cutoff `class-weight` calls nearly every record positive
(recall 0.9999 on the cell and 1.0000 overall in [C3]), so its recall carries no ranking and no
gate reads it; kappa is carried over from E3's Synthea runs; at 1.0 SMOTE, ADASYN, TVAE and CART add nothing, and at 0.985 TVAE adds nothing and CART and
the Bayesian network fill about 30% of the cell, so those arms read close to `none`; a capped
replica asks `split.kept_count` per cell, so its total can exceed `replica_rows` by rounding (2,001
for 2,000 in the smoke run);
the replica reads `bko` alone.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| An any-readmission label (`<30` or `>30`) | Blurs the early readmission Strack et al. study with any later one; `<30` is the outcome the archive is used for | 2026-11-02 |
| A 20-row fill | Null at about 56,000 training rows: `dup20 - none` recall +0.0004 on the chosen cell ([C3]) | 2026-11-02 |
| Year midpoints for age | Off-grid ages that no real record holds would break exact-copy counts; the decade index is the archive's own resolution | 2026-11-02 |
| Messages replicas for every arm | A full replica of the training split per holdout per Messages arm costs far past the unit's cap; see Rationale, The replica | 2026-11-02 |
| Fewer than 100 holdouts | User decision: 100, for replicability; `mde` at 30 is 1.0696 times its value at 100 | 2026-11-02 |
| Excluding SMOTE and ADASYN at 1.0 | Their zero fill is the result at 1.0: count-based methods have nothing to copy | 2026-11-02 |
| A cross-fraction beside reading | The held-out distance to closest record differs between fractions, since the training rows differ, so the fractions do not pair | 2026-11-02 |
| Replaying every slice | User decision: one slice per fraction verifies the code, and every resumed slice its resumption | 2026-11-02 |
| A wall-time duration rule | User decision: no reading, rule or design value uses wall time | 2026-11-02 |
| The probe's cell `(4-7, female, 1)` under edges (4, 8) | On the committed frame it scores fifth of nine and fails G4 on both replicate seed sets ([C4]) | 2026-11-02 |
| A 0.5 cutoff | G1 failed on all 18 cells: real recall at 0.5 is 0.0020 to 0.0157 ([C3]) | 2026-11-02 |

## Enforcement

- Owner: `src/bko/evaluate/diabetes.py`
- Pinned by: `tests/test_datasets_diabetes.py`, `tests/test_datasets_diabetes_pinned.py` (with
  `--releases`), `tests/test_evaluate_diabetes.py`, `tests/test_evaluate_power.py`

## Reversal Conditions

- The paid f = 0.985 run's `none - real` recall on the removed cell fails G3: its corrected
  interval's upper bound is not below 0, or |D| is under 0.1609, the dry run's largest
  2 x kappa x MDE against `none` ([C3]).
- The measured cost per record asked exceeds 1.25 times the estimate E6's Configuration in
  [`docs/EVIDENCE.md`](../../EVIDENCE.md#e6-diabetes-130-subgroup-removal) records before the paid
  run.

## Related

- [The Synthea downstream design: a 40-and-over frame, one thinned cell and an in-run reference](synthea-downstream-design.md)
- [E1's evaluation: repeated holdouts, one judged classifier, separation judged on paired intervals](e1-evaluation-protocol.md)
- [The baselines: each method fills a given quota of added rows, stratum by stratum](baseline-augmentation-design.md)
- [The CLLM arms: training records shown per cell, a boosting curator's drop rule, and an egress guard on every Messages arm](cllm-arm-design.md)
- [The ontology mapping: pinned OBO releases read by pronto, and a typed table of sourced rows](ontology-mapping-design.md)
