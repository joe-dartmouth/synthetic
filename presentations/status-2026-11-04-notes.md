# BKO project status, 2026-11-04 (morning)

ENGG 119: Bayesian Knowledge-driven Ontologies (BKO) for synthetic clinical data.
Question: can medical knowledge, held as a BKO, find coverage gaps in a clinical dataset and generate synthetic records that improve prediction for underrepresented groups?

## 1. Headline

- **Synthea (E3), a removed subgroup:** BKO raises recall on the removed group from 0.262 to 0.400 (+0.138 [+0.038, +0.237], 100 holdouts). That is level with the best generators (TVAE 0.403, CTGAN 0.393), ahead of SMOTE and ADASYN (0.372, 0.380), and about 40% of the way to the true rows (0.614). Under the HGB classifier BKO leads every method on that group (0.311 against CTGAN 0.251).
- **UCI Heart Disease (E1, E4, E5), no removed subgroup:** BKO is roughly neutral against no augmentation, and trails CTGAN, TVAE, the Gaussian copula and the Bayesian network on the flagged subgroups.
- **UCI Diabetes 130 (E6), 100% removal, final (100 holdouts):** BKO is the best method. Records drawn straight from the BKB (`bko-direct`, no language model) raise recall on the removed group from 0.066 to 0.538, against 0.556 when the true rows are put back. That beats CTGAN (0.478, +0.059 [+0.011, +0.108]), the Gaussian copula (0.351) and CLLM (0.282), with no AUC-PR cost on the group, and it leads under HGB too (0.359 against CTGAN 0.274). SMOTE, ADASYN, TVAE, CART and the Bayesian network add nothing for a group with no examples. The language-model BKO arm (0.308) beats CLLM but trails CTGAN.
- **E6 at 98.5% removal (19 true rows kept), the main reading:** still running, due about 20:00-22:30 EST tonight.

### Why BKO is worth choosing at equal accuracy

Where BKO's accuracy ties the best generators, it still offers things they cannot. The claims below are tagged by how firmly the evidence supports them.

1. **It works where there is no data.** (Supported: E6 at 100% removal, final.)
   - SMOTE, ADASYN, CART and TVAE need rows of the group; the Bayesian network adds 1-3 rows. With the group fully removed, they add essentially nothing.
   - With no true rows at all, BKB-drawn records (`bko-direct`) recover 0.538 removed-group recall against 0.556 for the true rows, the best of every method and ahead of CTGAN. It needs no rows of the group itself: it combines the BKB's concept rates with real field values from patients who share the outcome.
   - Real gaps often look like this: outcomes recorded at another hospital, lost follow-up, suppressed small cells, a new population. A method that needs examples cannot fill a gap it has no examples of.

2. **No patient record leaves the institution.** (Supported by design.)
   - BKO's prompts carry aggregates and ontology statements, never an individual record. CLLM, the other language-model method, sends 20 real records per call.
   - The pipeline can also run fully on-device: `bko-direct` uses no language model, and G6's local models run on-device.
   - Knowledge travels between sites while data stays put. A hospital can fill its gaps using public medical knowledge and its own aggregates, with no data-sharing agreement.
   - This is not *complete* privacy and should not be claimed as such. The BKB is built from the data's own frequencies, aggregates of a very small group can identify people, and there is no formal differential-privacy guarantee. The defensible claim: no record-level disclosure, plus a measured distance-to-closest-record check (E1's privacy criterion was met).

3. **It finds the gaps, not just fills them.** (Supported, E2.)
   - The other methods add rows wherever they are told to. BKO first identifies which groups lack which medically expected patterns. In E2 its gap definitions recovered the gaps of the removed group.
   - That gives a data team an audit they can act on, for example by collecting more data, even if no synthetic rows are used.

4. **Every row is explainable and auditable.** (Supported by design.)
   - Each generated row traces to a gap, the ontology statements behind it, and a logged prompt and reply. Every run replays byte for byte from its call log.
   - A GAN's or copula's row cannot be explained this way. For clinical and regulatory review, it matters to show why a synthetic patient looks as it does.

5. **Generated rows are checked against medical constraints.** (Supported, E1 validity met.)
   - Every record passes consistency checks and range limits before use, so implausible patients are dropped rather than learned from.

6. **It improves as knowledge improves.** (By design, not yet measured.)
   - MONDO and HPO releases are pinned and can be updated. New medical knowledge improves generation without retraining a generator on more patient data.

7. **It is cheap and needs no generator training.** (Supported.)
   - About $0.000136 per record asked with Haiku 5.5, and $0 with `bko-direct` or a local model. No deep generator has to be trained and tuned per dataset.

**What would weaken this case:**
- **CLLM matching BKO at 100% removal:** tested, and it did not happen. Both BKO arms beat CLLM (`bko` +0.026 [+0.005, +0.046], `bko-direct` +0.256 [+0.225, +0.286]). The open caveat is the reverse: BKO's best arm uses no language model, and the language-model arm trails CTGAN.
- **HGB gains are small for the language-model arm:** `bko` adds only +0.04 removed-group recall under HGB. `bko-direct` adds 0.359 and leads every method under HGB.
- **The AUC-PR cost:** BKO lowers AUC-PR on the removed group (about -0.03), like every method that adds rows.
- **UCI was neutral:** when the data already covers a group, BKO adds little. Its value is specific to coverage gaps.

## 2. Completed results since the last update

| Entry | Result |
|---|---|
| G6, generator model | Sonnet 5.5's BKO rows score higher replica quality than Haiku 5.5's (+0.066). The local models score lower: Gemma -0.072, Nemotron -0.145, Olmo -0.162 (shared test split, no copies). Cost per record asked: Sonnet $0.00154, Haiku $0.000136; local models 1.7e12 to 3.7e12 FLOPs. Every local run replays byte for byte. |
| G6, thinking cost | With thinking on, Gemma generated 584 tokens per record (480 of them reasoning), about 3x the FLOPs. Nemotron hit the 16k output cap on 197 of 349 calls mid-reasoning; counting re-asks, its compute for 5 holdouts was about 14x thinking-off. |
| G7, BKB likelihood curation | On UCI, rows the BKB scores as likelier were not measurably more faithful (the interval holds 0), so a curated BKO arm was not run. |
| Diabetes 130 frame | 69,283 first encounters of adults. Label: readmission within 30 days (9.0%). Ten ICD-9 diagnosis indicators and two lab states are mapped to MONDO/HPO and checked against the pinned releases. |
| E6 design | A 300-seed power dry run chose the target group: men aged 60-79 who were readmitted. Scoring uses the training prevalence (0.0902) as the cutoff, because 0.5 gives zero recall on any group. |
| CLLM curation on Diabetes | Paid pilot ($15.00): the curated CLLM arm kept 0 of 46,360 generated records. Its curator rejects every record with the rare label. It was dropped from E6, saving about $1,700. Uncurated CLLM stays. |

Earlier results, unchanged:
- E1: UCI baseline comparison.
- E2: Synthea gap detection. The per-subgroup and frequency-aware gap definitions recover the removed group's gaps.
- E3: as in the headline.
- E4: larger fill targets raise recall and lower AUC-PR for every method that adds rows.
- E5: training-set size. BKO minus no augmentation is near zero at every size.
- B1-B3: classical, Bayesian-network and CLLM baselines.

## 3. In progress: E6, removal test on Diabetes 130

### Setup

- **Target group:** men aged 60-79 who were readmitted within 30 days.
- **Two conditions:**
  - **98.5% removal:** 19 true rows stay in training; the main reading.
  - **100% removal:** no true rows stay; the stress test.
- **Fill:** each method fills the group back to 1,269 rows. The true-rows arm (`real`) puts the removed rows back, which gives the ceiling.
- **Arms:** 16 in all. They include no augmentation (`none`), class weighting, SMOTE, ADASYN, CART, the Gaussian copula, the Bayesian network, CTGAN, TVAE, uncurated CLLM, BKO and four BKO ablations, plus `real`.
- **Classifiers:** logistic regression (LR, the criteria model) and histogram gradient boosting (HGB).
- **Run:** 100 holdouts per condition, in 10 slices of 10, on Haiku 5.5 with a 32k output budget. Both conditions run at once.

### Progress (06:35 EST Nov 4)

| Condition | Holdouts finished | Spend | Status |
|---|---|---|---|
| 100% removal | 100 of 100 | $198.80 | **Complete** (all 10 slices exited cleanly, 01:00-01:23 EST) |
| 98.5% removal | about 60 of 100 | $164.96 so far | Running; expected about 20:00-22:30 EST Nov 4 (faster now that the 100% slices freed their cores) |

- **Spend:** about $364 so far. The projected total is about $475, well under the $1,500 cap.
- **Health:** no stops or failures. The machine is on AC power.
- **Why it is slow:** a 98.5% holdout took about 4 hours while 20 processes shared 18 cores. The biggest steps are the Bayesian network's row-by-row draw (82 min), TVAE and CTGAN (about 37 min each), and the API arms (about 66 min).

### Final reading, 100% removal (100 holdouts, all methods)

Means over 100 holdouts. "Removed group" is men aged 60-79 who were readmitted; LR is the criteria model.

| Method | Recall, removed group (LR) | Recall, all (LR) | AUC-PR, removed group (LR) | AUC-PR, all (LR) | Recall, removed group (HGB) |
|---|---|---|---|---|---|
| `real` (ceiling) | 0.556 | 0.530 | 0.156 | 0.141 | 0.585 |
| **`bko-direct`** | **0.538** | **0.526** | **0.152** | **0.138** | **0.359** |
| CTGAN | 0.478 | 0.477 | 0.136 | 0.125 | 0.274 |
| Gaussian copula | 0.351 | 0.378 | 0.144 | 0.129 | 0.045 |
| `bko-population` | 0.343 | 0.400 | 0.122 | 0.114 | 0.054 |
| `bko-frequency` | 0.343 | 0.400 | 0.122 | 0.114 | 0.053 |
| `bko` | 0.308 | 0.380 | 0.125 | 0.116 | 0.045 |
| CLLM (uncurated) | 0.282 | 0.360 | 0.131 | 0.120 | 0.087 |
| `bko-skeleton` | 0.260 | 0.351 | 0.125 | 0.114 | 0.030 |
| `none`, SMOTE, ADASYN, TVAE, CART, Bayesian network | 0.066 | 0.307 | 0.153 | 0.127 | 0.000 |
| class weighting | 1.000* | 1.000* | 0.154 | 0.127 | 0.003 |

\* Class weighting moves every predicted probability above the 0.0902 cutoff, so it labels everyone readmitted. Its recall of 1.0 comes with no precision; compare it on AUC-PR only.

**`bko-direct` (records drawn from the BKB, no language model) against each rival, paired over 100 holdouts, 95% intervals:**

| Against | Recall, removed group | Recall, all | AUC-PR, removed group | AUC-PR, all |
|---|---|---|---|---|
| `none` | +0.472 [+0.446, +0.497] | +0.220 [+0.208, +0.231] | -0.002 [-0.004, +0.001] | +0.011 [+0.009, +0.014] |
| CTGAN | +0.059 [+0.011, +0.108] | +0.050 [+0.019, +0.081] | +0.015 [+0.006, +0.024] | +0.012 [+0.007, +0.018] |
| Gaussian copula | +0.187 [+0.131, +0.243] | +0.148 [+0.114, +0.182] | +0.008 [-0.002, +0.018] | +0.009 [+0.003, +0.015] |
| CLLM (uncurated) | +0.256 [+0.225, +0.286] | +0.167 [+0.152, +0.182] | +0.020 [+0.012, +0.029] | +0.018 [+0.014, +0.022] |

**`bko` (language-model arm) against each rival, removed-group recall:**
- `none`: +0.242 [+0.217, +0.266]
- CLLM: +0.026 [+0.005, +0.046]
- Gaussian copula: -0.043 [-0.104, +0.018]
- CTGAN: -0.171 [-0.215, -0.127]

**What it shows:**
1. **The BKB alone is the best method at 100% removal.** `bko-direct` recovers 97% of the true rows' recall on the removed group (0.538 against 0.556). It beats `none`, CTGAN, the Gaussian copula, CLLM and `bko` on recall at both scopes and on AUC-PR over all patients, every paired interval above zero. The other methods have lower means and were not paired against it. Unlike every other row-adding method, it costs no AUC-PR on the removed group: the interval holds 0.
2. **It also works under HGB:** 0.359 removed-group recall, against 0.274 for CTGAN and under 0.09 for everything else except `real`.
3. **Knowledge beats examples.** Both BKO arms beat CLLM, the same language model shown real records, so the ontology adds value even when a language model writes the rows.
4. **The language model hurts here.** The language-model BKO arm (`bko`) trails `bko-direct` by 0.230 [0.199, 0.261] on removed-group recall, and trails CTGAN. Prescribing BKB concepts in the prompt (`bko-skeleton`) did not close that gap.
5. **Methods that copy rows fail completely.** SMOTE, ADASYN, TVAE, CART and the Bayesian network add no usable rows to a group with no examples, and score exactly as no augmentation does.
6. **Validity and copies:** `bko-direct` produced 126,900 rows and all of them passed validation; the language-model arms passed 98.7-99.7%. No arm's added rows copied a test record, except 1 row from `bko-frequency`.

The 98.5% reading (19 true rows kept), the main E6 reading, lands tonight. The interim reading below is superseded at 100% removal.

### Interim reading, BKO only

The interim covers the holdouts finished at reading time: 10 at 98.5% removal and 36 at 100%. It scores `none`, `bko` and `real` only, makes no API calls, and checks each holdout against the run's own records. Each cell is a mean, or a paired difference with its 95% interval (Nadeau-Bengio corrected).

**Recall on the removed group**

| Condition | Classifier | none | bko | real | bko - none | bko - real |
|---|---|---|---|---|---|---|
| 98.5% | LR | 0.070 | 0.330 | 0.568 | +0.260 [+0.234, +0.285] | -0.238 [-0.277, -0.199] |
| 100% | LR | 0.066 | 0.305 | 0.563 | +0.239 [+0.217, +0.260] | -0.258 [-0.285, -0.231] |
| 98.5% | HGB | 0.001 | 0.047 | 0.595 | +0.047 [+0.029, +0.065] | -0.547 [-0.573, -0.521] |
| 100% | HGB | 0.000 | 0.044 | 0.588 | +0.044 [+0.030, +0.057] | -0.544 [-0.571, -0.517] |

**Recall on all patients:**
- LR: bko - none is +0.084 [+0.063, +0.105] at 98.5% and +0.070 [+0.056, +0.084] at 100%.
- HGB: bko - none is null at both (-0.004 and +0.003, both intervals holding 0).

**AUC-PR, bko - none:**
- LR, removed group: -0.033 at 98.5% and -0.029 at 100%. All patients: -0.014 and -0.012. Every interval excludes 0.
- HGB, removed group: -0.021 and -0.019. All patients: null.

**What it shows:**
1. **LR recall:** BKO beats no augmentation on the removed group in every holdout at both conditions. It closes about half the gap to the true rows, more than on Synthea (about 40%).
2. **The cost:** BKO lowers AUC-PR. It trades ranking quality for recall at the fixed cutoff, as every method that adds rows did in E4.
3. **HGB:** BKO's rows help HGB only slightly (+0.04). HGB still misses about 95% of the group's positives, while the true rows bring it to about 0.59. BKO's rows teach LR's smooth boundary but do not match the true rows' feature patterns closely enough for HGB's splits. On Synthea, BKO led under HGB; on Diabetes so far it does not.
4. **The 100% condition does not hurt BKO:** its recall is nearly the same with 19 true rows (0.330) as with none (0.305). That fits the claim that BKO draws on knowledge rather than on the rows it is shown.

### First evidence on the other methods: rows each could add

The run logs show how many rows each method added to the removed group in the slices read so far. The target is 1,250 at 98.5% (1,269 minus the 19 kept) and 1,269 at 100%. No scores yet.

| Method | 98.5% removal | 100% removal |
|---|---|---|
| BKO and its 4 ablations | full | full |
| CTGAN, Gaussian copula, uncurated CLLM | full | full |
| SMOTE, ADASYN | full (built from the 19 kept rows) | **0** |
| CART | about 420 | **0** |
| Bayesian network | about 340 | **1-3** |
| TVAE | **0** | **0** |
| class weighting, none | 0 by design | 0 by design |

At 100% removal, SMOTE, ADASYN, CART, TVAE and the Bayesian network can produce essentially no rows for a group they never saw. At that level they score close to no augmentation. Only BKO, CTGAN, the Gaussian copula and CLLM can fill the gap.

## 4. Theories and expectations for the full E6 reading

These are forecasts, not results.

### What the 100% condition models

It models a coverage gap, not bad data. The training records are accurate, but one combination (older men who were readmitted) never appears, while the group and the outcome each appear separately. Real causes:
- **Outcomes recorded elsewhere:** readmissions at another hospital never reach the source system.
- **Missing follow-up:** outcomes go missing more for some groups (transfers, out-of-system deaths, insurer gaps).
- **A new site or population:** a model trained where a group's positives were rare is deployed where they are common.
- **Small-cell suppression:** privacy rules (for example CMS's under-11 rule) remove rare combinations from shared extracts.
- **Rare events:** a small site truly has zero cases yet.

It does not model wrong or noisy values, gradual gaps (that is the 98.5% condition), or columns that were never collected.

The two conditions ask different questions:
- **100%:** can medical knowledge stand in for cases never collected?
- **98.5%:** does knowledge help when only a few cases were collected?

### Expected ranking

**100% removal (BKO's home ground).**
- SMOTE, ADASYN, CART, TVAE and the Bayesian network add no rows, so expect them at or near `none`.
- The real contest is BKO against CTGAN, the Gaussian copula and uncurated CLLM.
- CTGAN and the copula can place rows in the group only by mixing traits learned from other groups, with no knowledge of how this group's readmissions look. Expect BKO to lead them on recall.
- **BKO vs CLLM is the key test.** Both are language-model generators; only BKO carries the ontology. A BKO lead over CLLM isolates what the knowledge adds.

**98.5% removal.**
- SMOTE and ADASYN build 1,250 rows from 19 true ones, so their rows hug those 19 patients. That may score well on recall but risks near-copies, which the copy and privacy checks will show.
- Expect a closer race between BKO, CTGAN, the copula, SMOTE and ADASYN, as on Synthea (E3), where BKO was level with the best.

**HGB.** BKO's +0.04 is small. A method whose rows sit closer to the true ones (SMOTE at 98.5%) may beat it under HGB, so BKO's lead there is not safe.

**AUC-PR.** Expect every method that adds rows to lose some AUC-PR against `none`, as in E4.

### Risks to the claim

- If CLLM had matched BKO at 100%, the gain would come from the language model, not the ontology. It did not: at 100% removal both BKO arms beat CLLM.
- If BKO lags under HGB, its rows are plausible enough for LR but not faithful enough for a split-based model. The BKB likelihood gate (below) tests whether the likelier rows are the more faithful ones.

## 5. Outstanding work and plans

1. **E6 close-out (after Nov 4, about 22:30 EST):**
   - Move and back up the run folders.
   - Replay one slice per condition byte for byte.
   - Merge the slices of each condition and read every arm against `real` and `none`.
   - Record E6 in the evidence register and open its PR.
2. **BKB likelihood gate on Diabetes (free):** split BKO's candidates at the median BKB log-probability and test whether the top half is more faithful. If it separates, run a curated BKO arm once (cap $80). If not, record the result and remove the reserved curation code.
3. **New task, `privacy-identifier-perturbation` (paid, blocked on E6):**
   - **What:** a language model rewrites the fields a patient could be identified by in `bko-direct`'s rows: race, the 8 count fields (days in hospital, lab procedures, medications and so on), the 8 drug columns, `change` and `diabetesMed`. It must not touch any ontology-mapped concept (diagnoses, lab states), age, sex or the label.
   - **Checks:** a row is refused if any mapped concept or cell coordinate changes. A holdout's perturbation is refused if column shapes or pair trends drift past a threshold, which the task's ADR fixes before any run.
   - **Measures:** a per-method privacy table for both E6 removal levels (5th-percentile and median distance, exact copies). The perturbed arm's paired difference against `bko-direct` on recall, AUC-PR and distance.
4. **Batch close-out:** a summary of every PR in this batch, with its changes, runs, spend and results.
5. **Candidates for the next plan:**
   - Vectorize the Bayesian-network draw, the largest local cost.
   - Raise the number of API calls in flight.
   - A full thinking-on local run.
   - A per-concept-normalized likelihood score.
   - The open cleanup backlog entries.

## 6. Spend this batch

- G6 local and thinking runs: $0.
- Diabetes generation pilots: under $1. CLLM curation pilot: $15.00.
- E6: about $140 so far, about $670 projected, $1,500 cap.

## 7. Pull requests in this batch

The batch has 6 planned PRs. Four have merged. The fifth (the E6 run) is in progress, and the sixth follows it.

| PR | Title | Merged | What it did |
|---|---|---|---|
| [#69](https://github.com/joe-dartmouth/synthetic/pull/69) | Close G6: replay the local runs and read them on a shared split | Nov 1 | Made the on-device model runs (Gemma, Nemotron, Olmo) replayable byte for byte from their logs. Added the shared test-split reading, which rescores runs on the test records none of them copied, so runs whose arms copied different records can still be compared. Recorded G6's generator-model results. |
| [#70](https://github.com/joe-dartmouth/synthetic/pull/70) | Measure what thinking on costs the local generators | Nov 2 | Added thinking-on variants of the local models and a compute metric (FLOPs per record). Measured the cost: Gemma about 3x, Nemotron about 14x counting re-asks after truncation. |
| [#71](https://github.com/joe-dartmouth/synthetic/pull/71) | Ready Diabetes 130 for E6: frame, ICD-9 rows and power | Nov 2 | Added the Diabetes 130 dataset: 69,283 first encounters, readmission within 30 days as the label. Mapped ICD-9 diagnosis codes to MONDO and HPO. Let each experiment set its own scoring cutoff. Chose E6's target group and size from a 300-seed power dry run. |
| [#72](https://github.com/joe-dartmouth/synthetic/pull/72) | Ready the harness for E6 and name its runs | Nov 3 | Made the harness fast enough for a 69k-row dataset: memoized the Bayesian network's mutual information, computed privacy distances in blocks, and capped replica size. Let a removal take a whole group (the 100% condition) and filled shortfalls once. Let a run stopped by its spend ceiling resume under a raised ceiling. Named E6's 20 run slices and dropped the curated CLLM arm. |
| PR5 (open branch `feat/diabetes-runs`) | Run E6 | pending | Runs E6's 20 paid slices, replays a slice per condition, merges the slices, reads every arm and records E6 in the evidence register. |
| PR6 | BKB likelihood gate on Diabetes | pending | Tests whether BKO rows the BKB scores as likelier are more faithful. If they are, runs a curated BKO arm once (cap $80). |

Each merged PR passed the full gate (`make check`) and a six-perspective diff review: code, security, tests, docs, language and observability. Every finding was fixed in the PR or filed in the cleanup backlog.

## 8. The techniques compared, and how BKO differs

### How BKO works

1. **Represent:** fuse the medical ontologies (MONDO for diseases, HPO for findings) with the dataset into a Bayesian Knowledge Base. Each statement, such as "this diagnosis goes with this finding", is weighted by how often it appears in the data.
2. **Detect:** find the gaps: groups (age band x sex x label) whose own records leave medically expected statements unsupported.
3. **Generate:** ask a language model (Haiku 5.5) for records for each gap group. The prompt carries no patient records. It carries:
   - the group and its record count;
   - each numeric field's allowed range;
   - the gaps and the ontology findings behind them;
   - a summary of similar patients: the share showing each concept, and each number's spread.
4. **Validate:** keep only records that pass consistency checks and range limits.
5. **Use:** add the kept records to the training data.

BKO is the only method here that brings outside medical knowledge. Every other method learns from the training rows alone, so it can only reproduce patterns the data already holds.

### How `bko-direct` works, in detail

`bko-direct` is BKO's pipeline with the language model removed: records are drawn straight from the fused BKB. It is the best method at 100% removal, so this is the part to explain carefully. Code: `src/bko/generate/direct.py` (`sample`), `src/bko/gaps/shares.py` (`shares`), `src/bko/evaluate/direct.py` (the arm). Design record: ADR `research/bkb-direct-generation`.

**Inputs.** For one holdout:
- the training rows (never the test split);
- the request: which group to fill and how many rows. In E6 that is men aged 60-79 who were readmitted, filled to 1,269 rows. All arms get the same request.
- the field mappings: the Diabetes frame's ICD-9 diagnosis indicators and two lab states, each mapped to a MONDO disease or an HPO finding. Examples: heart failure (MONDO:0005252), chronic kidney disease (MONDO:0005300), essential hypertension (MONDO:0001134), elevated HbA1c (HP:0040217), hyperglycemia (HP:0003074).

**Step 1: build the BKB's concept rates.** The fused BKB holds, for each mapped concept, two probabilities:
- `p_cell`: the share of the target group's own training records that show the concept;
- `p_group`: the share across its label group, that is, every readmitted patient of any age and sex.

They blend into one rate:

    share = (n_cell * p_cell + 20 * p_group) / (n_cell + 20)

Here `n_cell` is the group's record count, and 20 is the threshold below which a group counts as underrepresented. A small group leans on its label group, and a large group on itself.
- At 100% removal, `n_cell` = 0, so each share is exactly the readmitted-patient rate.
- At 98.5% removal, `n_cell` = 19, so the share is about half the group's own rate and half the readmitted rate.

**Step 2: draw which concepts the new patient has.** Each concept is drawn present or absent independently, at its share, from a seeded random generator. Concepts that follow the label are set from the label. Where one concept's codes lie inside another's, the outer is drawn given the inner, so a patient never shows the narrow diagnosis without the broad one.

**Step 3: fill the fields, column by column.**
- **Sex and label:** set to the target group's (male, readmitted).
- **Age:** copied from a random training record of the group. If the group has none, it comes from a readmitted patient in the same age band (60-79).
- **Concept fields** (the diagnosis indicators and lab states): fields that share a concept form a block. For each block it picks a random training record whose concept states match the draw exactly, from the group first and the label group as a fallback, and copies that block's values. In practice this sets each indicator to the drawn state.
- **Every other field** (race, the 8 count fields such as days in hospital and number of medications, the 8 drug columns, `change`, `diabetesMed`): each is copied **independently** from its own random record, from the group or, if it has none, from any readmitted patient.

**Step 4: validate and count.** Every row goes through the same validation as the language-model arms (consistency and range checks) and counts toward the group its own values name. It always passes: its values are copies of valid training values, and all 126,900 rows passed.

Every draw comes from one seeded generator, so a run reproduces byte for byte. There is no API call and no cost.

**What this means.**
- **It is a label-conditioned, ontology-structured hot deck.** At 100% removal, each new row is a patient whose diagnoses follow readmitted patients' rates, whose age comes from a readmitted 60-79-year-old, and whose other fields each come from a different random readmitted patient. One row mixes values from about 20-30 separate draws: age, around a dozen concept blocks, and 19 other fields.
- **The ontology's role is structural.** The mappings define which diagnosis codes form one concept (an ICD-9 range mapped to a MONDO disease), which fields travel together, and which concepts nest. The statistics come from the data. The `is_a` closure (parent concepts) is recorded but never used to set a field.
- **Why it wins under logistic regression:** LR adds up each feature's effect. Rows built from readmitted patients' marginal rates move every feature in the right direction for the missing group, so the classifier learns that older readmitted men exist and what they look like on average. The values are real, so nothing implausible gets learned, which fits its zero AUC-PR cost.
- **Why it is weaker under HGB** (0.359 against 0.585 for the true rows): independent draws break the correlations between fields, for example between days in hospital and number of medications. Tree models learn exactly those interactions.
- **Its key assumption:** the missing group looks like other patients with the same outcome. On Diabetes readmissions that holds well. A group whose positives differ from everyone else's (a different disease mechanism, say) would break it.
- **Privacy:** no single source patient dominates a row, because each non-concept field comes from a different random patient. That fits its distance results at 100% removal: its closest rows are farther from training patients than CTGAN's.
- **At 98.5% removal it behaves differently.** The group's 19 remaining patients come first: age, every non-concept field and any matching concept block are copied from those 19, and the label group is used only when none matches. So the 1,250 new rows are recombinations of 19 people's values, a closer analogue of SMOTE. Watch its recall and its distance there; the reading lands tonight.
- **How it differs from the language-model `bko` arm:** `bko` gives the language model the same group, gap statements and summary rates, and the model writes values. Those values are novel (a larger median distance) but less faithful, and that arm trails `bko-direct` by 0.230 recall.

### The methods

| Method | How it makes rows | Can it fill a group with no rows? |
|---|---|---|
| `none` | Adds nothing: the baseline. | n/a |
| `class-weight` | Adds nothing; weights the rare label more heavily in training. | n/a |
| SMOTE | Draws new points on lines between a real minority row and its nearest real neighbours. | No: needs rows of the group. |
| ADASYN | SMOTE, but makes more points near hard-to-classify rows. | No |
| CART | Fits a decision tree per column and samples values from leaves of real rows (synthpop-style). | No |
| Gaussian copula | Models each column's distribution plus pairwise correlations, then samples. | Yes, by combining traits from other groups. |
| Bayesian network | Learns a dependency graph between columns and samples it row by row. | Barely: 1-3 rows in E6. |
| CTGAN | Trains a generative adversarial network on the table; samples conditioned on the group. | Yes, from traits learned elsewhere. |
| TVAE | Trains a variational autoencoder on the table. | No in E6: it samples the whole table without conditioning, and 1,000,000 draws per attempt held no row of the target group, even at 98.5%. This is a property of TVAE on this data, not a harness defect; the E6 smoke check showed it too. |
| CLLM (uncurated) | Shows a language model 20 real training records per call and asks for more like them (Seedat et al. 2024). The curated version filters the output with a classifier; dropped from E6. | Yes |
| `bko` | The pipeline above, guided by each group's own gaps. | Yes |
| `bko-population` | Ablation: gaps found over all records together. | Yes |
| `bko-frequency` | Ablation: gaps from a group's underrepresented concepts alone. | Yes |
| `bko-skeleton` | Ablation: each request also prescribes each record's concepts, drawn from the BKB. | Yes |
| `bko-direct` | No language model. For each new record it draws which medical concepts are present (diagnoses, lab states), at the rates the BKB holds for the target group, falling back to its label group (all readmitted patients) when the group is empty. It sets the concept fields to match, copies every other field independently from a different random readmitted patient, takes age from a readmitted patient in the same age band, and sets sex and label to the target group's. Each row mixes real values from many patients (detail above). | Yes, by borrowing from patients with the same outcome and concept profile. |
| `real` | Puts the removed true rows back: the ceiling, not a method. | n/a |

### Ablations: what each one isolates

- **BKO vs CLLM:** both use the same language model. CLLM learns from example records; BKO learns from the ontology and aggregates. The gap between them is what the knowledge adds.
- **`bko` vs `bko-population` and `bko-frequency`:** which gap definition guides generation best.
- **`bko` vs `bko-skeleton` and `bko-direct`:** whether the language model adds anything beyond the BKB's own structure.

### Security and privacy

**What leaves the machine**

| Method | Data sent outside | Notes |
|---|---|---|
| `none`, `class-weight`, SMOTE, ADASYN, CART, copula, Bayesian network, CTGAN, TVAE, `bko-direct` | Nothing | Runs fully local. |
| CLLM | **Real patient records**: 20 training rows per call, sent to the API | The highest disclosure. On real patient data, it needs a provider agreement (a BAA and zero data retention) or a local model. |
| `bko` and its API ablations | Aggregates and ontology statements only: group counts, ranges, concept shares and numeric spreads | No individual record is sent. Aggregates of a very small group can still identify people, so a production version should apply a small-cell rule (for example, suppress counts under 11). |
| BKO with a local model (G6) | Nothing | Gemma, Nemotron and Olmo run on-device. Their row quality is lower than Haiku's or Sonnet's (G6). |

**How close each method's rows sit to real patients**
- **SMOTE and ADASYN:** each new row lies between two real patients. At 98.5% removal they build 1,250 rows from just 19 patients, the highest re-identification risk in the study.
- **CART:** samples from leaves of real rows, so it can reproduce real rows nearly whole.
- **CTGAN and TVAE:** can memorize training rows, mostly on small data. No differential-privacy training is used (DP-SGD variants exist).
- **Gaussian copula and Bayesian network:** model distributions rather than rows, so memorization is lower. Rare combinations can still leak through.
- **CLLM:** sees real records and can echo them.
- **BKO:** never sees individual records. Its rows can still match real people by chance, or through the model's memory of public datasets.

**Measured so far**
- **E1 privacy criterion, met:** at the 5th percentile, BKO's added rows sit farther from the training rows than test rows do.
- **Copies:** every run counts generated rows that equal a real test record in every column. On public UCI data, Sonnet 5.5's replica rows copied 457 of 5,900 test records word for word, where Haiku 5.5's copied 1. The model had memorized the public dataset. Haiku is the generator for every paid run.
- **E6, 100% removal: distance to the closest training patient, per method.** These are means over 100 holdouts. A larger distance means more private. The 5th percentile reads the closest 5% of a method's rows, the ones most at risk of matching a real person; the median reads a typical row.

| Method | 5th percentile | Median | Exact test copies |
|---|---|---|---|
| Real test patients (baseline: how close real patients sit to each other) | 0.0031 | 0.0138 | n/a |
| `real`: the removed true patients | 0.0125 | 0.0378 | 0 |
| Gaussian copula | 0.0229 | 0.0581 | 0 |
| `bko-skeleton` | 0.0230 | 0.0814 | 0 |
| `bko` (language model) | 0.0211 | 0.0965 | 0 |
| **`bko-direct`** | **0.0194** | **0.0539** | 0 |
| CLLM (uncurated) | 0.0171 | 0.1097 | 0 |
| CTGAN | 0.0164 | 0.0523 | 0 |
| `bko-population`, `bko-frequency` | 0.0150 | 0.0784-0.0787 | 0 and 1 |
| SMOTE, ADASYN, TVAE, CART | no rows added | | |
| Bayesian network | 0.0378 (1-3 rows, 77 holdouts) | 0.0454 | 0 |

**`bko-direct` paired against each rival, 5th percentile (mean difference over 100 holdouts, 95% interval):**
- vs CTGAN: +0.0030 [+0.0026, +0.0035], farther, so more private.
- vs CLLM: +0.0023 [+0.0021, +0.0025], farther.
- vs `bko`: -0.0017 [-0.0022, -0.0012], closer.
- vs the Gaussian copula: -0.0035 [-0.0038, -0.0031], closer.

These intervals are plain paired intervals over holdouts, not the corrected intervals used for utility.

**What it shows:**
- Every generator's closest rows sit 5-7x farther from training patients than real patients sit from each other (0.0031), and farther than the removed true patients (0.0125). No method produced near-copies by this measure.
- `bko-direct` sits in the middle: slightly more private than CTGAN and CLLM, slightly less than the copula and the language-model `bko`. Copying real field values, each from a different patient, did not make it a privacy outlier.
- The language-model arms (`bko`, CLLM) have far larger medians. Their typical row is more novel, which also fits their lower fidelity.
- The 98.5% removal table follows tonight. The task `privacy-identifier-perturbation` will test whether rewriting `bko-direct`'s identifying fields moves it farther away without costing recall.
- **`bko-direct` caveat:** every value it writes is a real patient's value, though each field comes from a different patient. It runs fully locally and its rows passed the distance check above. A rare value (say, an unusual count of emergency visits) is still copied verbatim. A deployment would add a minimum-distance filter or suppress rare values. A new task (below) tests a third option: a language model rewrites only the fields a patient could be identified by.

**Security of the pipeline**
- **Data:** Diabetes 130, UCI Heart Disease and Synthea are public, de-identified or synthetic. No PHI is used.
- **Storage:** the call logs hold every prompt and reply. They stay in the gitignored `runs/` folder and are never committed.
- **API safety:** API keys come from the environment. Tests cannot reach the paid API without `--live`, and every paid run has a spend ceiling.
- **Production:** real patient data would need API keys kept in a secrets manager, encrypted logs, and the provider agreement above, or the local-model path.
