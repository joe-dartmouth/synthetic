# Research plan

Sources: the team's proposal 2-pager (sections 1 to 6) and research brief (TL;DR, A1, B5, D10, D11, F15), both kept outside the repository, of which this document is the plan of record; Santos et al. [`santos-2024`][santos-2024]; facts verified outside those sources (pybkb, synthcity resolution); for [section 10](#10-related-work), the works in [literature.md](literature.md#ontologies-reasoners-and-knowledge-graphs), each entry saying how far it was read.

This is the plan of record. The 2-pager governs; the brief is background, explained in
[BACKGROUND.md](BACKGROUND.md). [Section 9](#9-source-reconciliation) records each point where the
two disagree and which one governs. The pipeline and its packages are in
[ARCHITECTURE.md](ARCHITECTURE.md), the dates in [ROADMAP.md](ROADMAP.md), and results in
[EVIDENCE.md](EVIDENCE.md).

## 1. Problem and organizational context

Clinical machine-learning systems fail at long-tail populations. A disease-detection model receives
thousands of records for common presentations and almost none for rare subtypes or underserved
subgroups, and the result is costly missed diagnoses.

The existing paths to more data are slow or lossy:

- Data-use agreements take weeks.
- HIPAA Safe Harbor strips the quasi-identifiers that make rare cases findable.
- SMOTE only interpolates between existing examples.

Commercial synthetic-data platforms (the 2-pager names Gretel, MDClone and MOSTLY AI) optimize for
statistical fidelity to the distribution they are given. When that distribution is incomplete,
fidelity reproduces the very gaps that cause the failure. [MARKET.md](MARKET.md) holds the vendor
facts.

The project builds the platform around knowledge instead, using Bayesian Knowledge-driven
Ontologies (BKOs), developed at Dartmouth's Thayer School of Engineering by Santos, Jurmain and
Ragazzi [`santos-2024`][santos-2024]. A BKO integrates description logic with Bayesian Knowledge
Bases (BKBs), and in contrast to a Bayesian network it treats incompleteness as a property of the
representation. [BACKGROUND.md](BACKGROUND.md#bayesian-knowledge-driven-ontologies) explains both.

## 2. Intended users and evaluation setting

The primary users are machine-learning engineers or clinical data scientists at a digital-health
startup or an academic laboratory. Clinicians are secondary users.

The initial evaluation setting is heart-disease prediction on the UCI Heart Disease dataset:

- **Task**: predict the presence of heart disease.
- **Underrepresentation**: fewer than 20 training records per age–sex subgroup-label combination.
- **Status of that threshold**: a prototype convenience criterion. It does not claim to define
  clinical rarity.

The archive's site, the label coding, the age bands, the handling of missing values and the
train/test split are set by ADR
[`research/uci-evaluation-frame`](adr/research/uci-evaluation-frame.md); `bko.datasets` holds the
values, and an experiment's configuration may fix other split seeds.

A complementary Synthea experiment assesses gap detection by deliberately ablating selected cases
from the generated population, which gives the detector a known set of removed cases to find.
ADR [`research/synthea-ablation-design`](adr/research/synthea-ablation-design.md) fixes how the
population is generated and what a removed case is.

A downstream Synthea experiment runs E1's arms on a frame of that population's patients alive on
the snapshot date and aged 40 and over. In each holdout, 0.9 of one cell's cases leave the training
rows, and the recall each method recovers on that cell's held-out cases measures the augmentation.
That recall is read at scope `flagged-subgroups`: recall counts only cases, and the removed cell's
subgroup is the only flagged one, so it is the recall on the removed cell's test cases; AUC-PR at
that scope covers the subgroup's test records of both labels. It is read with no criterion, each
arm paired in-run against `real`, the un-ablated training split, and `none`, the ablated split
unaugmented. ADR
[`research/synthea-downstream-design`](adr/research/synthea-downstream-design.md) fixes the frame,
the removal and the reference.

The three experiments are register entries
[E1](EVIDENCE.md#e1-uci-heart-disease-augmentation),
[E2](EVIDENCE.md#e2-synthea-gap-detection-ablation) and
[E3](EVIDENCE.md#e3-synthea-downstream-augmentation).

## 3. Approach

The approach has four steps. [ARCHITECTURE.md](ARCHITECTURE.md#pipeline) draws the pipeline and
names the package for each.

### Represent

- Load MONDO and HPO as BKO fragments.
- Add the user's dataset as a second kind of fragment, weighted by observed frequency: one per cell
  and one of every record ([Detect](#detect)).
- Map ontology concepts to dataset fields, starting with the disease label and chest-pain type.
  Document each mapping's source, encoding and limits, with clinician review where available.
- Govern numeric fields by separately specified range constraints.
- Exclude from scope any concept without a defensible corresponding field.

ADR [`research/ontology-mapping-design`](adr/research/ontology-mapping-design.md) sets how the
ontologies are loaded and how the mapping table is sourced, scoped and checked;
`bko.ontology.mapping` holds the rows.

### Detect

Apply the BKO instantiation algorithm to find statements the ontology implies but the data does not
support. Santos et al. state, as their Proposition 5.1.1, that such ungrounded statements
contribute nothing to the model's distribution, so the plan uses them as the formal definition of a
dataset gap rather than a heuristic. The decision and its reversal conditions are in ADR
[research/bko-formal-gap-definition](adr/research/bko-formal-gap-definition.md).

Gaps are read per cell of the evaluation setting. The primary definition grounds each cell's
records as a dataset fragment of their own, at the age-sex-label granularity of
[section 2](#2-intended-users-and-evaluation-setting) (age and sex on Synthea, whose label is the
ablated condition). Two definitions run beside it as ablations: one population-level fragment of
every record, and a frequency-aware score that flags a concept a cell shows credibly less often
than the rest of its label group, a coverage-style score in place of the brief's Dirichlet one (ADR [research/frequency-aware-gap-score](adr/research/frequency-aware-gap-score.md)).
E2 scores all three against its ablation's truth, and E1 runs one BKO arm per definition, judging
only the primary one.

Before gap detection rests on that formalism, `bko.ontology.santos` rebuilds, with `bko.bkb`, the
paper's MONDO/DO fusion example (its section 7), recorded as [R1](EVIDENCE.md#r1-mondo-and-do-fusion-reproduction). The
reproduction holds when `python -m bko.ontology reproduce` reports no `differs` row over the pinned
releases: each of the paper's 11 terms and 17 PTAs is found, its misprints aside; instantiation
from its anchor, as its algorithm states, holds the same 8 PTAs after two passes and all 17 at its
fixpoint; each of its two contradictory inferences has probability 0.33 to two decimals; and its
chain from sciatic neuropathy to inflammatory disease exists in the fused BKB and in neither the
MONDO nor the DO fragment, nor in the whole MONDO release.

### Generate and validate

Generation is guided by the ontology relationships relevant to each gap, and records are generated
through the Anthropic Messages API (2-pager section 4). A second generator draws records from the
fused BKB alone, with no API call: each mapped concept present at its share, field values read
from training records that show the drawn concepts (ADR
[research/bkb-direct-generation](adr/research/bkb-direct-generation.md)); it isolates what the
mapping table and the data's concept shares contribute, and no criterion reads it. Each candidate
record is validated before augmentation. The validity criterion counts the records that pass the
model's consistency checks, and the completeness criterion counts the detected gaps that close
after fusion ([section 4](#4-success-criteria)). ADR
[research/record-validation-design](adr/research/record-validation-design.md) sets what the
consistency checks are and how a candidate is counted, and ADR
[research/record-generation-design](adr/research/record-generation-design.md) sets what a
generation request carries and how its replies become candidates.

### Two kinds of gap, evaluated separately

Count-based gaps and BKO-detected gaps are evaluated separately. The ADR fixes count-based gaps as
the age–sex subgroup-label combinations under the
[section 2](#2-intended-users-and-evaluation-setting) threshold, kept as the comparison the formal definition
is measured against. BKO-detected gaps are read three ways ([Detect](#detect)), each evaluated
separately from the count-based ones.

## 4. Success criteria

Verbatim from 2-pager section 5:

| Criterion | Statement |
|---|---|
| Primary | "BKO-guided augmentation improves rare-class recall and AUC-PR by at least 10% over an unaugmented baseline, and matches or outperforms CTGAN, TVAE, and SMOTE at equal row counts." |
| Completeness | "At least half the detected gaps close after fusion." |
| Validity | "At least 95% of generated records pass the model's consistency checks." |
| Privacy | "Distance-to-closest-record remains above the real-data baseline." |

The controls and the null-result commitment, also verbatim: "Because evidence on whether generative
augmentation outperforms SMOTE is mixed, no-augmentation and class-weighting controls will be
included, and results will be reported transparently if knowledge guidance provides no benefit."

The comparison set is therefore: no augmentation, class weighting, SMOTE, ADASYN, CTGAN and TVAE
[`xu-2019`][xu-2019]. A result where knowledge guidance gives no benefit is reported, not dropped.
ADR [`research/baseline-augmentation-design`](adr/research/baseline-augmentation-design.md) records
how each method fills a requested quota of rows.

The 2-pager leaves these readings open. Each experiment fixes them in its configuration in
[EVIDENCE.md](EVIDENCE.md) before it runs:

- Whether "at least 10%" is a relative improvement or a difference in percentage points.
- Which class "rare-class recall" measures: the positive label overall, or the label within the
  underrepresented subgroup-label combinations of
  [section 2](#2-intended-users-and-evaluation-setting).
- ADASYN is a baseline in 2-pager section 4 but is not named in the primary criterion.
- 2-pager section 4 joins its four baselines (CTGAN, TVAE, SMOTE, ADASYN) with "or". The plan runs
  all four, and E1's configuration records that choice.
- What "matches or outperforms" counts as a match once results carry intervals.
- Which comparison "real-data baseline" names for distance to closest record
  ([section 5](#5-evaluation-contract)).
- What the completeness criterion counts: which gap definition's (cell, statement) pairs are the
  detected gaps, which generated records join before detection runs again, and whether exactly
  half of them closing meets the criterion.
- What the validity criterion counts: which candidates enter the rate (every one validation sees,
  those a later rule drops and those of refused attempts included), and whether exactly 95% meets
  it.

E1 departs from the primary and privacy criteria as worded; [section 9](#9-source-reconciliation)
records how.

## 5. Evaluation contract

The brief (B5) sorts synthetic-data metrics into three buckets. The plan's criteria draw on two of
them; fidelity is reported, with no pass threshold, for every generator that fills a replica of the
training split's cells. In E1 every row-adding arm fills one
([EVIDENCE.md](EVIDENCE.md#e1-uci-heart-disease-augmentation)).

| Bucket | Metrics the brief lists | Criterion it serves |
|---|---|---|
| Fidelity | Kolmogorov–Smirnov statistic (numerical marginals); Total Variation Distance (categorical); pairwise correlation difference; Hellinger distance; Jensen–Shannon / KL divergence; Wasserstein distance; propensity-score MSE (pMSE, range [0, 0.25)); discriminator/detection AUC (how well a classifier separates real from synthetic; lower is better) | None; reported |
| Utility | Train-on-Synthetic-Test-on-Real (TSTR) against Train-on-Real-Test-on-Real (TRTR); downstream classifier accuracy, AUC and F1. For imbalanced tasks, AUC-PR and macro-F1 rather than accuracy | Primary: rare-class recall and AUC-PR |
| Privacy | Distance to Closest Record (DCR); Nearest-Neighbor Distance Ratio (NNDR); membership-inference risk; differential-privacy guarantees if ambitious (CTAB-GAN+ and DP-SGD support them) | Privacy: DCR above the real-data baseline |

The validity and completeness criteria are not synthetic-data metrics: validity is the pass rate of
the consistency checks, and completeness is the share of detected gaps that close.

**Libraries.** The brief recommends SDMetrics and synthcity as the safest stack, and also lists
SynthEval, TableEvaluator and Anonymeter. The repository uses:

- scikit-learn for utility;
- SDMetrics for fidelity;
- SDMetrics' `calculate_dcr` for distance to closest record, read against the test rows (ADR
  [`research/e1-evaluation-protocol`](adr/research/e1-evaluation-protocol.md)).

synthcity is not a dependency; it does not resolve on this repository's interpreter (ADR
[tooling/python-314-without-synthcity](adr/tooling/python-314-without-synthcity.md)). SynthEval,
TableEvaluator and Anonymeter are not dependencies either. The brief warns that SDMetrics aggregates
metrics into single scores, which limits per-metric comparison, and that SynthEval is more granular.

**Reference point.** The brief offers a colorectal-cancer study [`colorectal-nd`][colorectal-nd],
reporting TSTR/TRTS AUC of 0.99/0.98 and pMSE 0.223, as a benchmark for "what good looks like".

The brief calls this evaluation well standardized and very doable in 10 weeks; it assumed three
students and 10 weeks ([section 8](#8-scope-assessment)).

## 6. Datasets

The plan uses UCI Heart Disease and Synthea, both openly available (2-pager sections 2 and 4), and
UCI Diabetes 130-US Hospitals for subgroup removal on real patients
([below](#limits-of-uci-heart-disease-as-the-evaluation-frame)). The other rows are the
brief's registry (D10), kept for reference.

| Dataset | Access | Description (brief D10) | In the plan |
|---|---|---|---|
| UCI Heart Disease (Cleveland) | Open | Classic 303-record binary/multiclass detection task; naturally small, which suits a low-data story; the evaluation frame is ADR [`research/uci-evaluation-frame`](adr/research/uci-evaluation-frame.md) | Evaluation setting ([section 2](#2-intended-users-and-evaluation-setting)) |
| Pima Indians Diabetes | Open | Classic small imbalanced binary task | No |
| UCI Diabetes 130-US Hospitals | Open, CC BY 4.0 [`clore-2014`][clore-2014]; de-identified under HIPAA [`strack-2014`][strack-2014] | Larger; used in recent synthetic-data audit benchmarks | Subgroup removal on real encounters, E6 ([EVIDENCE.md](EVIDENCE.md#e6-diabetes-130-subgroup-removal)) (below); pinned as `uci-diabetes-130` in `bko.datasets.sources`, its frame in [`DEVELOPMENT.md`](DEVELOPMENT.md#data) |
| NHANES | Open | Openly available national survey data | No |
| Synthea / SyntheticMass | Open, no restrictions | ~1M synthetic patients, SNOMED-CT/RxNorm/LOINC-coded; models the 10 most common primary-care reasons and 10 highest-morbidity chronic conditions | Gap-detection ablation and downstream augmentation ([section 2](#2-intended-users-and-evaluation-setting)) |
| MIMIC-IV demo, eICU demo | Open; no credentialing or CITI training | Demo subsets of the credentialed datasets | No |
| Full MIMIC-IV, MIMIC-IV-Note, MIMIC-IV-ED | Credentialed, on PhysioNet | Real EHR data | No |

**Synthea's caveats.** Synthea has documented fidelity flaws: per the JAMIA 2018 paper it
"incorrectly generated 20% of patients in the infant age group" and "failed to note diabetes onset
after age 52". The brief therefore treats it as a development and prototyping substrate, not a
gold-standard evaluation set. The plan uses it for the gap-detection ablation and the downstream
experiment, where the ground truth is the set of deliberately removed cases rather than clinical
fidelity. The caveat applies to the downstream experiment's recall: it measures how well a method
recovers Synthea's generated cases, not real patients.

**Credentialed access.** The 2-pager plans no credentialed data. If a member wants real EHR data,
full MIMIC-IV, MIMIC-IV-Note and MIMIC-IV-ED on PhysioNet require:

1. A PhysioNet account.
2. A credentialed-user application, which requires a reference.
3. CITI "Data or Specimens Only Research" training.
4. A signed dataset-specific DUA. Each dataset needs its own.

Approval normally takes a few days to 1–2 weeks, but PhysioNet has warned of delays "up to 45 days"
during high-volume periods. The brief's advice is to build the whole pipeline on open data so the
project never blocks on credentialing, and to start credentialing on day one as a parallel,
non-blocking track.

**Credentialed data stays local.** The pipeline sends each cell's description, each numeric
field's training-split minimum and maximum (age within the cell's band), the 25th, 50th and 75th
percentiles of each numeric field over the same-label records of the other cells (the `summary`
level), that reference group's record count and each mapped concept's share of it in percent (the
`profile` block, which `summary` includes), and ontology guidance to the Anthropic Messages API.
Every run at `SETTINGS`' level, `summary`, sends them. A `bko-skeleton` call also sends each asked
record's drawn concept names: ontology labels drawn at the fused BKB's aggregate shares, never a
record's values. Of the BKO arms, only `bko-direct-perturbed` sends record values, the 19
unmapped fields of rows of open Diabetes 130 data with each count field's training range (ADR
[`research/identifier-perturbation`](adr/research/identifier-perturbation.md)); every other BKO
call carries none, but every bound and percentile is a real training value: two per numeric field leave on each call, three more per
field at `summary`. Their exposure grows as the split or a reference group shrinks, since
a group of one record sends that record's values and a split of identical records sends a whole
record. The CLLM arms, a comparison baseline, send training records whole (ADR
[`research/cllm-arm-design`](adr/research/cllm-arm-design.md)). Data held under a DUA must not
reach that API or any other third-party service unless the DUA permits it, bounds and percentiles
included: every Messages arm refuses data whose source's `Agreement` (`bko.datasets.sources`)
records no `sending` permission; check the DUA's current terms before recording one. E3's Synthea
frame sends the same kinds of quantity, but its rows are generated, so none describes a
person ([`DEVELOPMENT.md`](DEVELOPMENT.md#generation), What leaves the machine).

### Limits of UCI Heart Disease as the evaluation frame

UCI Heart Disease cannot tell the arms apart, so a null result on it says little about knowledge
guidance:

- **Too few test records move a reading.** Each holdout holds out 59 of its 297 records, so the
  flagged subgroups' recall changes on a handful of holdouts: in the judged run it differs from
  the non-generative comparators' on 1 to 6 of 100 holdouts, and every primary and secondary
  interval contains zero with upper bounds up to about +0.06
  ([E1](EVIDENCE.md#e1-uci-heart-disease-augmentation), Result). An effect smaller than
  that cannot be read.
- **Its thin cells carry no missing signal.** The flagged cells are thin by count, but a
  classifier learns the label from 13 strong features over the whole split, so added rows that
  follow the training rows change little: every guidance level, from the cell and its bounds alone
  to the ontology chains and the reference group's summary, gives the logistic regression a mean
  recall and AUC-PR over all records within about 0.005 of each other's, and a flagged-subgroup
  recall within about 0.008 ([G3](EVIDENCE.md#g3-guidance-level), Result).
- **The generators know it.** The archive is public and old. Sonnet 5.5's replica fills copied 457
  of 5900 held-out test records word for word, in 98 of 100 holdouts, where Haiku 5.5's copied 1, so
  decontamination shrinks each holdout's test split and a cross-run pairing loses its holdouts
  ([G6](EVIDENCE.md#g6-generator-model)); a generator's knowledge here is partly recall of this
  dataset.

A frame that can read a difference needs a subgroup with enough test positives per holdout to
move an interval, a gap that leaves count-based methods little or nothing to copy (a subgroup's
training positives removed, as E3 does on Synthea), coded features the ontology can inform, and
records the generators have not memorized under terms that permit sending them. E3 tests the
removal on generated patients; E6
([EVIDENCE.md](EVIDENCE.md#e6-diabetes-130-subgroup-removal)) runs it on UCI Diabetes 130-US
Hospitals' real encounters, with a power calculation fixing its design before the first paid run. UCI stays the judged frame for E1's criteria.

## 7. Augmentation versus resampling

Whether synthetic augmentation beats SMOTE, ADASYN or class weighting is genuinely mixed (brief
D11): no method dominates, and sometimes no resampling at all is optimal. The brief treats this as
an asset, since it makes the head-to-head comparison a real question rather than a foregone
conclusion. It also says the proposal must include SMOTE, ADASYN, class weighting and a
no-augmentation baseline, because omitting them is "the most common reviewer criticism". The plan
includes all four ([section 4](#4-success-criteria)). The studies behind the conclusion are in
[BACKGROUND.md](BACKGROUND.md#augmentation-versus-resampling).

Most row-adding arms trade ranking for recall on the group they fill, as the rebalancing
literature leads one to expect; `bko-direct` at f = 1.0 is the exception E6 records (below). That
literature reports that resampling a training split moves a classifier's predicted risk more than
its ranking, which the classifier panel tests. Over 24 simulated scenarios, random undersampling,
random oversampling and SMOTE left logistic regression's median AUROC no higher than no correction
and overestimated the minority class's risk, and shifting an uncorrected model's threshold gave a
similar balance of sensitivity and specificity [`vandengoorbergh-2022`][vandengoorbergh-2022].
Carriero et al.'s abstract reports that, over a variety of machine learning algorithms, models built
without a correction calibrated equal to or better than corrected ones in every simulated scenario
[`carriero-2025`][carriero-2025]. The plan's literature review cites Dal Pozzolo et al. for
undersampling biasing posterior probabilities while leaving the ranking largely intact
[`dalpozzolo-2015`][dalpozzolo-2015]. Across seven sampling methods, eight classifiers and 31
datasets, sampling changed AUC-PR significantly in 211 of 1,736 cases, 166 of them losses, and AUROC
in 173, 106 of them losses, and no sampling method was best on more datasets than no sampling
[`kim-2022`][kim-2022]. AUC-PR, like AUROC, reads only the order of the scores; Saito and
Rehmsmeier's abstract reports that on imbalanced data the precision-recall plot is the more
informative one [`saito-2015`][saito-2015].

E6 shows the pattern on one subgroup, with SMOTE [`chawla-2002`][chawla-2002] and ADASYN
[`he-2008`][he-2008] among its arms; [EVIDENCE.md](EVIDENCE.md#e6-diabetes-130-subgroup-removal)
holds every figure. At f = 0.985, `real` raises removed-cell recall over `none` by +0.4878 yet moves
AUC-PR on the cell's age-sex subgroup by 0.0025
([EVIDENCE.md](EVIDENCE.md#e6-diabetes-130-subgroup-removal)), so recall and ranking come apart.
`class-weight`, which scores nearly every test record positive at the cutoff, reads AUC-PR level
with `none` at both scopes. Every generator or resampler arm that adds rows of the cell recovers
some of its recall, and all but the Bayesian network read lower subgroup AUC-PR than `none`. At
f = 1.0, `bko-direct` is level with `none` on subgroup AUC-PR, the Bayesian network, which adds a
few rows on most holdouts, scores as `none`, and the other generator and resampler arms that add the
cell's rows read below `none`. Over the whole test
split the sign depends on the fill: at f = 1.0 `bko-direct` raises AUC-PR against `none` and `bko`
lowers it. On UCI, E4's larger fill targets raise every row-adding arm's recall and lower its
whole-split AUC-PR, while SMOTE's, ADASYN's and CTGAN's flagged-subgroup AUC-PR stays nearly level
([EVIDENCE.md](EVIDENCE.md#e4-fill-above-the-threshold)).

AUROC splits into each group's own AUROC and cross-group terms that rank one group's positives
against another group's negatives, and a monotone change to one group's scores keeps that group's
AUROC while it moves the cross-group terms [`kallus-2019`][kallus-2019]. Carrying that onto AUC-PR
is this project's inference, since the paper reads AUROC alone: raising one subgroup's scores alike
leaves its own AUC-PR unchanged but moves its records against every other subgroup's. On this
project's reading, whole-split AUC-PR then tracks how well a fill sets the subgroup's risk level,
and subgroup AUC-PR tracks whether the added rows reorder its records. The cited studies rebalance a
whole class and mostly read AUROC, so reading them onto one filled subgroup is also this project's
inference. Welvaars et al.'s abstract reports that, in one readmission study, resampling raised AUC
and precision for most algorithms and recall for all, while calibration overestimated positives
[`welvaars-2023`][welvaars-2023]. The classifier panel (ADR
[`research/classifier-panel`](adr/research/classifier-panel.md)) reads E6's arms under four more
models and their calibration.

## 8. Scope assessment

The brief (F15) assessed five bullets from the original project description. Its verdicts, and what
the 2-pager adopted:

| # | Bullet (verbatim) | Brief's verdict | In the plan |
|---|---|---|---|
| 1 | "Ontology-guided gap detection + LLM fills gaps to fit ontology" | Achievable; should be the core: OLS/BioPortal, a coverage/gap score, CLLM/GReaT generation, SHACL/ontology validation | Adopted with the 2-pager's methods: BKO gap detection, Messages API generation, validation against consistency checks ([section 3](#3-approach)) |
| 2 | "Ontology identifies gaps → go back to human expert for real data; generate questions for an LLM to fill gaps" | Achievable as a lightweight human-in-the-loop UI feature that doubles as the plausibility and model-collapse safeguard | Not adopted; [IDEAS.md](IDEAS.md#human-in-the-loop-expert-questions) |
| 3 | "Generate data to train a traditional ML model (disease detection)" | Achievable; should be the evaluation backbone (TSTR on UCI/Pima/Synthea) | Adopted: heart-disease prediction on UCI Heart Disease ([section 2](#2-intended-users-and-evaluation-setting)) |
| 4 | "Use synthetic data to distill an SLM that beats the original SOTA LLM on a narrow benchmark" | High risk; defer to a stretch goal or Track B | Not adopted; [IDEAS.md](IDEAS.md#slm-distillation-stretch-goal) |
| 5 | "Compare LLM synthesis vs autoencoders (VAE) vs GANs" | Achievable; should be the headline comparison, scored with SDMetrics/synthcity | Adopted: CTGAN, TVAE, SMOTE and ADASYN baselines with no-augmentation and class-weighting controls ([section 4](#4-success-criteria)), scored without synthcity ([section 5](#5-evaluation-contract)) |

The brief's recommended primary thrust is bullets 1, 2, 3 and 5; the 2-pager adopts 1, 3 and 5. The
verdicts assume three students and 10 weeks. The plan has four members and the [ROADMAP](ROADMAP.md)
schedule, so read each verdict as relative feasibility, not as a schedule.

## 9. Source reconciliation

Where the brief and the 2-pager disagree:

| Topic | Brief | 2-pager / repository | Governs |
|---|---|---|---|
| Formalism | BKO "not a standard term"; reframe as a coverage score; do not implement MEBN | Santos et al. BKOs; ungrounded statements are the formal gap definition, and a coverage-style score runs only as an ablation | 2-pager (the paper read; [R1](EVIDENCE.md#r1-mondo-and-do-fusion-reproduction) reproduces its fusion example); ADR [research/bko-formal-gap-definition](adr/research/bko-formal-gap-definition.md) |
| BKB library | Not discussed | "no public BKB library exists" | A public library, [pybkb](ARCHITECTURE.md#pybkb), exists; `bko.bkb` ports its core, ADR [research/bkb-layer-source](adr/research/bkb-layer-source.md) |
| Team and time | Three students, 10 weeks | Four members, the [ROADMAP](ROADMAP.md) schedule | 2-pager |
| Generator | CLLM [`seedat-2024`][seedat-2024] or fine-tuned GReaT [`borisov-2023`][borisov-2023] | Anthropic Messages API | 2-pager; alternatives in [IDEAS.md](IDEAS.md#cllm-and-great-as-alternative-generators) |
| Evaluation stack | SDMetrics + synthcity | scikit-learn, SDMetrics, synthcity | Repository drops synthcity (ADR [tooling/python-314-without-synthcity](adr/tooling/python-314-without-synthcity.md)) |
| Ontology access | OLS/BioPortal APIs (brief A2, F16) | owlready2, pronto, OLS4 | Repository reads pinned OBO releases with pronto and drops owlready2 and OLS4 (ADR [research/ontology-mapping-design](adr/research/ontology-mapping-design.md)) |
| Datasets | UCI, Pima, Synthea; MIMIC optional | 2-pager: UCI Heart Disease + Synthea; repository: also UCI Diabetes 130-US Hospitals [`clore-2014`][clore-2014], open and larger | 2-pager for E1's criteria; the repository adds Diabetes 130 for subgroup removal on real patients ([section 6](#limits-of-uci-heart-disease-as-the-evaluation-frame)) |
| SLM distillation | Stretch goal or Track B | Absent | [IDEAS.md](IDEAS.md#slm-distillation-stretch-goal) |
| Primary criterion | Not discussed | 2-pager: at least 10% over no augmentation, and matching CTGAN, TVAE and SMOTE; repository: separation from every comparison method, however small, judged on paired intervals ([section 4](#4-success-criteria)) | Repository; ADR [research/e1-evaluation-protocol](adr/research/e1-evaluation-protocol.md) |
| Privacy criterion | Not discussed | 2-pager: distance to closest record remains above the real-data baseline; repository: the added rows' 5th-percentile distance reaches a share `E1` fixes of the test rows', judged on paired intervals | Repository; ADR [research/e1-evaluation-protocol](adr/research/e1-evaluation-protocol.md) |
| Validity checks | Ontology or SHACL validation rejects implausible rows | 2-pager: at least 95% of generated records pass the model's consistency checks, which it does not list; repository: each field's number, codebook or range, and decimals, with no check from an ontology relationship | Repository; ADR [research/record-validation-design](adr/research/record-validation-design.md) |

## 10. Related work

**ELK.** ELK classifies ontologies in the EL family of description logics, on which OWL's EL
profile is based: from an ontology's axioms its calculus derives every entailed subsumption
between named classes and only those, in polynomial time, and reduces them to a taxonomy of direct
links [`kazakov-2014`][kazakov-2014]. A derived link is entailed or it is not; ELK weighs no source
and reads no frequency. The release the paper describes supports EL+bottom (EL with the bottom
concept), a fragment of OWL EL, not every OWL EL feature. A BKO keeps each source's statements and
gives conflicting inferences probabilities [`santos-2024`][santos-2024], and this pipeline adds
dataset fragments whose priors are concept shares (ADR
[`research/dataset-fragment-encoding`](adr/research/dataset-fragment-encoding.md)); ELK has no
counterpart to either. The two compose rather than compete. MONDO's release build classifies the
editors' source with ELK through ROBOT, relaxes and reduces the result, and builds `mondo.owl` and
`mondo.obo` from it [`mondo-2026`][mondo-2026], the `reason`, `relax` and `reduce` chain ROBOT's
paper describes for a release [`jackson-2019`][jackson-2019], so the MONDO hierarchy
`bko.ontology` loads is ELK's classification reduced to direct links. HPO's build configuration,
an Ontology Development Kit file [`matentzoglu-2022`][matentzoglu-2022], sets its release to use
no reasoner, and `hp.obo` is built from the unclassified primary release, while `hp.owl` is a
merge of the editors' file [`hpo-2026`][hpo-2026]. On this project's reading of those build
files, a link that follows from an HPO logical definition and that HPO's editors have not asserted
is absent from what `bko.ontology` loads. E9's hierarchy comparison classifies both pinned OWL
releases with ELK and lists each ancestor pair among the mapped concepts and their ancestors that
one hierarchy holds and the other lacks; an HPO pair can also come from the two files' different
builds (ADR [`research/logic-only-ablation`](adr/research/logic-only-ablation.md) defines E9;
[DEVELOPMENT.md](DEVELOPMENT.md#elk-comparison)).

**BioGPT.** Luo et al.'s abstract reports BioGPT as a generative Transformer language model
pre-trained on large-scale biomedical literature and evaluated on six biomedical language tasks,
among them end-to-end relation extraction and PubMedQA, with a case study generating descriptions
of biomedical terms [`luo-2022`][luo-2022]. It is trained to write literature text, not records
under a schema. To stand in as a generator it would need fine-tuning on the training records, as
the brief's fine-tuned GReaT would ([section 9](#9-source-reconciliation)); GReaT's abstract
reports sampling realistic tabular records from an auto-regressive generative language model
[`borisov-2023`][borisov-2023]. Fine-tuning moves records into model weights, whereas no BKO arm's
call carries a training record, though each sends training bounds and percentiles
([section 6](#6-datasets)). Nori et al.'s abstract reports that a
general model steered by prompting alone, GPT-4 with Medprompt, reached state-of-the-art results
on all nine MultiMedQA benchmarks, ahead of specialist models such as Med-PaLM 2
[`nori-2023`][nori-2023]. Those benchmarks measure medical question answering, not record
generation.

**Translator.** The Biomedical Data Translator Consortium set out to build a universal biomedical
data translator [`translator-2019`][translator-2019]. Its reports' abstracts describe a
federated knowledge-graph system that integrates clinical, genomic, pharmacological and other
sources under the Biolink Model's classes and predicates, and answers a user's query by retrieval
and inference, revealing evidence, provenance and confidence [`fecho-2022`][fecho-2022],
[`unni-2022`][unni-2022], [`fecho-2025`][fecho-2025]. BKO fusion also joins sources and keeps each
statement's source, but it fuses them into one BKB whose conflicting inferences carry
probabilities [`santos-2024`][santos-2024]. This pipeline also fuses in a dataset's records, so its
output is the ontology statements those records do not ground (ADR
[`research/bko-formal-gap-definition`](adr/research/bko-formal-gap-definition.md)). Translator
answers questions about biomedical entities; this pipeline asks what a training split lacks.

[borisov-2023]: https://arxiv.org/abs/2210.06280
[carriero-2025]: https://doi.org/10.1002/sim.10320
[chawla-2002]: https://doi.org/10.1613/jair.953
[clore-2014]: https://doi.org/10.24432/C5230J
[colorectal-nd]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11330441/
[dalpozzolo-2015]: https://doi.org/10.1109/SSCI.2015.33
[fecho-2022]: https://doi.org/10.1111/cts.13301
[fecho-2025]: https://doi.org/10.1111/cts.70284
[he-2008]: https://doi.org/10.1109/IJCNN.2008.4633969
[hpo-2026]: https://github.com/obophenotype/human-phenotype-ontology/tree/920d4eb3608b50a2c19ddb10290224ab35a759c5/src/ontology
[jackson-2019]: https://doi.org/10.1186/s12859-019-3002-3
[kallus-2019]: https://arxiv.org/abs/1902.05826
[kazakov-2014]: https://doi.org/10.1007/s10817-013-9296-3
[kim-2022]: https://doi.org/10.1371/journal.pone.0271260
[luo-2022]: https://doi.org/10.1093/bib/bbac409
[matentzoglu-2022]: https://doi.org/10.1093/database/baac087
[mondo-2026]: https://github.com/monarch-initiative/mondo/blob/9bfccbe9217cdfa6a225058bfa6e4bd55512b432/src/ontology/Makefile
[nori-2023]: https://arxiv.org/abs/2311.16452
[saito-2015]: https://doi.org/10.1371/journal.pone.0118432
[santos-2024]: https://doi.org/10.1371/journal.pone.0296864
[seedat-2024]: https://arxiv.org/abs/2312.12112
[strack-2014]: https://doi.org/10.1155/2014/781670
[translator-2019]: https://doi.org/10.1111/cts.12591
[unni-2022]: https://doi.org/10.1111/cts.13302
[vandengoorbergh-2022]: https://doi.org/10.1093/jamia/ocac093
[welvaars-2023]: https://doi.org/10.1093/jamiaopen/ooad033
[xu-2019]: https://arxiv.org/abs/1907.00503
