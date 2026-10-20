# Research plan

Sources: [2-pager](../papers/proposal-2-pager.pdf) sections 1 to 6; [research brief](../papers/research-brief.pdf) TL;DR, A1, B5, D10, D11, F15; [Santos et al.](../papers/santos-2024.pdf), read from the stored PDF; facts verified outside the PDFs (pybkb, synthcity resolution).

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

The two experiments are register entries
[E1](EVIDENCE.md#e1-uci-heart-disease-augmentation) and
[E2](EVIDENCE.md#e2-synthea-gap-detection-ablation).

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
through the Anthropic Messages API (2-pager section 4). Each candidate record is validated before
augmentation. The validity criterion counts the records that pass the model's consistency checks,
and the completeness criterion counts the detected gaps that close after fusion
([section 4](#4-success-criteria)). ADR
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

The plan uses UCI Heart Disease and Synthea, both openly available (2-pager sections 2 and 4). The
other rows are the brief's registry (D10), kept for reference.

| Dataset | Access | Description (brief D10) | In the plan |
|---|---|---|---|
| UCI Heart Disease (Cleveland) | Open | Classic 303-record binary/multiclass detection task; naturally small, which suits a low-data story; the evaluation frame is ADR [`research/uci-evaluation-frame`](adr/research/uci-evaluation-frame.md) | Evaluation setting ([section 2](#2-intended-users-and-evaluation-setting)) |
| Pima Indians Diabetes | Open | Classic small imbalanced binary task | No |
| UCI Diabetes 130-US Hospitals | Open | Larger; used in recent synthetic-data audit benchmarks | No |
| NHANES | Open | Openly available national survey data | No |
| Synthea / SyntheticMass | Open, no restrictions | ~1M synthetic patients, SNOMED-CT/RxNorm/LOINC-coded; models the 10 most common primary-care reasons and 10 highest-morbidity chronic conditions | Gap-detection ablation |
| MIMIC-IV demo, eICU demo | Open; no credentialing or CITI training | Demo subsets of the credentialed datasets | No |
| Full MIMIC-IV, MIMIC-IV-Note, MIMIC-IV-ED | Credentialed, on PhysioNet | Real EHR data | No |

**Synthea's caveats.** Synthea has documented fidelity flaws: per the JAMIA 2018 paper it
"incorrectly generated 20% of patients in the infant age group" and "failed to note diabetes onset
after age 52". The brief therefore treats it as a development and prototyping substrate, not a
gold-standard evaluation set. The plan uses it for the gap-detection ablation, where the ground
truth is the set of deliberately removed cases rather than clinical fidelity.

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

**Credentialed data stays local.** The pipeline sends each cell's description, plausibility ranges
and ontology guidance to the Anthropic Messages API, never a record. Data held under a DUA must
not reach that API or any other third-party service unless the DUA permits it; check its current
terms before any such run.

## 7. Augmentation versus resampling

Whether synthetic augmentation beats SMOTE, ADASYN or class weighting is genuinely mixed (brief
D11): no method dominates, and sometimes no resampling at all is optimal. The brief treats this as
an asset, since it makes the head-to-head comparison a real question rather than a foregone
conclusion. It also says the proposal must include SMOTE, ADASYN, class weighting and a
no-augmentation baseline, because omitting them is "the most common reviewer criticism". The plan
includes all four ([section 4](#4-success-criteria)). The studies behind the conclusion are in
[BACKGROUND.md](BACKGROUND.md#augmentation-versus-resampling).

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
| Datasets | UCI, Pima, Synthea; MIMIC optional | UCI Heart Disease + Synthea | 2-pager |
| SLM distillation | Stretch goal or Track B | Absent | [IDEAS.md](IDEAS.md#slm-distillation-stretch-goal) |
| Primary criterion | Not discussed | 2-pager: at least 10% over no augmentation, and matching CTGAN, TVAE and SMOTE; repository: separation from every comparison method, however small, judged on paired intervals ([section 4](#4-success-criteria)) | Repository; ADR [research/e1-evaluation-protocol](adr/research/e1-evaluation-protocol.md) |
| Privacy criterion | Not discussed | 2-pager: distance to closest record remains above the real-data baseline; repository: the added rows' 5th-percentile distance reaches a share `E1` fixes of the test rows', judged on paired intervals | Repository; ADR [research/e1-evaluation-protocol](adr/research/e1-evaluation-protocol.md) |
| Validity checks | Ontology or SHACL validation rejects implausible rows | 2-pager: at least 95% of generated records pass the model's consistency checks, which it does not list; repository: each field's number, codebook or range, and decimals, with no check from an ontology relationship | Repository; ADR [research/record-validation-design](adr/research/record-validation-design.md) |

[borisov-2023]: https://arxiv.org/abs/2210.06280
[colorectal-nd]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11330441/
[santos-2024]: https://doi.org/10.1371/journal.pone.0296864
[seedat-2024]: https://arxiv.org/abs/2312.12112
[xu-2019]: https://arxiv.org/abs/1907.00503
