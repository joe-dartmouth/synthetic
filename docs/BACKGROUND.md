# Background

Sources: [2-pager](../papers/proposal-2-pager.pdf) sections 1, 3 and 6; [research brief](../papers/research-brief.pdf) A1, A2, A3, B4, B6, D11; facts gathered outside the PDFs on 2026-10-13: Santos et al.'s metadata and a search summary of
its abstract, and pybkb's repository read through the GitHub API.

This is background for a teammate new to the area. The plan itself is in
[RESEARCH.md](RESEARCH.md); this file explains the ideas it rests on and the prior work around it.
Each section names the [literature.md](literature.md) group that holds its works. Every work there
is registered but not fetched, so each claim below is the citing source's claim. A sentence tagged
*(general background)* goes beyond the two sources.

## Bayesian Knowledge-driven Ontologies

### Ontologies and description logic

An ontology's content is classes and the relations between them (brief A1). The plan uses two
biomedical ontologies: MONDO, whose entries include disease subtypes, and HPO, whose entries are
phenotypes (brief A2 speaks of "an HPO phenotype or a MONDO disease subtype"). The first milestone
reproduces the fusion of MONDO with DO from Santos et al. (2-pager section 6).

Description logic is the formal logic behind the OWL web ontology language (general background); the
brief identifies OWL 2 DL with the description logic SROIQ(D). A description-logic reasoner derives
statements that an ontology implies without stating them (general background). Those implied
statements are what the plan's gap detector tests against the data.

### Bayesian Knowledge Bases

A Bayesian Knowledge Base (BKB) is the probabilistic half of a BKO. The sources support this much:

- **BKOs unite the two.** Santos et al. unite description-logic reasoning with BKBs, extending an
  earlier BKO framework (from a search summary of the paper's abstract; the paper is not fetched).
- **BKBs are built from fragments and fused.** By the same abstract summary, Santos et al. fuse
  conflicting ontologies probabilistically instead of deleting knowledge. pybkb's fusion entry point
  is `fuse(bkfs, reliabilities, ...)`: it takes a set of fragments and a reliability for each.
- **The plan builds three fragments.** MONDO and HPO each become a fragment, and the dataset becomes
  a second kind of fragment, weighted by observed frequency (2-pager section 3).
- **Reasoning runs over the fused result.** pybkb's one working inference
  ([ARCHITECTURE](ARCHITECTURE.md#pybkb)) gives each state of one target component its log joint
  mass with complete evidence, and `bko.bkb` ports it.

The two sources do not describe a BKB's internal structure; Santos et al. is the reference. Read
from pybkb's code, which `bko.bkb` ports (ADR
[`research/bkb-layer-source`](adr/research/bkb-layer-source.md)): an **I-node** is a pair of a
component and one of its states; an **S-node** gives a probability to one I-node, its head, given a
set of I-nodes, its tail. Fusion adds to each S-node's tail a source I-node naming the fragment that
holds it and gives each source a prior S-node weighted by its reliability; `bko.bkb.fusion` states
how the priors are normalized.

### Incompleteness as a property of the representation

The 2-pager contrasts BKOs with Bayesian networks: a BKO treats "incompleteness as a property of
the representation".

A Bayesian network gives every variable a conditional probability table over its parents, so it
assigns a probability to every combination of values, including combinations the data never shows;
the model's assumptions fill in what is missing (general background).

In a BKO, by the 2-pager's account (section 3), a statement can be implied by the ontology and still
have no support in the data. Such an ungrounded statement contributes nothing to the model's
distribution. The representation therefore keeps a record of where knowledge is missing instead of
filling it in, and that record is what the gap detector reads: the plan defines a dataset gap as an
ungrounded statement ([RESEARCH.md](RESEARCH.md#detect)).

A hypothetical illustration: if the ontology fragments imply that a disease subtype occurs with a
given chest-pain type, and the dataset fragment holds no record of that combination, the statement
is implied but ungrounded, so it is a gap.

### Fusion

The 2-pager uses "fusion" at two points. At Represent, the ontology and dataset fragments are fused
into one BKO. After generation, the completeness criterion counts the detected gaps that close
"after fusion", and the timeline pairs the step with generation ("LLM repair and fusion", section
6). The reading here is that validated records are fused back into the BKO and detection runs
again; the 2-pager does not spell this out.

Works: [Probabilistic ontologies and BKOs](literature.md#probabilistic-ontologies-and-bkos).

## Probabilistic ontologies: PR-OWL, MEBN and BayesOWL

The brief (A1) surveys the older formalisms for putting probabilities into ontologies. They are
background to BKOs, not alternatives the plan weighs. The brief's advice not to implement MEBN or
PR-OWL concerns those formalisms; [RESEARCH section 9](RESEARCH.md#9-source-reconciliation) records
why its claim that the BKO term is non-standard does not govern.

- **PR-OWL** (Probabilistic OWL) extends the OWL web ontology language with formal probabilistic
  semantics (Costa & Laskey, FOIS 2006). It is built on MEBN.
- **MEBN** (Multi-Entity Bayesian Networks) is a first-order probabilistic logic that unifies
  Bayesian probability with first-order logic. Knowledge is written as MEBN fragments (**MFrags**),
  organized into MEBN Theories (**MTheories**) that guarantee a unique joint probability
  distribution over random variables. A query produces a Situation-Specific Bayesian Network
  (**SSBN**).
- **PR-OWL 2**, PR-OWL's successor, is built on OWL 2 DL (description logic SROIQ(D)) and has high
  computational complexity and scalability problems.
- **BayesOWL** is an earlier approach that extends OWL with Bayesian networks as the reasoning
  layer.

**Tooling.** Multiple survey papers from 2008–2017 say PR-OWL and MEBN have historically "lacked a
software tool that implements their underlying concepts". The brief calls this a red flag for a
10-week implementation; it assumed three students and 10 weeks, where the plan has four members and
the
[ROADMAP](ROADMAP.md) schedule.

**What a distribution over an ontology means.** In these formalisms, putting "a probability
distribution over an ontology's classes and relations" means representing, for each class or
relation, uncertainty about whether an instance belongs to the class or whether the relation holds.
The uncertainty is held in conditional probability tables attached to ontology fragments (MFrags).

Works: [Probabilistic ontologies and BKOs](literature.md#probabilistic-ontologies-and-bkos).

## Gap detection

The brief (A2) lists four ways to find where a dataset is incomplete relative to an ontology. The
plan's formal detector is the BKO one above; these are the alternatives and the vocabulary around
them. [IDEAS.md](IDEAS.md#alternative-gap-detectors) records them as unscheduled options.

### Open-world and closed-world assumptions

OWL uses the open-world assumption: a statement that is absent is unknown rather than false (general
background; the brief says only that OWL uses it). SHACL uses the closed-world assumption: "what is
not stated is false". The brief calls this the key conceptual pivot: completeness checking requires
temporarily adopting the closed-world assumption, because under the open-world one a missing fact is
never a violation.

### SHACL and ShEx

SHACL (Shapes Constraint Language) and ShEx validate data against ontology-based constraints. SHACL
core can also compute data-quality metrics. The study "Is SHACL Suitable for Data Quality
Assessment?" defined SHACL shapes for the 69 data-quality metrics of Zaveri et al. (2012) and
reports: "Seven dimensions present a coverage between 50-90%: Timeliness, Understandability,
Consistency, Completeness, Syntactic Validity, Interoperability, Interpretability, and Versatility"
(the quotation names eight dimensions), and "SHACL can check the usage of classes, it cannot check
property usage."

The limit to remember: per the ABECTO paper, SHACL "cannot detect wrong values in a plausible
range". Validation finds missing or invalid fields, not subtle distributional gaps.

### Knowledge-graph completion

Knowledge-graph completion, or link prediction, infers edges missing from a knowledge graph.

### Coverage analysis

Coverage or enrichment analysis compares observed instance counts per ontology concept against
expected distributions, to flag under-represented regions of concept space. The brief calls it the
most student-friendly way to operationalize gap detection, and defines a gap as an ontology concept
(such as an HPO phenotype or a MONDO disease subtype) that is present in the ontology but has zero
or statistically under-represented support in the dataset relative to an ontology-weighted prior. It
suggests computing this with OLS/BioPortal APIs, pandas and a simple SHACL or count check, and (A1)
a Bayesian/Dirichlet smoothing prior over expected and observed counts.

The plan's count-based gaps are of this kind, and serve as the comparison the formal definition is
measured against ([RESEARCH.md](RESEARCH.md#two-kinds-of-gap-evaluated-separately)).

Works: [Gap detection and data quality](literature.md#gap-detection-and-data-quality).

## Knowledge-guided generation

The brief (A3) calls ontology- and knowledge-graph-guided synthetic data a live, credible research
area:

- **BioGAN** (Panaccione et al., Bioengineering 2025) is the first to put graph neural networks into
  a GAN for transcriptomic data. It uses gene-regulatory and co-expression networks to preserve
  biological plausibility and was validated on E. coli and human gene expression. Its finding: a
  priori biological knowledge improves both the quality and the utility of synthetic data.
- **GM-GAN and GR-GAN** ("Novel Knowledge-Guided Generative Methods for Synthetic Transcriptomic
  Data") inject biological knowledge graphs into a GAN; graph-informed models outperform
  unconditional ones.
- **KIPPS** ("Knowledge Infusion in Privacy Preserving Synthetic Data Generation") uses knowledge
  graphs to impose domain constraints so generated data obeys rules, framed around cybersecurity and
  healthcare.
- **OntoBook** turns medical ontology structure into pretraining signal: it generates random walks
  through ontology graphs and has an LLM rewrite them as synthetic "textbook" prose. The brief calls
  it a direct, citable precedent for generating data that fits an ontology.
- **Knowledge-graph resources** that ground such work: Hetionet, PrimeKG, RTX-KG2, the Monarch KG
  and the Biolink Model.

The brief's takeaway: the idea has 2024–2026 precedent, but mostly in high-dimensional
transcriptomics and omics rather than simple clinical tabular data. The team's contribution can
honestly be to "apply ontology-guided gap-filling to a clinical tabular detection task and
benchmark it against standard generators."

Works: [Knowledge-guided generation](literature.md#knowledge-guided-generation).

## Tabular generator families

The brief (B4) groups tabular synthetic-data generators into five families, all in the Synthetic
Data Vault or synthcity ecosystems:

| Family | Methods |
|---|---|
| GAN | CTGAN, CopulaGAN, CTAB-GAN / CTAB-GAN+, medGAN, ADS-GAN |
| VAE | TVAE |
| Statistical | Gaussian Copula, synthpop, Bayesian networks (CLBN/PrivBN) |
| Diffusion | TabDDPM, TabSyn |
| LLM | GReaT (Borisov et al.), Distill-GReaT, TabuLa (Zhao et al. 2025), TapTap, HARMONIC (NeurIPS 2024), CLLM (Curated LLM, Seedat et al., ICML 2024) |

CTGAN and TVAE come from Xu et al. (NeurIPS 2019). The plan's generator and baselines are in
[ARCHITECTURE.md](ARCHITECTURE.md#tools).

What the literature says about relative performance, per the brief:

- **GReaT** reports that it "decreases the discriminator performance by 16.2% over other competitive
  baselines", meaning its synthetic rows are harder to tell from real ones. It is expensive: the
  original paper reports training or fine-tuning time of ~9h10m for GReaT against 0:46 for TVAE and
  1:10 for CTGAN, and sampling time of 17s against 0.045s for CTGAN.
- **A GPT-4o zero-shot study** found GPT-4o outperformed CTGAN at preserving means, 95% CIs,
  bivariate correlations and privacy, without fine-tuning. Its failure mode: the LLM output violated
  "definitional ranges (e.g., negative measurements)".
- **CLLM** "empirically demonstrate[s] on 7 real-world datasets that CLLM enables superior
  downstream performance compared to 6 widely used tabular data generative models" (including
  CTGAN, TVAE, TabDDPM, SMOTE and GReaT), with gains "especially noticeable in the low-data regime
  (n<100)" and "for underrepresented subgroups". The brief calls this the single most important
  finding for the team; its code is public.
- **"Why LLMs Are Bad at Synthetic Table Generation"** argues that in-context LLM generation has
  real failure modes and proposes fine-tuning fixes. The brief recommends citing it for balance.
- **A prosumer-hardware benchmark** (CODATA Data Science Journal 2025) gives head-to-head
  CTGAN/TVAE/GReaT numbers on imbalance, augmentation, privacy (DCR) and ML utility, which makes it
  a methodological template.

The plan generates records by prompting a model through the Anthropic Messages API, which is
in-context generation. The GPT-4o study's range violations and the in-context failure modes above
are therefore the closest evidence for how the plan's generator can go wrong.

Works: [Tabular generators](literature.md#tabular-generators).

## Failure modes and mitigations

The brief (B6) names four failure modes:

| Failure mode | What happens | Mitigation the brief cites | Where the plan meets it |
|---|---|---|---|
| Model collapse | Training successively on synthetic output loses the distribution's tails and eventually degenerates (Shumailov et al., Nature 631:755–759, 2024) | Always mix real and synthetic data rather than training on synthetic alone; Gerstgrasser et al. (2024) show that accumulating real data alongside synthetic avoids collapse | Augmentation adds records to the real training data ([RESEARCH.md](RESEARCH.md#3-approach)) |
| Mode collapse in GANs | A GAN covers only part of the distribution | CTGAN's conditional vector and "packing", and CTAB-GAN+'s log-probability conditional vector, target this for imbalanced data | Applies to the CTGAN baseline |
| LLM hallucination | Clinically implausible records, such as negative lab values or impossible code combinations | Ontology or SHACL validation after generation; CLLM-style curation that filters generated rows by predictive confidence or aleatoric uncertainty | Numeric range constraints and validation of each record before augmentation ([RESEARCH.md](RESEARCH.md#3-approach)) |
| Memorization and leakage | Synthetic records copy real ones | Measure with DCR and membership inference, and report it as a feature rather than an afterthought | The privacy criterion ([RESEARCH.md](RESEARCH.md#4-success-criteria)) |

The brief says the model-collapse result justifies the team's "keep a human-expert / real-data loop"
design. The plan keeps the real-data half; the human-expert half is unscheduled
([IDEAS.md](IDEAS.md#human-in-the-loop-expert-questions)).

Works: [Evaluation and failure modes](literature.md#evaluation-and-failure-modes).

## Augmentation versus resampling

Whether generative augmentation beats SMOTE, ADASYN or class weighting is mixed. The brief (D11)
cites:

- **For GANs.** An omics-data study found GAN-based augmentation beat SMOTE and random oversampling
  on small high-dimensional data. A cancer-prediction study raised average ROC-AUC from 0.8276 to
  >0.9734 with GAN resampling.
- **For simpler baselines.** The survey "The use of generative adversarial networks to alleviate
  class imbalance in tabular data" (Journal of Big Data 2022) notes "83% of unique experiments among
  our researched papers boasted evaluation metrics for machine learning efficacy results where
  experimental methods outperformed baseline methods": GANs beat SMOTE and ADASYN in most cases but
  not all. It also reports cases where "GAN-based methods yielded classifiers which were no more
  performant… than SMOTE and ADASYN", and cases where no resampling at all was optimal. A
  GAN-versus-SMOTE/ADASYN study found the GAN "improved results… but did not perform better than
  SMOTE or ADASYN."
- **The nuanced consensus.** No method dominates. For high-dimensional, limited-sample,
  privacy-sensitive settings, classical copula and synthpop often give the best utility-privacy
  tradeoff (a pharmacogenetics study). CLLM claims superiority in the low-data regime, where SMOTE
  struggles: SMOTE interpolates between existing minority points and cannot extrapolate beyond them.

What the plan does with this is in [RESEARCH.md](RESEARCH.md#7-augmentation-versus-resampling).

Works: [Augmentation versus resampling](literature.md#augmentation-versus-resampling).

## Glossary

Expansions marked (general background) are not given in the sources.

| Term | Meaning |
|---|---|
| ADASYN | Adaptive Synthetic Sampling; an oversampling baseline (2-pager section 4) |
| AUC-PR | Area under the precision-recall curve (general background); preferred over accuracy for imbalanced tasks (brief B5) |
| BKB | Bayesian Knowledge Base: the probabilistic layer a BKO unites with description logic, built from fragments that can be fused |
| BKO | Bayesian Knowledge-driven Ontology (Santos et al.): description logic plus BKBs, treating incompleteness as a property of the representation |
| CLLM | Curated LLM (Seedat et al.): prompt-based LLM generation with curation of the generated rows |
| Closed-world assumption | What is not stated is false; SHACL's assumption |
| Count-based gap | A subgroup-label combination under the record-count threshold of [RESEARCH.md](RESEARCH.md#2-intended-users-and-evaluation-setting) |
| CTGAN | Conditional Tabular GAN (Xu et al.); a baseline |
| DCR | Distance to Closest Record; the privacy criterion's metric |
| Description logic | The formal logic behind OWL (general background); OWL 2 DL corresponds to SROIQ(D) (brief A1) |
| DO | The ontology Santos et al. fuse with MONDO (2-pager section 6); Disease Ontology (general background) |
| Fragment | Part of a BKB built from one source, such as an ontology or the dataset |
| Fusion | Combining BKB fragments into one, weighted by each source's reliability |
| GReaT | An LLM-based tabular generator, fine-tuned (Borisov et al.) |
| HPO | A phenotype ontology the plan loads as a fragment; Human Phenotype Ontology (general background) |
| Knowledge-graph completion | Inferring missing edges in a knowledge graph (link prediction) |
| MEBN | Multi-Entity Bayesian Networks: a first-order probabilistic logic |
| MFrag | A MEBN fragment, holding probabilistic knowledge |
| MONDO | A disease ontology the plan loads as a fragment |
| MTheory | A MEBN Theory: MFrags that together define a unique joint distribution |
| NNDR | Nearest-Neighbor Distance Ratio; a privacy metric |
| OLS4 | The ontology service the 2-pager names; the brief calls it OLS and pairs it with BioPortal. Ontology Lookup Service (general background) |
| Open-world assumption | An absent statement is unknown, not false; OWL's assumption |
| pMSE | Propensity-score mean squared error, a fidelity metric with range [0, 0.25) |
| PR-OWL, PR-OWL 2 | Probabilistic extensions of OWL built on MEBN |
| SHACL | Shapes Constraint Language: closed-world validation of data against shapes |
| ShEx | A shape-based validation language like SHACL; Shape Expressions (general background) |
| SMOTE | Synthetic Minority Over-sampling Technique; interpolates between existing minority examples |
| SSBN | Situation-Specific Bayesian Network: what a MEBN query produces |
| TSTR, TRTR | Train-on-Synthetic-Test-on-Real and Train-on-Real-Test-on-Real: the utility comparison |
| TRTS | Train-on-Real-Test-on-Synthetic (general background) |
| TVAE | Tabular Variational AutoEncoder (Xu et al.); a baseline |
| Ungrounded statement | A statement the ontology implies and the data does not support; the plan's formal dataset gap |
