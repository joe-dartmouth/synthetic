# Ideas

Sources: [RESEARCH.md](RESEARCH.md), the plan of record, and the works [literature.md](literature.md) registers.

Ideas outside the plan. Each records what it is, why it is not in the plan, and its sources.
Adopting one means changing [RESEARCH.md](RESEARCH.md) first; an entry that points at a task unit
is scheduled through that unit, which makes the change in its own PR.

## SLM distillation stretch goal

**What it is.** If the team is ahead of schedule, distill a small model on the best synthetic data
and show that it narrows the gap to a frontier LLM on one narrow, self-defined task (brief F15,
bullet 4 of the original project description). The study fine-tunes a language model as a
classifier of each arm's records (ADR
[`research/slm-comparison-design`](adr/research/slm-comparison-design.md)); what this entry holds
is what that design leaves out: a distilled model compared with a frontier one, and the medical
benchmarks below.

**Why it is not in the plan.** The 2-pager has no distillation thrust. The brief rates it high risk:
the tabular comparison and the distillation each "is a full project", and the engineering and
evaluation load is too much on top of the core. That estimate assumed three students and 10 weeks,
with "~6 usable weeks" for this work; the plan has four members and the [ROADMAP](ROADMAP.md)
schedule. The brief also notes that frontier models score ~95% on MedQA, so beating a frontier LLM
on a public medical benchmark is effectively off the table for a student team; any claim should be
scoped to a narrow, self-defined slice with a cost and efficiency framing.

**The evidence that it can work** (brief C7):

- **Meerkat-7B/8B** [`kim-2025`][kim-2025] was fine-tuned on chain-of-thought extracted from 18
  medical textbooks (441K examples). Meerkat-7B scored 70.6% (single) / 77.1% (ensemble) on MedQA,
  the first 7B model to pass the USMLE 60% threshold, beating MediTron-70B's 70.2% and GPT-3.5.
  Meerkat-70B (82.6%) surpassed GPT-4 by 1.2%. Fine-tuning used ~28× less GPU time than continued
  pretraining.
- **Mistral-7B QLoRA on biomedical claim verification**
  [`claimverification-2026`][claimverification-2026] reached 88.4% macro-F1 on SciFact and 65.2% on
  HealthVer, beating GPT-4o (85.6%, 53.2%) and GPT-5 (77.9%, 42.4%) at 44.5× lower cost, using just
  1,008 training examples. The brief calls it the single best citation for a small model beating a
  frontier one at low cost with few examples.
- **phi-1** [`gunasekar-2023`][gunasekar-2023] (1.3B) hit 50.6% pass@1 on HumanEval, trained on 6B
  tokens of textbook-quality data plus 1B tokens of GPT-3.5-generated synthetic data: the
  foundational result that synthetic data lets small models punch above their weight.
- **MEDITRON-70B** [`meditron-2023`][meditron-2023] and BioMistral-7B are the brief's domain
  baselines.

**The sobering evidence** (brief C7):

- On broad clinical tasks, open models still trail frontier ones. "Towards evaluating… versatile
  LLMs for medicine" (npj Digital Medicine 2024) shows MEDITRON at 68.27 against GPT-4's 84.73 and
  Claude-3.5's 92.93 on SEER treatment planning.
- A construct-validity study [`construct-2025`][construct-2025] shows MedQA scores do not correlate
  well with real-world EHR performance, so beating a benchmark does not establish clinical utility.
- The brief's assessment: the result is real but was achieved by well-resourced labs.

**Benchmarks** (brief C8):

| Benchmark | Size and split | Notes |
|---|---|---|
| MedQA (USMLE) | 1,273-question test set; the 4-option English variant is widely used; training set GBaker/MedQA-USMLE-4-options (10,178 examples) on Hugging Face | Easy to run; passing threshold 60% |
| MedMCQA | ~194K questions (AIIMS/NEET); evaluate on the 4,183-question validation set | Test answer keys are not public |
| PubMedQA | 500 expert-labeled yes/no/maybe items with abstract context | Easy |
| MMLU clinical subsets | Anatomy, clinical knowledge, college biology/medicine, medical genetics, professional medicine; no training set | Good for cross-domain transfer testing |
| BioASQ (factoid QA), HealthBench, MedXpertQA | Not given | More specialized or heavier |

The harnesses are EleutherAI lm-evaluation-harness (the de facto standard) and HELM.

**Frontier reference scores.** The brief reports, as of 2026, that frontier models score ~95% on
MedQA: o4-mini-high 95.2%, Gemini 2.5 Pro 94.6% and DeepSeek R1 92.1%, per the
LayerLens/pricepertoken leaderboard of Oct 1 2026. The best sub-10B open models with fine-tuning
reach ~70–77% (Meerkat-7B). Model names and leaderboard scores move quickly; these are the brief's
snapshot and have not been checked. The realistic win, per the brief, is not beating the frontier
model on MedQA but showing that "on a narrow, self-defined slice, a fine-tuned small model closes
most of the gap to a frontier model at a fraction of cost."

**Fine-tuning tooling and cost** (brief C9):

- **Methods**: LoRA / QLoRA (4-bit NF4), with Unsloth (fastest, lowest VRAM), Axolotl (most
  configurable) and Hugging Face TRL/PEFT.
- **Base models** (0.5B–8B): Qwen (2.5/3), Llama 3.1-8B, Gemma, Phi, SmolLM. Open models ship 128K
  context and are "competitive with generic big ones on narrow tasks."
- **Compute** (the brief's 2026 figures): a 7–8B QLoRA fine-tune needs only 6–10 GB VRAM and runs in
  2–4 h on an A100 or 6–8 h on an RTX 4090. It costs ~$3–10 on a rented A100 or under ~$3 on a
  RunPod RTX 4090 (~$0.34/hr) (editor's reconstruction of a garbled line in the source). On a free
  Colab T4, small fine-tunes (1k–10k examples, 7B) finish in 15–60 minutes with Unsloth.
- **The brief's takeaway**: compute is not the bottleneck; data quality, evaluation rigor and scope
  are.

**Sources.** Brief TL;DR, C7, C8, C9, F15 (bullet 4 and the stretch-goal recommendation), Caveats.
Works: [Small-model distillation and benchmarks](literature.md#small-model-distillation-and-benchmarks).

ADR [`research/slm-comparison-design`](adr/research/slm-comparison-design.md) fixes the narrowed
form the study runs.

## Track B agentic gap-filling copilot

**What it is.** The brief's Track B follow-on: frame the SLM distillation plus an agentic
"data-gap-filling copilot" as the internal-workflow extension of the project.

**Why it is not in the plan.** The 2-pager names no Track B work; the brief frames the project as
Track A, which requires a product. The brief gives no design for the copilot beyond that framing.

**Sources.** Brief TL;DR, F15.

## Human-in-the-loop expert questions

**What it is.** When the ontology identifies gaps, go back to a human expert for real data, and
generate questions for an LLM to fill the gaps. The original bullet reads: "Ontology identifies gaps
→ go back to human expert for real data; generate questions for an LLM to fill gaps". The brief
rates it achievable as a lightweight UI feature that doubles as the safeguard for clinical
plausibility and against model collapse. Its Demo-Day flow shows it as an optional tab where, for
the thinnest gaps, the app drafts questions for a human expert or an LLM to supply real values
([ARCHITECTURE.md](ARCHITECTURE.md#demo-day-flow)).

**Why it is not in the plan.** The 2-pager does not include it. The plan's safeguards are
validation of each record before augmentation and adding generated records to real training data
([BACKGROUND.md](BACKGROUND.md#failure-modes-and-mitigations)). Clinicians are secondary users,
and clinician review appears only in the concept-to-field mapping
([RESEARCH.md](RESEARCH.md#represent)).

**Sources.** Brief B6, F15 (bullet 2), F16 (step 4); 2-pager sections 2 and 3.

## CLLM and GReaT as alternative generators

**What it is.** The brief's recommended generators: prompt-based CLLM [`seedat-2024`][seedat-2024]
or fine-tuned GReaT [`borisov-2023`][borisov-2023]. CLLM shows a model real training records as
in-context examples and drops each generated row that a classifier fit on the training records
gives both low confidence in its label and low aleatoric uncertainty: a conjunction that acts
almost as a confidence threshold (ADR
[`research/cllm-arm-design`](adr/research/cllm-arm-design.md), Rationale). The brief cites its
gains in the low-data regime (n<100) and for underrepresented subgroups, and its code is public.
GReaT fine-tunes a language model on the table and is costly to train (~9h10m in its paper,
against 0:46 for TVAE). REaLTabFormer [`solatorio-2023`][solatorio-2023] also trains a transformer
on the table.

**Status.** CLLM is built as two comparison arms, `cllm` with its curation and `cllm-uncurated`
without it (ADR [`research/cllm-arm-design`](adr/research/cllm-arm-design.md)); their runs and
measures are [EVIDENCE.md](EVIDENCE.md#b3-cllm-arms) B3. GReaT and
REaLTabFormer are ruled out: each trains a model on the table, a training cost the brief's GReaT
figures show and the plan's other arms do not pay, and the CLLM arms already ask an LLM for
records without the ontology.

A curated `bko` arm, `bko-curated`, keeps each cell's highest-scoring records by their likelihood
under the fused BKB in place of a classifier's confidence. Its gate, fixed in ADR
[`research/bkb-candidate-curation`](adr/research/bkb-candidate-curation.md), did not separate
fidelity on UCI and separates it on Diabetes 130, where the arm runs and G7 reports its
effect ([EVIDENCE.md](EVIDENCE.md#g7-bkb-likelihood-curation)).

**Why it is not in E1's comparison set.** The 2-pager generates through the Anthropic Messages
API, guided by ontology relationships, and its baselines are CTGAN, TVAE, SMOTE and ADASYN, so the
CLLM arms run in B3's named runs, `b3-uci` and `b3-synthea`, with no verdict rather than among
E1's comparators. For balance, the brief also cites "Why LLMs Are Bad at Synthetic Table Generation"
[`llms-2024`][llms-2024]. Its experiments fine-tune; the in-context generation the Messages API
approach uses appears in it only through a survey it cites.

**Sources.** Brief TL;DR, B4, B6, F15 (bullet 1); 2-pager section 4. Details:
[BACKGROUND.md](BACKGROUND.md#tabular-generator-families).

## Diffusion generators: TabSyn

**What it is.** TabSyn [`zhang-2024`][zhang-2024] runs denoising diffusion in a latent space a
variational autoencoder learns over tabular rows. TabDDPM, which diffuses the rows themselves, is
built as a baseline (ADR [research/diffusion-baseline](adr/research/diffusion-baseline.md)). The
brief lists both as its diffusion family
([BACKGROUND.md](BACKGROUND.md#tabular-generator-families)).

**Why it is not in the plan.** Deferred. synthcity does not resolve on Python 3.14 (ADR
[tooling/python-314-without-synthcity](adr/tooling/python-314-without-synthcity.md)), so TabSyn
needs a standalone implementation, and a second diffusion model beside TabDDPM adds a VAE to train
on the same split. At UCI's about 240 training records the deep generators get few optimizer
steps (ADR
[research/baseline-augmentation-design](adr/research/baseline-augmentation-design.md)); the larger
Synthea and Diabetes splits are where it would be worth a unit.

**Sources.** Brief B4.

## Tree-based generators: ARF and ForestDiffusion

**What it is.** Generators built from tree ensembles: adversarial random forests
[`watson-2023`][watson-2023] alternate a forest that tells real rows from synthetic ones with
sampling from the forest's leaves, and ForestDiffusion
[`jolicoeur-martineau-2023`][jolicoeur-martineau-2023] trains gradient-boosted trees as the
diffusion or flow model.

**Why it is not in the plan.** Deferred. B1 runs sequential CART synthesis, the simplest
tree-based synthesizer, beside `bko` ([`docs/EVIDENCE.md`](EVIDENCE.md#b1-classical-baselines));
ARF and ForestDiffusion are worth a unit if it reads well there. Each resolves on Python 3.14 under a scratch copy of the lock (checked
2026-10-28); neither has been installed or run.

**Sources.** None in the brief or the 2-pager.

## Alternative gap detectors

**What it is.** The brief's gap-detection threads other than BKO instantiation:

- **SHACL validation** under the closed-world assumption [`shacl-2025`][shacl-2025], which finds
  missing or invalid fields but not subtle distributional gaps.
- **Knowledge-graph completion** (link prediction) to infer missing edges.

**Why it is not in the plan.** The plan's formal gap is an ungrounded BKO statement (ADR
[research/bko-formal-gap-definition](adr/research/bko-formal-gap-definition.md)); a coverage-style
score, in place of the brief's Dirichlet one, runs only as an ablation beside it
([RESEARCH.md](RESEARCH.md#detect)). SHACL could
still serve the Validate stage, as the brief suggests for post-generation checks (B6, F16).

**Sources.** Brief A1, A2, B6, F16. Details:
[BACKGROUND.md](BACKGROUND.md#gap-detection).

## Exact belief updating over a BKB

**What it is.** The probability of each state of a target component given partial evidence: sum
over the components the evidence leaves open, over acyclic inferences only, then normalize across
the target's states. [pybkb](ARCHITECTURE.md#pybkb) implements no such update to port.

**Why it is not in the plan.** `bko.bkb` ports the one inference pybkb implements, the log joint
mass of each state of a target under complete evidence (ADR
[research/bkb-layer-source](adr/research/bkb-layer-source.md)). No planned unit reads a posterior:
gap detection lists ungrounded statements and fusion repair re-runs detection after fusing records
back. Adopting it means changing [RESEARCH.md](RESEARCH.md) and adding a unit whose acceptance
reads a posterior, with a hand-worked case to check it against, since pybkb has none.

**Sources.** pybkb at the commit the ADR names.

## Opus 5.5 as a generator

**What it is.** The generator comparison's runs at Opus 5.5: the baselines and the `bko` arm at the
judged configuration with `SETTINGS.model` alone changed, as G6 runs Sonnet 5.5
([EVIDENCE.md](EVIDENCE.md#g6-generator-model)).

**Why it is not in the plan.** Not run in G6, by the project's choice: at the judged run's tokens per
record asked its estimate is about twice Sonnet's, and G6 reads the model axis through Sonnet and
three on-device models. Adopting it means a named run with its own spend ceiling, whose commit is
the sign-off.

**Sources.** None in the brief or the 2-pager.

## A full thinking-on local run

**What it is.** G6's 100-holdout runs of Gemma 4 26B-A4B and Nemotron 3 Nano 30B-A3B with
thinking on, at their `-thinking` model ids, read beside G6's thinking-off runs and `e1-5` on every
E1 measure ([EVIDENCE.md](EVIDENCE.md#g6-generator-model)).

**Why it is not in the plan.** G6's thinking-cost runs measure thinking on over holdouts 0 to 4
alone, for tokens and FLOPs (ADR
[research/local-generator-backend](adr/research/local-generator-backend.md)). Such a run first has
to decide how a thinking reply fits `max_tokens`: in `g6-12` most of Nemotron's replies stopped
at `max_tokens` with no record decoded, and the fill asked those records again, so its projected
compute depends on that choice. Either fix changes what a call sends: raise `max_tokens` for a
`-thinking` id, or budget its reasoning so the answer fits. Adopting it means changing
[RESEARCH.md](RESEARCH.md) and adding a named run for each model at the chosen setting.

**Sources.** None in the brief or the 2-pager.

## A rewrite arm that moves records

**What it is.** A redesign of E10's language-model rewrite of `bko-direct`'s 19 unmapped Diabetes
fields, so that its added rows sit farther from training records than `bko-direct`'s while
keeping the label-1 distributions. With per-call label-group quotas the rewrite reached column
shapes of 0.977 to 0.978 against the label-1 training rows, but its added-row `dcr-p5` exceeded
`bko-direct`'s on 1 of 4 re-pilot holdouts and its NNDR p5 was lower on all 4 (E10's Read against
in [EVIDENCE.md](EVIDENCE.md#e10-identifier-perturbation)). A likely reason, not yet tested, is
that matching the label group's marginals draws the rows toward its dense region.

**Why it is not in the plan.** The re-pilot failed G6, so the rewrite returns to design (ADR
[research/identifier-perturbation](adr/research/identifier-perturbation.md), Reversal Conditions),
and E10's full runs are PRAM only. Adopting it means a new prompt or arm recorded in that ADR and a
new paid re-pilot that passes its gate before any full run.

**Sources.** None in the brief or the 2-pager.

## Downstream runs at `profile` guidance

**What it is.** Every run of record whose BKO arms generate through `SETTINGS` reruns with
`SETTINGS.guidance` at `profile`, the level G3's rule picks when each level is read against its
`ontology` replicate ([EVIDENCE.md](EVIDENCE.md#g3-guidance-level)): E1, E3, E4 and E5 on UCI
Heart Disease and Synthea, E6 and E10 on Diabetes 130, and the readings of those runs, E7's and
E8's among them. Each is at `summary` now.

**Why it is not in the plan.** G3's rerun keeps `summary`, a stated departure from its rule: beside
E1's run of record `profile` and `summary` cannot be told apart, and every run of record is at
`summary`. Adopting `profile` changes what each call sends, so, per this file's header, it means
changing [RESEARCH.md](RESEARCH.md) first, then `SETTINGS`, a new tag and paid reruns of each named
run above under its own spend ceiling.

**Sources.** None in the brief or the 2-pager.

## Candidate datasets and ontologies outside healthcare

**What it is.** A research pass of 2026-11-22 over tabular datasets and open hierarchies outside
medicine, read for whether a BKO could find coverage gaps in them as it does in clinical data.
Every count, and every terms summary not quoted with a citation, is the research pass's reading
of 2026-11-22 and was not re-checked; only the quoted terms were re-read from their pages. A
candidate needs a binary outcome, two cell axes, a machine-readable coded hierarchy and terms that
permit the use. Fit: as is means person-level
rows with a numeric age and a binary sex; gen means the cell axes must be generalised; weak
carries its reason. "Terms not yet read" means the page was not read on 2026-11-22 and nothing
quoted from it is checked. UNVERIFIED means the figure or field was not confirmed.

Transport:

- FARS 2023 person file: 92,768 rows by 126 columns; outcome `INJ_SEVNAME` fatal, 44.2%; cells
  `AGE` by `SEXNAME`; FARS and MMUCC codes, hierarchy UNVERIFIED. Its data.gov record
  [`fars-nd`][fars-nd] gives `accessLevel` as "public" and states no licence (read 2026-11-22).
  Fit: as is.
- CRSS 2023 person file: 122,388 rows by 112 columns, a weighted sample; outcome `INJ_SEV`, fatal
  (K) 1,237 and serious (A) 6,633; cells `AGE` by `SEX`; codes as FARS. Terms not yet read. Fit:
  as is.
- NTSB aviation, ASRS and FRA rail: rows, cells and terms UNVERIFIED; terms not yet read. Not
  checked.

Cyber and security incidents:

- GitHub Advisory Database (rows UNVERIFIED; ecosystem by severity; CWE), UNSW-NB15 (175,341
  train and 82,332 test; protocol by service; the research pass read its terms as academic
  research only), EPSS, MITRE ATT&CK and
  CAPEC (a tactic, technique and sub-technique hierarchy, a knowledge source), CIC-IDS2017 and
  NSL-KDD (no hierarchy; the research pass read NSL-KDD's page as saying it is no longer
  available). Terms not yet read.
  Fit: gen, weak or knowledge source.

Conflict, defence and procurement:

- GDELT 2.0 Events: over 250 million events; outcome material conflict (QuadClass 4), the rule
  to be fixed; cells actor country by actor type; the CAMEO hierarchy, QuadClass over root, base
  and event code. Its terms page [`gdelt-nd`][gdelt-nd] says "all datasets released by the GDELT
  Project are available for unlimited and unrestricted use" and requires "a citation to the
  GDELT Project and a link to this website" (read 2026-11-22). Fit: gen; a fit for defence and
  security.
- UCDP GED 26.1 (fatalities over a threshold; region by violence type; a shallow hierarchy),
  USAspending contract awards (single bid, `NumberOfOffersReceived` of 1; agency by award band;
  NAICS and Product or Service Code), EU TED CSV 2006 to 2023 (`NUMBER_OFFERS` of 1 or
  `CANCELLED`; country by contract type; CPV, whose old and new versions match many-to-many).
  UCDP's terms not yet read; USAspending's and TED's were read on 2026-11-25 by the ADR named
  below; row counts UNVERIFIED. Fit: gen.
- ACLED, the Global Terrorism Database, SIPRI arms transfers, ICEWS and Correlates of War: the
  research pass read terms that exclude or weaken them (on its reading, ACLED forbids training
  use without written consent, GTD forbids redistribution, SIPRI needs a licence for commercial
  use). Terms
  not yet read. No record-level DoD, Air Force or Naval mishap, maintenance or personnel file was
  found.

Product safety and justice:

- NEISS 2024: 361,672 rows by 25 columns and 10.5% admitted, both UNVERIFIED; outcome
  disposition 4, treated and admitted for hospitalisation; cells `Age` by `Sex`; flat product list
  with diagnosis and body-part codes. Its data.gov record carries the U.S. government public
  domain label, read on 2026-11-25 by the ADR named below. Fit: as is.
- Stanford Open Policing (88 jurisdictions, 229,860,452 stops; arrest or search; driver age by
  sex; violation text without a code system), COMPAS two-year (7,214 by 51; `two_year_recid`,
  45.1%; `age` by `sex`; the research pass read the repository as having no licence) and NIBRS
  (UNVERIFIED). NIBRS's data.gov record carries the U.S. government public domain label, read on
  2026-11-25 by the ADR named below; the others' terms not yet read. Fit: as is, with a weak
  hierarchy or a licence risk.

Finance, labour, technology, education:

- SBA 7(a) and 504 FOIA loans (charged off; status codes UNVERIFIED; state by term band; NAICS),
  CFPB Consumer Complaint Database (19,371,701 rows; monetary relief, rate UNVERIFIED; product by
  state; product, sub-product and issue hierarchies). SBA's data.gov record carries the U.S.
  government public domain label, read on 2026-11-25 by the ADR named below; CFPB's terms not
  yet read; CFPB's row count UNVERIFIED. Fit: gen.
- UCI Default of Credit Card Clients (30,000 by 23; default; age by sex; no hierarchy), the
  Polish and Taiwanese bankruptcy sets and FDIC failed banks (weak), FIBO, SEC EDGAR with
  US-GAAP XBRL, HMDA, Lending Club and Kaggle sets (a knowledge source, or not checked). Terms not
  yet read.
- UCI Census-Income KDD (299,285 by 41; income over $50K; `AAGE` by `ASEX`; detailed industry and
  occupation codes under major codes), O*NET 31.0 and ESCO over ACS PUMS or KDD (SOC-based; a
  knowledge source). Terms not yet read. Fit: as is.
- Backblaze Drive Stats (354,415 drives in Q2 2026; failure; manufacturer by capacity band;
  model under manufacturer), OSV.dev, PatentsView with CPC, NASA MDP and PROMISE, OULAD, UCI
  Dropout and Academic Success (4,424 by 36) and UCI Student Performance (649 by 30; no
  hierarchy). Terms not yet read. Fit: gen or weak.
- Open ontologies outside clinical medicine: ENVO, FoodOn, AGRO, PO, ChEBI and GO. No tabular
  dataset with a binary outcome was found for them; the datasets are unknown.

**Ranked.** ADR
[`research/non-health-dataset-selection`](adr/research/non-health-dataset-selection.md) fixes the
rule a non-health frame meets and ranks every candidate here under it. It chooses NVD CVE records
with the KEV label over CWE, and the VERIS Community Database over the VERIS schema, which have
left this entry; SBA 7(a) and 504 and TED over CPV are its runners-up. UCI Census-Income KDD, a
runner-up of ADR
[`research/confirmatory-dataset-selection`](adr/research/confirmatory-dataset-selection.md), is
excluded under its condition 1, since its income label is a threshold on an amount. It re-read
FARS, CRSS, SBA 7(a) and 504, NIBRS, USAspending, TED, NEISS and GDELT from their own pages on
2026-11-25 and 2026-11-26, and its quotes and verdicts govern where this entry says terms not yet
read, UNVERIFIED or a fit.

**What the search did not reach.** AGROVOC, UNSPSC, GeoNames, COICOP, ISIC and HS, legal
ontologies, NCES public-use files, OSHA and FEMA claims.

**Why it is not in the plan.** The study's non-health families are ACS PUMS and the two frames
that ADR chooses ([RESEARCH.md](RESEARCH.md)). Every candidate listed above, outside What the
search did not reach, is a runner-up or excluded under its rule; a runner-up joins the plan when
a re-read quotes what it lacks. What the search did not reach is unranked.

**Sources.** [`fars-nd`][fars-nd] and [`gdelt-nd`][gdelt-nd], each read from its primary page on
2026-11-22; none in the brief or the 2-pager.

## A knowledge-only draw with an external prevalence table

**What it is.** A variant of the knowledge-only arm `bko-kb` (ADR `research/matched-knowledge-controls`) that draws
each concept's presence at a prevalence read from a cited external table, such as a published
survey's, in place of uniform over the assignments K allows. The draw would then hold how common
each concept is as well as which combinations K allows, and still copy no training value.

**Why it is not in the plan.** An external prevalence table is a knowledge source beside the
mapping table and its pinned releases, which are what the matched controls isolate as K, so
`bko-kb` is registered uniform, as a floor. Adopting it means a cited table registered in
[literature.md](literature.md), a new arm name and a change to [RESEARCH.md](RESEARCH.md).

**Sources.** None in the brief or the 2-pager.

## A sweep of `bko-direct`'s group weight

**What it is.** `bko-direct` mixes a cell's own concept share with its label group's at the fixed
weight `GROUP_WEIGHT`, 20 (`bko.gaps.shares`). A sweep runs the arm at several weights, or at a
weight fitted by empirical Bayes as the planned `eb-marginals` control fits one. Weights fitted on
holdout 0 of E6's frame sit mostly above 20, so the fixed weight may pool less than the data
support.

**Why it is not in the plan.** A change to `GROUP_WEIGHT` changes `bko-direct`'s draw, which
`DRAW_DIGEST` (`tests/test_generate_direct.py`) pins, and moves the pooled control arms with it.
Adopting it means a weight field on the draw spec under new arm names, and a change to
[RESEARCH.md](RESEARCH.md).

**Sources.** None in the brief or the 2-pager.

## The matched controls on UCI Heart Disease

**What it is.** E19's matched knowledge controls (ADR `research/matched-knowledge-controls`) run on UCI Heart Disease,
whose mapping groups span several fields and whose numeric fields carry cutoffs. On E6's frame
every mapping group is a single field with no nesting, so at a cell of no record `bko-direct` and
`bko-nomap` draw one distribution (ADR
[`research/no-mapping-ablation`](adr/research/no-mapping-ablation.md), Consequences); on UCI Heart
Disease K's grouping and cutoffs could separate them there.

**Why it is not in the plan.** E19 registers the controls on E6's frame, the direct-method study's
development frame. Adopting it means new runs on `uci` and a change to [RESEARCH.md](RESEARCH.md).

**Sources.** None in the brief or the 2-pager.

[borisov-2023]: https://arxiv.org/abs/2210.06280
[claimverification-2026]: https://arxiv.org/abs/2606.12854
[construct-2025]: https://arxiv.org/abs/2503.10694
[fars-nd]: https://catalog.data.gov/dataset/fatality-analysis-reporting-system-fars
[gdelt-nd]: https://www.gdeltproject.org/about.html
[gunasekar-2023]: https://arxiv.org/abs/2306.11644
[jolicoeur-martineau-2023]: https://arxiv.org/abs/2309.09968
[kim-2025]: https://doi.org/10.1038/s41746-025-01653-8
[llms-2024]: https://arxiv.org/abs/2406.14541
[meditron-2023]: https://arxiv.org/abs/2311.16079
[seedat-2024]: https://arxiv.org/abs/2312.12112
[shacl-2025]: https://arxiv.org/abs/2507.22305
[solatorio-2023]: https://arxiv.org/abs/2302.02041
[watson-2023]: https://arxiv.org/abs/2205.09435
[zhang-2024]: https://arxiv.org/abs/2310.09656
