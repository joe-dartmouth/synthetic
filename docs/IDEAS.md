# Ideas

Sources: [research brief](../papers/research-brief.pdf) TL;DR, A1, A2, B4, B6, C7, C8, C9, F15, F16, Caveats; [2-pager](../papers/proposal-2-pager.pdf) sections 2, 3 and 4.

Unscheduled ideas. Each records what it is, why it is not in the plan, and its sources. None is
scheduled; adopting one means changing [RESEARCH.md](RESEARCH.md) first.

## SLM distillation stretch goal

**What it is.** If the team is ahead of schedule, distill a small model on the best synthetic data
and show that it narrows the gap to a frontier LLM on one narrow, self-defined task (brief F15,
bullet 4 of the original project description).

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
or fine-tuned GReaT [`borisov-2023`][borisov-2023]. CLLM curates generated rows by predictive
confidence or aleatoric uncertainty; the brief cites its gains in the low-data regime (n<100) and
for underrepresented subgroups, and its code is public. GReaT is costly to train (~9h10m in its
paper, against 0:46 for TVAE). Either could join the comparison as a further baseline, and
CLLM-style curation could join the Validate stage.

**Why it is not in the plan.** The 2-pager generates through the Anthropic Messages API, guided by
ontology relationships, and its baselines are CTGAN, TVAE, SMOTE and ADASYN. For balance, the brief
also cites "Why LLMs Are Bad at Synthetic Table Generation" [`llms-2024`][llms-2024] on the failure
modes of in-context generation, which the Messages API approach is.

**Sources.** Brief TL;DR, B4, B6, F15 (bullet 1); 2-pager section 4. Details:
[BACKGROUND.md](BACKGROUND.md#tabular-generator-families).

## Alternative gap detectors

**What it is.** The brief's gap-detection threads other than BKO instantiation:

- **SHACL validation** under the closed-world assumption [`shacl-2025`][shacl-2025], which finds
  missing or invalid fields but not subtle distributional gaps.
- **Knowledge-graph completion** (link prediction) to infer missing edges.

**Why it is not in the plan.** The plan's formal gap is an ungrounded BKO statement (ADR
[research/bko-formal-gap-definition](adr/research/bko-formal-gap-definition.md)); the brief's
coverage score runs only as an ablation beside it ([RESEARCH.md](RESEARCH.md#detect)). SHACL could
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

[borisov-2023]: https://arxiv.org/abs/2210.06280
[claimverification-2026]: https://arxiv.org/abs/2606.12854
[construct-2025]: https://arxiv.org/abs/2503.10694
[gunasekar-2023]: https://arxiv.org/abs/2306.11644
[kim-2025]: https://doi.org/10.1038/s41746-025-01653-8
[llms-2024]: https://arxiv.org/abs/2406.14541
[meditron-2023]: https://arxiv.org/abs/2311.16079
[seedat-2024]: https://arxiv.org/abs/2312.12112
[shacl-2025]: https://arxiv.org/abs/2507.22305
