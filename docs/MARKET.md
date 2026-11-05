# Market and regulation

Sources: [RESEARCH.md](RESEARCH.md), the plan of record, and the works [literature.md](literature.md) registers.

The Track A framing: who sells synthetic health data, how large the market is said to be, what
regulation drives demand, and who buys. The project's argument against the incumbents is in
[RESEARCH.md](RESEARCH.md#1-problem-and-organizational-context): platforms such as Gretel, MDClone
and MOSTLY AI optimize fidelity to the distribution they are given, so they reproduce its gaps.

Every fact below is the brief's. None of its sources has been fetched
([literature.md](literature.md#regulation-and-market)), and the brief flags several facts as
uncertain; its cautions stay beside the facts they qualify.

## Vendor landscape

The brief (E12) calls synthetic health data a real, funded market with recent M&A.

- **Gretel.ai.** Founded 2019; raised ~$67M (Greylock, Anthos; Series B $52.2M, Oct 2021, at a
  ~$320M valuation) (editor's reconstruction of a garbled line in the source). Acquired by NVIDIA,
  as reported by Wired on March 19, 2025, citing "two people with knowledge of the deal": the price
  "came in at nine figures and was higher than the $320 million at which Gretel was most recently
  valued," with "roughly 80 employees" folded into NVIDIA and neither company issuing a release.
  Product: synthetic tabular, text and time-series data with tunable differential privacy.
  *Caution*: the price is nine-figure but undisclosed.
- **MDClone** (Israel, founded 2016). A self-service health-data "ADAMS" platform with a patented
  Synthetic Data Engine. Raised $104M total, including a $63M Series C (Mar/Apr 2022) led by Warburg
  Pincus & Viola Growth. Clients include Washington University, Intermountain and Regenstrief. The
  brief's best "existing product to reimagine" candidate, because it is a health-specific,
  workflow-centric product.
- **MOSTLY AI** (Vienna, founded 2017). Raised ~$31M, including a $25M Series B (Jan 2022) led by
  Molten Ventures ("Europe's largest synthetic-data investment"). Released an open-source Synthetic
  Data SDK (Feb 2025). The brand was acquired by Syntho, confirmed by a Syntho press release (June
  9, 2026): "Syntho … today announced the acquisition of the MOSTLY AI Brand. Going forward, the
  MOSTLY AI brand will continue under the name 'MOSTLY AI, powered by Syntho,'" with Syntho
  acquiring the trademark and related assets after the company wound down operations. *Caution*: the
  wind-down is among the statuses the brief says to cite cautiously.
- **Tonic.ai** (SF, founded 2018). Raised $45M, including a $35M Series B (Sep 2021) led by Insight
  Partners. Developer-focused synthetic and de-identified test data; publishes HIPAA Expert
  Determination guidance as marketing.
- **Replica Analytics** (Ottawa; CEO Dr. Khaled El Emam). Acquisition by Aetion announced Jan 4,
  2022, terms undisclosed. "Replica Synthesis" generates privacy-protected synthetic copies of
  real-world data and is now part of Aetion's regulatory-grade RWE platform.
- **Hazy** (London, founded 2017). Raised ~$9–15M. Its principal software assets were acquired by
  SAS (Nov 2024, terms undisclosed) and folded into "SAS Data Maker".
- **Syntegra** (SF, founded 2019). $5.625M seed (Jan 2022). Its "Medical Mind" generated synthetic
  versions of NIH N3C COVID data. *Caution*: current status is ambiguous (reported out of business
  in 2023, or M&A in 2025); the brief says to verify before citing.
- **Betterdata** (Singapore, founded 2021). ~$1.55M seed (Apr 2023, Investible-led); programmable
  synthetic-data SaaS.
- **Synthea.** Open source (MITRE); the free baseline the brief says the team should actually use.
  The plan uses it for the gap-detection ablation ([RESEARCH.md](RESEARCH.md#6-datasets)).
- **SYNTHEMA.** Not a company: an EU Horizon Europe research project (grant 101095530, ~€7M, 16
  partners) on AI synthetic data for rare haematological diseases.
- **"Simulacra"** could not be verified as a vendor; the brief suspects a naming confusion.

## Market size

The brief (E13) says estimates vary widely by analyst and should be reported as a range:

| Analyst | Scope | Figures |
|---|---|---|
| Grand View Research | Global synthetic data generation | "valued at USD 218.3 million in 2023 and is projected to grow from USD 528.3 million in 2026 to USD 1,788.1 million by 2030, at a CAGR of 35.3% from 2024 to 2030". North America held 34.5% (2023) and tabular data 38.8% of revenue; healthcare & life sciences is the largest end-use segment |
| MarketsandMarkets (via MathCo) | Synthetic data | $0.3B (2023) to $2.1B (2028), 45.7% CAGR |
| Various analysts, including DataM Intelligence | Synthetic data in healthcare | ~$500–660M in 2025, growing at ~24–31.5% CAGR through the early-mid 2030s. DataM Intelligence: $657.92M in 2025 to $5.88B by 2033 |

The brief labels the healthcare line "Grand View, synthetic-data-in-healthcare specifically" but
attributes its range to various analysts and its point estimate to DataM Intelligence. All of these
are consultancy estimates that vary 2–3× between analysts: cite them as ranges and treat them as
order-of-magnitude, not precise.

## Regulatory drivers

The brief (E13) names the regulation behind the business pain.

**HIPAA** de-identification has two routes:

- **Safe Harbor**: remove 18 enumerated identifiers. Mechanical; destroys utility; does not handle
  quasi-identifiers well for rare conditions.
- **Expert Determination** (45 CFR 164.514(b)(1)): a qualified expert certifies "very small"
  re-identification risk. Preserves utility and allows date-shifting.

Synthetic data is marketed as a way to sidestep both. For rare diseases and small populations, Safe
Harbor is inadequate because residual quasi-identifiers re-identify people, which the brief calls a
direct selling point for synthetic data. The 2-pager makes the complementary point from the utility
side: Safe Harbor strips the quasi-identifiers that make rare cases findable
([RESEARCH.md](RESEARCH.md#1-problem-and-organizational-context)).

**GDPR** applies to all personal data of EU data subjects; synthetic data that is truly non-personal
falls outside it.

**EU AI Act** data-governance provisions (Article 10) require high-quality, representative training
data for high-risk systems, and synthetic data is cited as a mitigation.

**FDA**:

- In January 2025 FDA issued its first draft guidance on AI in drug and biologic development,
  "Considerations for the Use of Artificial Intelligence to Support Regulatory Decision-Making for
  Drug and Biological Products" (Docket FDA-2024-D-4689). It is built around a 7-step, risk-based
  credibility-assessment framework tied to a model's "context of use". Synthetic data is not
  substantively addressed in the draft; an FDA/CTTI workshop report explicitly flags "guidance on
  the use of synthetic data" as a future policy gap.
- FDA separately issued "Artificial Intelligence-Enabled Device Software Functions: Lifecycle
  Management and Marketing Submission Recommendations" (Docket FDA-2024-D-4488, Jan 2025), a
  total-product-lifecycle framework. FDA has authorized >1,000 AI-enabled devices.
- FDA's Real-World Evidence program stems from the 21st Century Cures Act (2016). The "Framework for
  FDA's Real-World Evidence Program" (Dec 2018) defines RWD and RWE but does not treat synthetic
  data as an RWD source.
- *Caution*: FDA has not issued synthetic-data-specific guidance; it is explicitly flagged as a
  policy gap. Do not imply that an approval pathway exists.

**Why firms buy synthetic data**: IRB and DUA delays, data-access cost, privacy liability, and
rare-disease data scarcity.

## Buyer personas

The brief's candidate personas (E14):

| Persona | Budget | Procurement | Pain |
|---|---|---|---|
| Pharma clinical data scientists | Highest willingness to pay | Enterprise procurement/MSA | Trial data scarcity, RWE generation, cross-border data sharing |
| Health-system ML engineers | Moderate | Not given | IRB/governance delays in accessing their own EHR; MDClone's actual customer |
| Digital-health startups without data access | Low budget, high need | Want cheap or open tools | Cold start with no data |
| Medical-device makers | Not given | Buy validation data to support FDA submissions | Need representative edge-case and rare-condition data |
| EHR vendors and academic researchers | Researchers: low willingness to pay, high volume | Academic/open procurement | Not given |

**Recommended persona.** The brief recommends "a digital-health-startup or academic ML engineer
facing a rare/under-represented clinical class with too little data and no fast path to more",
because it makes the value of ontology-guided gap-filling concrete and fits a product rather than a
research framing. The 2-pager's primary users match it: machine-learning engineers or clinical data
scientists at a digital-health startup or an academic laboratory
([RESEARCH.md](RESEARCH.md#2-intended-users-and-evaluation-setting)).
