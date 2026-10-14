# Literature registry

Sources: every work cited in the [2-pager](../papers/proposal-2-pager.pdf) (section 1) and the [research brief](../papers/research-brief.pdf) (A1 to F16, Key Papers, Caveats); Santos et al. metadata verified outside the PDFs.

This file is the citation registry: one entry per work, report, docket or identifier the two sources
cite. Other documents cite a work by its key and link here for the identifier.

## How this registry works

Each entry is one bullet in this order: the key, the canonical identifier, the intake status, then
the authors, title and venue as the source gives them and the source sections that cite it.

```
- `key-YYYY` · [label spelling the identifier](https://canonical-url) · `intake-status` · Authors, "Title", venue. Cited in: ...
```

**Keys** are the lowercase first-author surname, a hyphen and the four-digit publication year, with
a trailing letter when two works share both. Where the source names no author, the key uses the
work's own short name (KIPPS, OntoBook) or a distinctive word from its title, method, publisher or
subject. Where the source gives no year, the year comes from the identifier when the identifier
encodes it (an arXiv ID's first four digits are the year and month, so `2507` is July 2025);
otherwise the key ends in `-nd` (no date) and the entry says the year is not given.

**Identifiers** are links whose label spells the identifier. A work for which neither source gives
an identifier reads `identifier not given in source` in place of the link. Titles in quotation marks are verbatim from
the citing source, ellipses included; the Santos et al. title is verbatim from its verified
metadata.

**Intake status** records how far the document itself has been taken in:

| Status | Meaning |
|---|---|
| `not-fetched` | Identifier recorded; document not retrieved |
| `fetched-not-read` | Document retrieved and stored; not read |
| `read-not-extracted` | Read; claims not yet extracted into the docs |
| `extracted` | Read, and the claims the docs rely on are checked against it |

Every entry is `not-fetched`: nothing here has been retrieved, so every claim attributed to a work
is the citing source's claim, not a checked one.

## Probabilistic ontologies and BKOs

- `costa-2006` · identifier not given in source · `not-fetched` · Costa & Laskey, "PR-OWL: A Framework for Probabilistic Ontologies", FOIS 2006. PR-OWL and Multi-Entity Bayesian Networks (MEBN). Cited in: brief A1, Key Papers 7.
- `santos-2024` · [doi:10.1371/journal.pone.0296864](https://doi.org/10.1371/journal.pone.0296864) · `not-fetched` · Santos, Jurmain, Ragazzi, "Bayesian-knowledge driven ontologies: A framework for fusion of semantic knowledge under uncertainty and incompleteness", PLOS ONE 19(3): e0296864, published 2024-03-27; also [PMC10971756](https://pmc.ncbi.nlm.nih.gov/articles/PMC10971756/). Dartmouth Thayer School of Engineering. Extends the earlier BKO framework, uniting description-logic reasoning with Bayesian Knowledge Bases, and fuses conflicting ontologies probabilistically instead of deleting knowledge. Metadata verified 2026-10-13. Cited in: 2-pager sections 1 and 3 (as "PLOS ONE, 2024").

## Gap detection and data quality

- `abecto-nd` · identifier not given in source · `not-fetched` · The "ABECTO paper": SHACL "cannot detect wrong values in a plausible range". Authors, title and year not given in source. Cited in: brief A2.
- `shacl-2025` · [arXiv:2507.22305](https://arxiv.org/abs/2507.22305) · `not-fetched` · "Is SHACL Suitable for Data Quality Assessment?". Defines SHACL shapes for the 69 data-quality metrics of Zaveri et al. Authors not given in source. Cited in: brief A2.
- `zaveri-2012` · identifier not given in source · `not-fetched` · Zaveri et al. (2012): the 69 data-quality metrics that `shacl-2025` maps to SHACL shapes. Title not given in source. Cited in: brief A2.

## Knowledge-guided generation

- `gmgan-2026` · [arXiv:2608.13256](https://arxiv.org/abs/2608.13256) · `not-fetched` · "Novel Knowledge-Guided Generative Methods for Synthetic Transcriptomic Data": Graph-Modulated GAN (GM-GAN) and Graph-Regularized GAN (GR-GAN). Authors not given in source. Cited in: brief A3.
- `kipps-2024` · [arXiv:2409.17315](https://arxiv.org/abs/2409.17315) · `not-fetched` · KIPPS, "Knowledge Infusion in Privacy Preserving Synthetic Data Generation". Authors not given in source. Cited in: brief A3.
- `ontobook-2026` · [arXiv:2607.18927](https://arxiv.org/abs/2607.18927) · `not-fetched` · OntoBook: ontology random walks rewritten by an LLM into synthetic "textbook" prose. Authors not given in source. Cited in: brief A3.
- `panaccione-2025` · [doi:10.3390/bioengineering12060658](https://doi.org/10.3390/bioengineering12060658) · `not-fetched` · Panaccione et al., "BioGAN", Bioengineering 2025. Graph neural networks inside a GAN for transcriptomic data. Cited in: brief A3, Key Papers 8.

## Tabular generators

- `borisov-2023` · [arXiv:2210.06280](https://arxiv.org/abs/2210.06280) · `not-fetched` · Borisov et al., "Language Models are Realistic Tabular Data Generators" (GReaT), ICLR 2023. Cited in: brief B4, Key Papers 2.
- `ganvsllm-nd` · [ResearchGate 389176619](https://www.researchgate.net/publication/389176619) · `not-fetched` · GPT-4o zero-shot study, titled in the source only as "GANs vs LLMs… synthetic tabular". Authors and year not given in source. Cited in: brief B4.
- `harmonic-2024` · identifier not given in source · `not-fetched` · HARMONIC, NeurIPS 2024. Authors and title not given in source. Cited in: brief B4.
- `llms-2024` · [arXiv:2406.14541](https://arxiv.org/abs/2406.14541) · `not-fetched` · "Why LLMs Are Bad at Synthetic Table Generation". Authors not given in source. Cited in: brief B4.
- `prosumer-2025` · identifier not given in source · `not-fetched` · "Benchmarking Tabular Data Synthesis… on Prosumer Hardware", CODATA Data Science Journal 2025. Authors not given in source. Cited in: brief B4, Key Papers 9.
- `seedat-2024` · [arXiv:2312.12112](https://arxiv.org/abs/2312.12112) · `not-fetched` · Seedat et al., "Curated LLM (CLLM)…", ICML 2024. Code: [github.com/seedatnabeel/CLLM](https://github.com/seedatnabeel/CLLM). Cited in: brief TL;DR, B4, D11, Key Papers 1.
- `xu-2019` · [arXiv:1907.00503](https://arxiv.org/abs/1907.00503) · `not-fetched` · Xu et al., "Modeling Tabular Data using Conditional GAN" (CTGAN, TVAE), NeurIPS 2019. Cited in: brief Key Papers 3.
- `zhao-2025` · identifier not given in source · `not-fetched` · Zhao et al. 2025, TabuLa. Title not given in source. Cited in: brief B4.

## Evaluation and failure modes

- `colorectal-nd` · [PMC11330441](https://pmc.ncbi.nlm.nih.gov/articles/PMC11330441/) · `not-fetched` · A colorectal-cancer synthetic-data study reporting TSTR/TRTS AUC of 0.99/0.98 and pMSE 0.223. Authors, title and year not given in source. Cited in: brief B5.
- `gerstgrasser-2024` · identifier not given in source · `not-fetched` · Gerstgrasser et al. (2024): accumulating real data alongside synthetic data avoids model collapse. Title not given in source. Cited in: brief B6.
- `shumailov-2024` · [doi:10.1038/s41586-024-07566-y](https://doi.org/10.1038/s41586-024-07566-y) · `not-fetched` · Shumailov et al., "AI models collapse when trained on recursively generated data", Nature 631:755–759 (2024). Cited in: brief B6, Key Papers 4.

## Augmentation versus resampling

- `cancer-nd` · [PMC11640389](https://pmc.ncbi.nlm.nih.gov/articles/PMC11640389/) · `not-fetched` · A cancer-prediction study: GAN resampling raised average ROC-AUC from 0.8276 to >0.9734. Authors, title and year not given in source. Cited in: brief D11.
- `gansmote-2019` · [arXiv:1904.09135](https://arxiv.org/abs/1904.09135) · `not-fetched` · A GAN-vs-SMOTE/ADASYN study: the GAN "improved results… but did not perform better than SMOTE or ADASYN." Authors and title not given in source. Cited in: brief D11.
- `imbalance-2022` · [doi:10.1186/s40537-022-00648-6](https://doi.org/10.1186/s40537-022-00648-6) · `not-fetched` · "The use of generative adversarial networks to alleviate class imbalance in tabular data", Journal of Big Data 2022 (a survey). Authors not given in source. Cited in: brief D11.
- `omics-nd` · [PMC10979623](https://pmc.ncbi.nlm.nih.gov/articles/PMC10979623/) · `not-fetched` · An omics-data study: GAN-based augmentation beat SMOTE and random oversampling on small high-dimensional data. Authors, title and year not given in source. Cited in: brief D11.
- `pharmacogenetics-nd` · [PMC12492482](https://pmc.ncbi.nlm.nih.gov/articles/PMC12492482/) · `not-fetched` · A pharmacogenetics study: classical copula and synthpop often give the best utility-privacy tradeoff in high-dimensional, limited-sample, privacy-sensitive settings. Authors, title and year not given in source. Cited in: brief D11.

## Small-model distillation and benchmarks

- `claimverification-2026` · [arXiv:2606.12854](https://arxiv.org/abs/2606.12854) · `not-fetched` · Mistral-7B QLoRA on biomedical claim verification (SciFact, HealthVer). Authors and title not given in source. Cited in: brief TL;DR, C7.
- `construct-2025` · [arXiv:2503.10694](https://arxiv.org/abs/2503.10694) · `not-fetched` · A construct-validity study: MedQA scores do not correlate well with real-world EHR performance. Authors and title not given in source. Cited in: brief C7.
- `gunasekar-2023` · [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) · `not-fetched` · Gunasekar et al., "Textbooks Are All You Need" (phi-1), 2023. Cited in: brief C7, Key Papers 6.
- `kim-2025` · [doi:10.1038/s41746-025-01653-8](https://doi.org/10.1038/s41746-025-01653-8) · `not-fetched` · Kim et al., "Small language models learn enhanced reasoning skills from medical textbooks" (Meerkat), npj Digital Medicine 2025. The brief gives the article number s41746-025-01653-8; the `10.1038/` prefix is Nature Portfolio's DOI prefix, added here. Cited in: brief TL;DR, C7, Key Papers 5.
- `layerlens-2026` · identifier not given in source · `not-fetched` · LayerLens/pricepertoken leaderboard, Oct 1 2026: the frontier MedQA reference scores. Cited in: brief C8.
- `meditron-2023` · [arXiv:2311.16079](https://arxiv.org/abs/2311.16079) · `not-fetched` · MEDITRON-70B. Authors and title not given in source. Cited in: brief C7.
- `versatile-2024` · identifier not given in source · `not-fetched` · "Towards evaluating… versatile LLMs for medicine", npj Digital Medicine 2024. Authors not given in source. Cited in: brief C7.

## Datasets

- `walonoski-2018` · identifier not given in source · `not-fetched` · Walonoski et al., "Synthea", JAMIA 2018 (25(3):230). Brief D10 attributes Synthea's documented fidelity flaws to "the JAMIA 2018 paper"; this registry reads that as the same work, which the brief does not state outright. Cited in: brief D10, Key Papers 10.

## Regulation and market

- `aiact-nd` · identifier not given in source · `not-fetched` · EU AI Act, data-governance provisions (Article 10). Year not given in source. Cited in: brief E13.
- `ctti-nd` · identifier not given in source · `not-fetched` · FDA/CTTI workshop report flagging "guidance on the use of synthetic data" as a future policy gap. Title and year not given in source. Cited in: brief E13.
- `cures-2016` · identifier not given in source · `not-fetched` · 21st Century Cures Act (2016), the origin of FDA's Real-World Evidence program. Cited in: brief E13.
- `datam-nd` · identifier not given in source · `not-fetched` · DataM Intelligence, synthetic data in healthcare market estimate. Title and year not given in source. Cited in: brief E13.
- `fda-2018` · identifier not given in source · `not-fetched` · FDA, "Framework for FDA's Real-World Evidence Program" (Dec 2018). Cited in: brief E13.
- `fda-2025a` · [FDA-2024-D-4689](https://www.regulations.gov/docket/FDA-2024-D-4689) · `not-fetched` · FDA draft guidance, "Considerations for the Use of Artificial Intelligence to Support Regulatory Decision-Making for Drug and Biological Products", January 2025. Cited in: brief E13.
- `fda-2025b` · [FDA-2024-D-4488](https://www.regulations.gov/docket/FDA-2024-D-4488) · `not-fetched` · FDA, "Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations", Jan 2025. Cited in: brief E13.
- `gdpr-nd` · identifier not given in source · `not-fetched` · General Data Protection Regulation (EU). Year not given in source. Cited in: brief E13.
- `grandview-nd` · identifier not given in source · `not-fetched` · Grand View Research (grandviewresearch.com), global synthetic data generation market report. Title and year not given in source. Cited in: brief E13.
- `hipaa-nd` · [45 CFR 164.514(b)(1)](https://www.ecfr.gov/current/title-45/section-164.514) · `not-fetched` · HIPAA de-identification, the Expert Determination route. Year not given in source. Cited in: brief E13.
- `marketsandmarkets-nd` · identifier not given in source · `not-fetched` · MarketsandMarkets synthetic data market estimate, reported via MathCo. Title and year not given in source. Cited in: brief E13.
- `synthema-nd` · [grant 101095530](https://cordis.europa.eu/project/id/101095530) · `not-fetched` · SYNTHEMA, an EU Horizon Europe research project (~€7M, 16 partners) on AI synthetic data for rare haematological diseases. Year not given in source. Cited in: brief E12.
- `syntho-2026` · identifier not given in source · `not-fetched` · Syntho press release (June 9, 2026) announcing the acquisition of the MOSTLY AI brand. Cited in: brief E12.
- `wired-2025` · identifier not given in source · `not-fetched` · Wired report (March 19, 2025) on NVIDIA's acquisition of Gretel. Cited in: brief E12.
