# Bayesian Knowledge-Driven Ontologies for Synthetic Data

Partiban Anathurai, Shirley Asangi, Joe Laws, Krish Sapru. ENGG 119 project.

Testing whether medical knowledge can guide synthetic patient-data generation to fill coverage gaps
and improve heart-disease prediction for underrepresented groups.

> **Status: every pipeline stage is built.** The toolchain and the `make check` gate run. Each
> stage under `src/bko/` has a README naming its role, its owner and its status.

## Overview

Clinical [machine-learning](https://en.wikipedia.org/wiki/Machine_learning) models fail on long-tail populations: rare subtypes and underserved
subgroups have too few training records. Commercial synthetic-data platforms optimize fidelity to
the data they are given, so they reproduce its gaps. This project instead uses Bayesian
Knowledge-driven Ontologies ([BKOs](https://journals.plos.org/plosone/); Santos, Jurmain and Ragazzi, [PLOS ONE](https://journals.plos.org/plosone/) 2024), which combine
[description logic](https://en.wikipedia.org/wiki/Description_logic) with [Bayesian Knowledge Bases](https://github.com/di2ag/pybkb) and treat incompleteness as a property of the
representation. The pipeline:

1. **Represent.** Load [MONDO](https://mondo.monarchinitiative.org/) (Monarch Disease Ontology) and [HPO](https://hpo.jax.org/) (Human Phenotype Ontology) as BKO fragments, and the dataset as a second fragment weighted
   by observed frequency.
2. **Detect.** Statements the ontology implies but the data does not support are the plan's formal
   definition of a dataset gap ([ADR](https://adr.github.io/) `research/bko-formal-gap-definition`).
3. **Generate and validate.** Generate candidate records for each gap through the Anthropic
   [Messages API](https://docs.anthropic.com/en/api/messages), guided by the ontology, or draw them from the fused BKB with
   the direct sampler, and validate each before augmentation.
4. **Evaluate.** On [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), compare rare-class recall and [AUC-PR](https://en.wikipedia.org/wiki/Precision_and_recall) (Area Under the Precision-Recall Curve) against no
   augmentation, class weighting, [SMOTE](https://arxiv.org/abs/1106.1813), [ADASYN](https://ieeexplore.ieee.org/document/4633969), [CTGAN](https://arxiv.org/abs/1907.00503) and [TVAE](https://arxiv.org/abs/1907.00503).

## Pending experiments

These follow the first full evaluation. Each is one task unit, and the set's graph and status
board is [`tasks/experiments-index.md`](tasks/experiments-index.md).

**Generation**

1. BKO-curated LLM records: score each LLM candidate by its likelihood under the fused BKB and keep
   the best; first measure whether the score separates records that match the real cell, then run
   it as an arm ([`generate-bkb-curation`](tasks/generate-bkb-curation.md)).
2. BKB-drawn concepts filled by the LLM: the BKB chooses each record's concepts, and the LLM fills
   the fields consistent with them (conditional generation)
   ([`generate-bkb-skeleton-fill`](tasks/generate-bkb-skeleton-fill.md)).

**Evaluation**

3. Dose-response: fill underrepresented cells to 30 and 40 rows instead of 20, every row-adding
   method at equal row counts ([`evaluate-dose-response`](tasks/evaluate-dose-response.md)).
4. Data-scarcity curve: train on 25, 50 and 75% of the training rows with the test split fixed, to
   see whether augmentation pays when data is scarcer than UCI
   ([`evaluate-data-scarcity-curve`](tasks/evaluate-data-scarcity-curve.md)).
5. Generator comparison: Haiku, Sonnet and Opus 5.5 at the fixed configuration, since the earlier
   Opus and Haiku runs differed in more than the model
   ([`evaluate-generator-comparison`](tasks/evaluate-generator-comparison.md)).
6. Synthea downstream experiment: build a heart-disease prediction table from the Synthea
   population, remove a known share of women with the disease from training, and measure how much
   recall each method recovers; the large cells answer the power problem UCI cannot
   ([`evaluate-synthea-downstream`](tasks/evaluate-synthea-downstream.md)).

## Setup

Requires [uv](https://docs.astral.sh/uv/) on macOS or Linux. uv provides the interpreter named in
`.python-version`.

```bash
make setup     # dev tools and the data, synth, eval, ontology, llm and app extras from the lock
make check     # the whole gate
make cells     # fetch UCI Heart Disease and print its age-sex cell counts
make synthea   # generate the Synthea population (needs a JDK) and ablate one cell
make reproduce # rebuild Santos et al.'s MONDO/DO fusion example and compare it with the paper
make app       # serve the Streamlit app on 127.0.0.1 (a live BKO fill spends money)
```

Tools used above: [Synthea](https://synthetichealth.github.io/synthea/) (synthetic patient generator), [JDK](https://openjdk.org/) (Java Development Kit), [DO](https://disease-ontology.org/) (Human Disease Ontology), [Streamlit](https://streamlit.io/).

## Repository guide

The plan of record is [`docs/RESEARCH.md`](docs/RESEARCH.md); the source documents are in
[`papers/`](papers/proposal-2-pager.pdf). [`AGENTS.md`](AGENTS.md) holds the commands, layout,
conventions and document index.
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) is the full record of the environment and the gate.
Decisions ([ADRs](https://adr.github.io/)) are in [`docs/adr/`](docs/adr/README.md).

## License

Copyright (C) 2026 Dartmouth College. The code is licensed under the [GNU General Public
License, version 3 only](https://www.gnu.org/licenses/gpl-3.0.html) ([`LICENSE`](LICENSE)); ADR
[`tooling/gpl-3-license`](docs/adr/tooling/gpl-3-license.md) records why.
`src/bko/bkb/` (BKB: Bayesian Knowledge Base) and its tests under `tests/` adapt code and data from
[pybkb](https://github.com/di2ag/pybkb) by Chase Yakaboski, also licensed under the [GNU General
Public License, version 3](https://www.gnu.org/licenses/gpl-3.0.html); ADR [`research/bkb-layer-source`](docs/adr/research/bkb-layer-source.md)
records the commit and the changes.
