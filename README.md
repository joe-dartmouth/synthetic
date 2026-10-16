# Bayesian Knowledge-Driven Ontologies for Synthetic Data

Partiban Anathurai, Shirley Asangi, Joe Laws, Krish Sapru. ENGG 119 project.

Testing whether medical knowledge can guide synthetic patient-data generation to fill coverage gaps
and improve heart-disease prediction for underrepresented groups.

> **Status: scaffold.** The toolchain and the `make check` gate run. Each pipeline stage under
> `src/bko/` has a README naming its role, its owner and whether the stage is built.

## Overview

Clinical machine-learning models fail on long-tail populations: rare subtypes and underserved
subgroups have too few training records. Commercial synthetic-data platforms optimize fidelity to
the data they are given, so they reproduce its gaps. This project instead uses Bayesian
Knowledge-driven Ontologies (BKOs; Santos, Jurmain and Ragazzi, PLOS ONE 2024), which combine
description logic with Bayesian Knowledge Bases and treat incompleteness as a property of the
representation. The planned pipeline:

1. **Represent.** Load MONDO and HPO as BKO fragments, and the dataset as a second fragment weighted
   by observed frequency.
2. **Detect.** Statements the ontology implies but the data does not support are the plan's formal
   definition of a dataset gap (ADR `research/bko-formal-gap-definition`).
3. **Generate and validate.** Generate candidate records for each gap through the Anthropic
   Messages API, guided by the ontology, and validate each before augmentation.
4. **Evaluate.** On UCI Heart Disease, compare rare-class recall and AUC-PR against no
   augmentation, class weighting, SMOTE, ADASYN, CTGAN and TVAE.

## Setup

Requires [uv](https://docs.astral.sh/uv/) on macOS or Linux. uv provides the interpreter named in
`.python-version`.

```bash
make setup     # dev tools and the data, synth and eval extras from the lock
make check     # the whole gate
make cells     # fetch UCI Heart Disease and print its age-sex cell counts
make synthea   # generate the Synthea population (needs a JDK) and ablate one cell
```

## Repository guide

The plan of record is [`docs/RESEARCH.md`](docs/RESEARCH.md); the source documents are in
[`papers/`](papers/proposal-2-pager.pdf). [`AGENTS.md`](AGENTS.md) holds the commands, layout,
conventions and document index.
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) is the full record of the environment and the gate.
Decisions are in [`docs/adr/`](docs/adr/README.md).

## License

Copyright (C) 2026 Dartmouth College. The code is licensed under the GNU General Public
License, version 3 only ([`LICENSE`](LICENSE)); ADR
[`tooling/gpl-3-license`](docs/adr/tooling/gpl-3-license.md) records why.
