# `bko.datasets`: dataset loading

Load the evaluation datasets: UCI Heart Disease, the primary heart-disease prediction task, and Synthea, the population used to test gap detection by ablating selected cases.

UCI Heart Disease loads as its complete Cleveland records, split into training and test records,
with the age-sex subgroup-label cells of the training records counted (ADR [`research/uci-evaluation-frame`](../../../docs/adr/research/uci-evaluation-frame.md)). The values that define the data are constants in this package, and the ones the stages read are held together in one frame value, `bko.datasets.frames.FrameSpec`, whose UCI instance is `bko.datasets.uci.UCI_FRAME`, with UCI's numeric ranges (`RANGES`); the command and the `data/` layout are in [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#data). Synthea loads as a pinned generated population, from which an ablation removes chosen patients as ground truth (ADR [`research/synthea-ablation-design`](../../../docs/adr/research/synthea-ablation-design.md)). It also pins the MONDO, HPO and DO releases that `bko.ontology` reads.

- **Dependencies:** The `data` extra (`pyproject.toml`).
- **Owner:** Joe Laws
- **Status:** built.
