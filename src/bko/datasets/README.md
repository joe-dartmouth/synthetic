# `bko.datasets`: dataset loading

Load the evaluation datasets: UCI Heart Disease, the primary heart-disease prediction task, and Synthea, the population used to test gap detection by ablating selected cases.

UCI Heart Disease loads as its complete Cleveland records, split into training and test records,
with the age-sex subgroup-label cells of the training records counted (ADR [`research/uci-evaluation-frame`](../../../docs/adr/research/uci-evaluation-frame.md)). The values that define the data are constants in this package; the command and the `data/` layout are in [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#data).

- **Dependencies:** The `data` extra (`pyproject.toml`).
- **Owner:** Joe Laws
- **Status:** not built; the package holds only its docstring. Planned in [`proposal-datasets`](../../../tasks/proposal-datasets.md).
