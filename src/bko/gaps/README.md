# `bko.gaps`: gap detection

Find statements the ontology implies but the data does not support. Santos et al. state, as their Proposition 5.1.1, that these ungrounded statements contribute nothing to the model's distribution, so the plan uses them as the formal definition of a dataset gap. Count-based gaps are detected and evaluated separately.

- **Dependencies:** The `data` extra (`pyproject.toml`), and `bko.ontology` with its `ontology` extra, and `bko.bkb`.
- **Owner:** Partiban Anathurai
- **Status:** not built; the package holds only its docstring. Planned in [`proposal-gap-detection`](../../../tasks/proposal-gap-detection.md).
