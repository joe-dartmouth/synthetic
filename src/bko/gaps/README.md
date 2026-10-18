# `bko.gaps`: gap detection

Find statements the ontology implies but the data does not support. Santos et al. state, as their Proposition 5.1.1, that these ungrounded statements contribute nothing to the model's distribution, so the plan uses them as the formal definition of a dataset gap. Count-based gaps are detected and evaluated separately. `bko.gaps.observe` turns a dataset into a fragment (ADR [`research/dataset-fragment-encoding`](../../../docs/adr/research/dataset-fragment-encoding.md)), `bko.gaps.detect` lists both kinds of gap, and `bko.gaps.experiment` holds E2's ablations and their reading; [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#gap-detection) holds the command.

- **Dependencies:** The `data` extra (`pyproject.toml`), and `bko.ontology` with its `ontology` extra, and `bko.bkb`.
- **Owner:** Partiban Anathurai
- **Status:** built for the population-level definition and count-based gaps. The per-subgroup (primary) and frequency-aware definitions of ADR [`research/bko-formal-gap-definition`](../../../docs/adr/research/bko-formal-gap-definition.md) join in [`proposal-gap-definitions`](../../../tasks/proposal-gap-definitions.md).
