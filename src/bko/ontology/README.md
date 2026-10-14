# `bko.ontology`: ontology loading and field mapping

Load MONDO and HPO through owlready2, pronto and OLS4, and map ontology concepts to dataset fields, starting with the disease label and chest-pain type. Each mapping records its source, encoding and limits; numeric fields are governed by separately specified range constraints.

- **Dependencies:** The `ontology` extra (`pyproject.toml`).
- **Owner:** Partiban Anathurai
- **Status:** not built; the package holds only its docstring. Planned in [`proposal-ontology-mapping`](../../../tasks/proposal-ontology-mapping.md).
