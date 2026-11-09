# `bko.validate`: record validation

Check each candidate record against the model's consistency checks before it joins the augmented dataset.

Each field must hold a number inside its frame's codebook or range constraint, with the frame's decimals, and a candidate names no key outside the frame's columns. The frame is a required keyword, so no caller checks against UCI's (`bko.datasets.uci.UCI_FRAME`) by default. No check reads an ontology relationship. A candidate is refused, never repaired, and counted under every check it fails. ADR [`research/record-validation-design`](../../../docs/adr/research/record-validation-design.md) records the checks and the counting rule.

- **Dependencies:** The `data` extra (`pyproject.toml`), and `bko.ontology.schema`, which reads no release.
- **Status:** built; a library with no command line.
