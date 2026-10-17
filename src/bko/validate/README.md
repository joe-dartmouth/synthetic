# `bko.validate`: record validation

Check each candidate record against the model's consistency checks before it joins the augmented dataset.

Each field must hold a number inside its codebook or the mapping table's range constraint, with the archive's decimals, and a candidate names no key outside the evaluation frame. No check reads an ontology relationship. A candidate is refused, never repaired, and counted under every check it fails. ADR [`research/record-validation-design`](../../../docs/adr/research/record-validation-design.md) records the checks and the counting rule.

- **Dependencies:** The `data` extra (`pyproject.toml`), and `bko.ontology.mapping`, which reads no release.
- **Owner:** Shirley Asangi
- **Status:** built; a library with no command line.
