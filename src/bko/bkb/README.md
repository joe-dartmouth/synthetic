# `bko.bkb`: Bayesian Knowledge Bases

Represent, fuse and reason over Bayesian Knowledge Bases: the inference layer a Bayesian Knowledge-driven Ontology instantiates. The ontologies and the dataset each become a fragment, the dataset weighted by observed frequency.

The layer is adapted from pybkb, by Chase Yakaboski (GPL-3.0): the BKB structure (`bko.bkb.structure`), source-weighted fusion (`bko.bkb.fusion`), Santos et al.'s by default and pybkb's on request, the log joint mass of a target's states under complete evidence (`bko.bkb.reason`) and a JSON document for BKBs (`bko.bkb.serialize`), and, as this project's own, instantiation from an anchor and the probability of one inference (`bko.bkb.inference`). ADR [`research/bkb-layer-source`](../../../docs/adr/research/bkb-layer-source.md) names the pybkb commit, what was taken and how the port differs; [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#bkb-fixtures) says how its fixtures were exported from pybkb.

A fused BKB is valid when `check_fused` in `bko.bkb.fusion` passes: every input S-node present with its source tail and unchanged probability, every other fused S-node that is not a source prior mapped to an input, each source component's priors summing to 1 and every probability in [0, 1]; mutual exclusion is not required (ADR [`research/data-contradiction-handling`](../../../docs/adr/research/data-contradiction-handling.md)).

- **Dependencies:** No optional extra.
- **Owner:** Partiban Anathurai
- **Status:** built.
