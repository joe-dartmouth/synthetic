# `bko.bkb`: Bayesian Knowledge Bases

Represent, fuse and reason over Bayesian Knowledge Bases: the inference layer a Bayesian Knowledge-driven Ontology instantiates. The ontologies and the dataset each become a fragment, the dataset weighted by observed frequency.

The layer is adapted from pybkb, by Chase Yakaboski (GPL-3.0): the BKB structure (`bko.bkb.structure`), source-weighted fusion (`bko.bkb.fusion`), Santos et al.'s by default and pybkb's on request, the log joint mass of a target's states under complete evidence (`bko.bkb.reason`) and a JSON document for BKBs (`bko.bkb.serialize`), and, as this project's own, instantiation from an anchor and the probability of one inference (`bko.bkb.inference`). ADR [`research/bkb-layer-source`](../../../docs/adr/research/bkb-layer-source.md) names the pybkb commit, what was taken and how the port differs; [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#bkb-fixtures) says how its fixtures were exported from pybkb.

- **Dependencies:** No optional extra.
- **Owner:** Partiban Anathurai
- **Status:** built. The dataset fragment joins in [`proposal-gap-detection`](../../../tasks/proposal-gap-detection.md).
