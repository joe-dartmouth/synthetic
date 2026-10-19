# `bko.evaluate`: evaluation

Rare-class recall and AUC-PR (scikit-learn), fidelity (SDMetrics), privacy as distance to closest record (SDMetrics), and the validity rate of generated records.

The harness scores every arm (the baselines and the three BKO arms) on E1's repeated holdouts and judges a candidate arm against E1's criteria. `bko.evaluate.experiment.E1` holds the configuration, ADR [`research/e1-evaluation-protocol`](../../../docs/adr/research/e1-evaluation-protocol.md) records the protocol, and [`docs/EVIDENCE.md`](../../../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation) states how each criterion is read. The commands are in [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#evaluation).

- **Dependencies:** The `eval` extra (`pyproject.toml`), `bko.baselines` and `bko.datasets` with their `synth` and `data` extras, and, for the BKO arms, `bko.gaps`, `bko.validate`, `bko.generate` with its `llm` extra and `bko.ontology` with its `ontology` extra.
- **Owner:** Joe Laws
- **Status:** built.
