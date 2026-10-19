# `bko.evaluate`: evaluation

Rare-class recall and AUC-PR (scikit-learn), fidelity (SDMetrics), privacy as distance to closest record (SDMetrics), the validity rate of generated records, and the share of detected gaps closed after fusion.

The harness scores every arm on E1's repeated holdouts and judges a candidate arm against E1's criteria. `bko.evaluate.experiment.E1` holds the configuration, ADR [`research/e1-evaluation-protocol`](../../../docs/adr/research/e1-evaluation-protocol.md) records the protocol, and [`docs/EVIDENCE.md`](../../../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation) states how each criterion is read. The commands are in [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#evaluation).

- **Dependencies:** The `eval` extra (`pyproject.toml`), and `bko.baselines` and `bko.datasets` with their `synth` and `data` extras.
- **Owner:** Joe Laws
- **Status:** built for utility, fidelity, privacy and the criteria. The BKO arms and the validity rate join in [`proposal-final-evaluation`](../../../tasks/proposal-final-evaluation.md).
