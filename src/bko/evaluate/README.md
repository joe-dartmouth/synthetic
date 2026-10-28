# `bko.evaluate`: evaluation

Rare-class recall and AUC-PR (scikit-learn), fidelity (SDMetrics), privacy as distance to closest record (SDMetrics), and the validity rate of generated records.

The harness scores every arm (the baselines and the three BKO arms, or, in the free run `g2`, the direct arm `bko-direct`, which draws records from the fused BKB with no API call; ADR [`research/bkb-direct-generation`](../../../docs/adr/research/bkb-direct-generation.md)) on E1's repeated holdouts and judges a candidate arm against E1's criteria. `bko.evaluate.experiment.E1` holds the configuration, ADR [`research/e1-evaluation-protocol`](../../../docs/adr/research/e1-evaluation-protocol.md) records the protocol, and [`docs/EVIDENCE.md`](../../../docs/EVIDENCE.md#e1-uci-heart-disease-augmentation) states how each criterion is read. The CLI's `__main__` parses each command and dispatches it: `context` holds its `Context`, what a command reaches outside the process, `data` holds the `Data` (records, frame, field rows, name and digest) a named run's dataset chooses and every stage of `score` reads, and `scoring` holds the `score` path. The commands are in [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#evaluation).

- **Dependencies:** The `eval` extra (`pyproject.toml`), `bko.baselines` and `bko.datasets` with their `synth` and `data` extras, and, for the BKO arms, `bko.gaps`, `bko.validate`, `bko.generate` with its `llm` extra and `bko.ontology` with its `ontology` extra; the direct arm reads `bko.generate.direct` in place of the Messages API.
- **Owner:** Joe Laws
- **Status:** built.
