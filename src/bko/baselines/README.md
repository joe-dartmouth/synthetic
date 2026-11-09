# `bko.baselines`: baseline augmentation

The comparison methods, at equal row counts: no augmentation, class weighting, SMOTE and ADASYN (imbalanced-learn), CTGAN, TVAE and the Gaussian copula (SDV), sequential CART (scikit-learn), and a Bayesian network learned in-house by PrivBayes' GreedyBayes search (`bko.baselines.bayes_net`; ADR [`research/bayesian-network-baseline`](../../../docs/adr/research/bayesian-network-baseline.md)).

Each method reads the training split `bko.datasets` computes and fills exactly the rows per stratum
it is asked for, so methods given one request meet equal row counts stratum by stratum; the controls
add no rows. The methods, their parameters and the output contract are in ADR [`research/baseline-augmentation-design`](../../../docs/adr/research/baseline-augmentation-design.md); the command is in [`docs/DEVELOPMENT.md`](../../../docs/DEVELOPMENT.md#baselines).

- **Dependencies:** The `synth` extra (`pyproject.toml`).
- **Status:** built.
