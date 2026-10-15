"""Dataset loading: the `bko.datasets` pipeline stage.

The submodules are imported directly, so importing the stage loads nothing heavy: `sources`
(pinned downloads), `fetch` (download and verify), `uci` (the UCI evaluation frame), `cells` (the
age-sex subgroup-label cells), `split` (the train/test split), `synthea_run` (generating a pinned
Synthea population), `synthea` (loading it), `ablation` (removing cases as ground truth) and
`errors`. `python -m bko.datasets` is the command line.
"""
