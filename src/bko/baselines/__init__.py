"""Baseline augmentation: the `bko.baselines` pipeline stage.

The submodules are imported directly, so importing the stage loads nothing heavy: `request` (the
methods, strata and quota), `records` (the output contract), `controls` (no augmentation, class
weighting), `oversample` (SMOTE, ADASYN), `synthesizers` (CTGAN, TVAE), `methods` (`augment`) and
`errors`. `python -m bko.baselines` is the command line.
"""
