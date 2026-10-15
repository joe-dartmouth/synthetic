"""The two controls: no augmentation, and class weighting. Neither adds a row."""

from __future__ import annotations

import pandas as pd

from bko.baselines.records import Baseline, no_rows, unit_weights
from bko.baselines.request import Request
from bko.datasets.cells import LABEL


def no_augmentation(train: pd.DataFrame, _request: Request) -> Baseline:
    """The training split as it is: no rows added, every weight 1.0."""
    return Baseline(added=no_rows(), weights=unit_weights(train), params={})


def class_weighting(train: pd.DataFrame, _request: Request) -> Baseline:
    """Weight each record n / (k * n_c): k labels present, n_c records with its label.

    scikit-learn's `class_weight="balanced"` heuristic, computed here so the stage declares no
    scikit-learn of its own. The weights sum to n. The label decides them, whatever the request's
    strata: the 2-pager's control weights classes.
    """
    sizes = train[LABEL].value_counts().sort_index()
    by_label = {
        int(label): len(train) / (len(sizes) * int(n))
        for label, n in zip(sizes.index.tolist(), sizes.tolist(), strict=True)
    }
    weights = train[LABEL].map(by_label).astype("float64").rename("weight")
    return Baseline(added=no_rows(), weights=weights, params={"weights": by_label})
