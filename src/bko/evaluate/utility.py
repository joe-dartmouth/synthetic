"""Recall, precision and AUC-PR of one classifier's probabilities on the test split, per scope.

A record is called positive at `THRESHOLD` or above. A measure a scope cannot define is NaN, never
scikit-learn's fallback: recall needs a positive record, AUC-PR a positive and a negative one, and
precision a record called positive.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import Final

import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from bko.baselines.request import Stratum
from bko.datasets.cells import LABEL, cell_keys
from bko.evaluate.scores import Measure, Scope


THRESHOLD: Final = 0.5
"""The probability at or above which a record is called positive."""


def in_scope(
    test: pd.DataFrame,
    scope: Scope,
    flagged: Collection[Stratum],
    *,
    edges: Sequence[int],
) -> pd.Series:
    """Which test records `scope` covers, as booleans on the test index; `edges` band them."""
    if scope is Scope.ALL:
        return pd.Series(True, index=test.index)
    subgroups = {(band, sex) for band, sex, *_ in flagged}
    cells = cell_keys(test, edges=edges)
    covered = [
        (band, sex) in subgroups for band, sex in zip(cells["age_band"], cells["sex"], strict=True)
    ]
    return pd.Series(covered, index=test.index)


def utility(
    test: pd.DataFrame,
    probability: pd.Series,
    flagged: Collection[Stratum],
    *,
    edges: Sequence[int],
) -> dict[tuple[Measure, Scope], float]:
    """Each measure in each scope, `probability` holding P(disease) per test record by index."""
    scores: dict[tuple[Measure, Scope], float] = {}
    matched = probability.loc[test.index].to_numpy()
    for scope in Scope:
        mask = in_scope(test, scope, flagged, edges=edges).to_numpy()
        truth = test[LABEL].to_numpy()[mask]
        score = matched[mask]
        called = (score >= THRESHOLD).astype("int64")
        positives = int(truth.sum())
        negatives = len(truth) - positives
        scores[Measure.RECALL, scope] = (
            float(recall_score(truth, called)) if positives else math.nan
        )
        scores[Measure.PRECISION, scope] = (
            float(precision_score(truth, called)) if called.any() else math.nan
        )
        scores[Measure.AUC_PR, scope] = (
            float(average_precision_score(truth, score)) if positives and negatives else math.nan
        )
    return scores
