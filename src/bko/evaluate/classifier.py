"""The downstream model every arm is scored with: one logistic regression, fixed before any run.

Measured predictors are standardised and coded ones one-hot encoded over their codebook, so every
fit has the same columns whatever codes its training rows hold. The fields and codebook are the
frame's, UCI's by default. The weights reach the regression only, as `class_weight="balanced"`
would. ADR `research/e1-evaluation-protocol` records why.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.errors import EvaluationError


MAX_ITER: Final = 1000
"""lbfgs iterations before a fit counts as not converged."""

type Factory = Callable[[FrameSpec], Pipeline]
"""Builds an unfitted classifier of a frame's features whose last step is named `model`."""


def build(frame: FrameSpec = UCI_FRAME) -> Pipeline:
    """The unfitted classifier: `frame`'s features, then a logistic regression named `model`."""
    codes = [sorted(frame.codebook[name]) for name in frame.categorical]
    features = ColumnTransformer(
        [
            ("measured", StandardScaler(), list(frame.numeric)),
            ("coded", OneHotEncoder(categories=codes), list(frame.categorical)),
        ],
        sparse_threshold=0.0,
    )
    model = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=MAX_ITER)
    return Pipeline([("features", features), ("model", model)])


def probabilities(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    factory: Factory = build,
    frame: FrameSpec = UCI_FRAME,
) -> pd.Series:
    """Fit on `training`, weighted by its `weight` column; return each test record's P(disease).

    Raises:
        EvaluationError: If the fit does not converge within `MAX_ITER` iterations.
    """
    pipeline = factory(frame)
    features = list(frame.features)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(
                training.loc[:, features],
                training[frame.label],
                model__sample_weight=training["weight"].to_numpy(),
            )
        except ConvergenceWarning as exc:
            msg = f"the classifier did not converge in {MAX_ITER} iterations"
            raise EvaluationError(msg) from exc
    positive = pipeline.predict_proba(test.loc[:, features])[:, 1]
    return pd.Series(positive, index=test.index, name="probability", dtype="float64")
