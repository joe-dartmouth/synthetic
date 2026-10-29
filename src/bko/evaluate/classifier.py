"""The downstream models every arm is scored with, fixed before any run.

The criteria read a logistic regression (`Model.LR`); a histogram gradient boosting model
(`Model.HGB`) is scored beside it with no verdict. Coded predictors are one-hot encoded over their
codebook, so every fit has the same columns whatever codes its training rows hold; the regression
also standardises measured predictors, which the trees split on as they are. The fields and
codebook are the frame's, which every builder requires. The weights reach the model only, as
`class_weight="balanced"` would. Every fit runs on one OpenMP thread. ADR
`research/e1-evaluation-protocol` records why.
"""

from __future__ import annotations

import enum
import warnings
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from bko.datasets.frames import FrameSpec
from bko.evaluate.errors import EvaluationError


MAX_ITER: Final = 1000
"""lbfgs iterations before a fit counts as not converged."""

type Factory = Callable[[FrameSpec], Pipeline]
"""Builds an unfitted classifier of a frame's features whose last step is named `model`."""


class Model(enum.StrEnum):
    """A downstream model; its value suffixes the measure names of every model but `LR`."""

    LR = "lr"
    """The logistic regression the criteria read."""
    HGB = "hgb"
    """The histogram gradient boosting model, reported with no verdict."""


def build(frame: FrameSpec) -> Pipeline:
    """The unfitted classifier: `frame`'s features, then a logistic regression named `model`."""
    model = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=MAX_ITER)
    return Pipeline([("features", _features(frame, scaled=True)), ("model", model)])


def build_hgb(frame: FrameSpec) -> Pipeline:
    """The unfitted boosting classifier: `frame`'s features, then the model named `model`.

    Early stopping is off and every feature is read, so no random number is drawn and the seed
    moves nothing; a leaf of 1 record sits below the gap fill's smallest cell, 2 rows.
    """
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=1,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features=None,
        early_stopping=False,
        class_weight=None,
        random_state=0,
    )
    return Pipeline([("features", _features(frame, scaled=False)), ("model", model)])


def _features(frame: FrameSpec, *, scaled: bool) -> ColumnTransformer:
    """Measured predictors, standardised when `scaled`, then coded ones one-hot over their codes."""
    codes = [sorted(frame.codebook[name]) for name in frame.categorical]
    measured = StandardScaler() if scaled else "passthrough"
    return ColumnTransformer(
        [
            ("measured", measured, list(frame.numeric)),
            ("coded", OneHotEncoder(categories=codes), list(frame.categorical)),
        ],
        sparse_threshold=0.0,
    )


MODELS: Final[Mapping[Model, Factory]] = MappingProxyType({Model.LR: build, Model.HGB: build_hgb})
"""Every downstream model's factory; the harness scores every arm with each."""


def probabilities(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    factory: Factory = build,
    frame: FrameSpec,
) -> pd.Series:
    """Fit on `training`, weighted by its `weight` column; return each test record's P(disease).

    The fit and the prediction run on one OpenMP thread: the boosting model scores the same on any
    count, and unbounded threads crawl under the gate's parallel workers.

    Raises:
        EvaluationError: If the fit does not converge within `MAX_ITER` iterations.
    """
    pipeline = factory(frame)
    features = list(frame.features)
    with threadpool_limits(limits=1, user_api="openmp"), warnings.catch_warnings():
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
