"""The downstream models every arm is scored with, fixed before any run.

The criteria read a logistic regression (`Model.LR`); a histogram gradient boosting model
(`Model.HGB`) is scored beside it with no verdict. `PANEL` adds a random forest, extra trees, a
multilayer perceptron and a categorical naive Bayes, also with no verdict. Coded predictors are
one-hot encoded over their codebook, so every fit has the same columns whatever codes its training
rows hold; the regression and the perceptron also standardise measured predictors, which the trees
split on as they are. Naive Bayes bins measured predictors and reads coded ones as ordinals over
their codebook. The fields and codebook are the frame's, which every builder requires. The weights
reach the model only, as `class_weight="balanced"` would. Every fit runs on one OpenMP thread.
ADRs `research/e1-evaluation-protocol` and `research/classifier-panel` record why.
"""

from __future__ import annotations

import enum
import logging
import warnings
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from bko.datasets.frames import FrameSpec
from bko.evaluate.errors import EvaluationError


logger = logging.getLogger(__name__)

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
    RF = "rf"
    """The random forest, reported with no verdict."""
    ET = "et"
    """The extra trees, reported with no verdict."""
    MLP = "mlp"
    """The multilayer perceptron, fit for `MLP_EPOCHS` epochs and reported with no verdict."""
    NB = "nb"
    """The categorical naive Bayes, reported with no verdict."""


TREES: Final = 500
"""Trees in each forest."""

MLP_EPOCHS: Final = 15
"""The perceptron's epochs: lowest validation log-loss on E6 holdout 0's `none` training split."""

NB_BINS: Final = 10
"""Equal-width bins each measured predictor is cut into for naive Bayes."""


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


def build_rf(frame: FrameSpec) -> Pipeline:
    """The unfitted random forest: `frame`'s unscaled features, then the model named `model`."""
    model = RandomForestClassifier(
        n_estimators=TREES,
        criterion="gini",
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=True,
        class_weight=None,
        n_jobs=1,
        random_state=0,
    )
    return Pipeline([("features", _features(frame, scaled=False)), ("model", model)])


def build_et(frame: FrameSpec) -> Pipeline:
    """The unfitted extra trees: `frame`'s unscaled features, then the model named `model`."""
    model = ExtraTreesClassifier(
        n_estimators=TREES,
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=False,
        class_weight=None,
        n_jobs=1,
        random_state=0,
    )
    return Pipeline([("features", _features(frame, scaled=False)), ("model", model)])


def build_mlp(frame: FrameSpec) -> Pipeline:
    """The unfitted perceptron: `frame`'s scaled features, then the model named `model`.

    It stops at `MLP_EPOCHS` epochs with no early stopping, so its `ConvergenceWarning` is
    expected (`FIXED_EPOCHS`).
    """
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=200,
        learning_rate_init=1e-3,
        early_stopping=False,
        max_iter=MLP_EPOCHS,
        random_state=0,
    )
    return Pipeline([("features", _features(frame, scaled=True)), ("model", model)])


def build_nb(frame: FrameSpec) -> Pipeline:
    """The unfitted naive Bayes: binned and coded features, then the model named `model`.

    Each measured predictor is cut into `NB_BINS` equal-width bins over its training range, and
    each coded one is its code's position in the codebook. Every feature declares its full
    category count, so a test category no training row holds still scores.
    """
    codes = [sorted(frame.codebook[name]) for name in frame.categorical]
    bins = KBinsDiscretizer(n_bins=NB_BINS, encode="ordinal", strategy="uniform", subsample=None)
    features = ColumnTransformer(
        [
            ("measured", bins, list(frame.numeric)),
            ("coded", OrdinalEncoder(categories=codes), list(frame.categorical)),
        ],
        sparse_threshold=0.0,
    )
    counts = [NB_BINS] * len(frame.numeric) + [len(code) for code in codes]
    model = CategoricalNB(alpha=1.0, min_categories=counts)
    return Pipeline([("features", features), ("model", model)])


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
"""The default downstream models' factories; the harness scores every arm with each."""

PANEL: Final[Mapping[Model, Factory]] = MappingProxyType(
    {
        Model.LR: build,
        Model.HGB: build_hgb,
        Model.RF: build_rf,
        Model.ET: build_et,
        Model.MLP: build_mlp,
        Model.NB: build_nb,
    }
)
"""Every model's factory, in panel order; pass it as `models` to score the whole panel."""

FIXED_EPOCHS: Final = frozenset({Model.MLP})
"""Models fit for a fixed iteration count, whose `ConvergenceWarning` is recorded, not raised."""


def probabilities(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    factory: Factory = build,
    frame: FrameSpec,
    model: Model = Model.LR,
) -> pd.Series:
    """Fit on `training`, weighted by its `weight` column; return each test record's P(disease).

    `model` names the fit in errors and logs. The fit and the prediction run on one OpenMP thread:
    the boosting model scores the same on any count, and unbounded threads crawl under the gate's
    parallel workers. A `FIXED_EPOCHS` model's `ConvergenceWarning` is recorded and its fit logged
    with its iteration count; any other warning it raises is re-issued.

    Raises:
        EvaluationError: If a model outside `FIXED_EPOCHS` does not converge within its
            iteration limit.
    """
    pipeline = factory(frame)
    features = list(frame.features)
    fit_params = {"model__sample_weight": training["weight"].to_numpy()}
    fit_args = (training.loc[:, features], training[frame.label])
    with threadpool_limits(limits=1, user_api="openmp"):
        if model in FIXED_EPOCHS:
            _fit_fixed(pipeline, fit_args, fit_params, model)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    pipeline.fit(*fit_args, **fit_params)
                except ConvergenceWarning as exc:
                    limit = pipeline.named_steps["model"].get_params().get("max_iter")
                    msg = f"the {model} classifier did not converge in {limit} iterations"
                    raise EvaluationError(msg) from exc
        positive = pipeline.predict_proba(test.loc[:, features])[:, 1]
    return pd.Series(positive, index=test.index, name="probability", dtype="float64")


def _fit_fixed(
    pipeline: Pipeline,
    fit_args: tuple[pd.DataFrame, pd.Series],
    fit_params: Mapping[str, object],
    model: Model,
) -> None:
    """Fit `pipeline` recording its `ConvergenceWarning`; log `n_iter_` and whether it converged."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(*fit_args, **fit_params)
    converged = True
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            converged = False
        else:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    n_iter = pipeline.named_steps["model"].n_iter_
    logger.info("fit model=%s n_iter=%d converged=%s", model, n_iter, str(converged).lower())
