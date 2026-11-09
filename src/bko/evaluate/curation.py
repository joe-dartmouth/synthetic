"""The CLLM arms' curator: a boosting classifier's training dynamics pick which added rows drop.

`classifier_discard` fits a gradient boosting classifier on a holdout's training rows and reads
each added row at every one of its first `STAGES` boosting stages: the row's confidence is the mean
probability the stages give its own label, and its aleatoric uncertainty the mean of p(1 - p). A
row is dropped when both are low (`discards`): the classifier confidently puts it at the other
label. The rule and its thresholds are CLLM's (Seedat et al. 2024; its results code reads them
off Data-IQ's training dynamics), with fixed thresholds, so a row's fate never depends on its
batch and a replay keeps the same rows. `classifier_scores` returns the two scores themselves,
which a curating attempt keeps on each row's candidates line (`bko.evaluate.holdout_candidates`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import islice
from typing import Final

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from bko.datasets.frames import FrameSpec
from bko.evaluate.holdout_candidates import ALEATORIC, CONFIDENCE


logger = logging.getLogger(__name__)

type Discard = Callable[[pd.DataFrame], pd.Series]
"""A curator: given added rows in a frame's columns, which to drop, by row, True to drop."""

type Scores = Callable[[pd.DataFrame], pd.DataFrame]
"""A curator's reading: given added rows in a frame's columns, each row's `CONFIDENCE` and
`ALEATORIC`, indexed as the rows."""

STAGES: Final = 99
"""The boosting stages read per row: the first 99 of the classifier's 100."""

CONFIDENCE_AT_MOST: Final = 0.2
"""A row whose mean confidence in its own label is at most this may be dropped."""

ALEATORIC_AT_MOST: Final = 0.15
"""A row whose mean aleatoric uncertainty is at most this may be dropped."""


def curator() -> HistGradientBoostingClassifier:
    """The unfitted curator: CLLM's boosting settings, with no early stop so every stage exists."""
    return HistGradientBoostingClassifier(
        max_iter=STAGES + 1,
        learning_rate=0.3,
        max_depth=6,
        l2_regularization=1.0,
        min_samples_leaf=5,
        early_stopping=False,
        random_state=0,
    )


def discards(confidence: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
    """Which rows CLLM drops: confidence and aleatoric uncertainty both at most their thresholds."""
    return (confidence <= CONFIDENCE_AT_MOST) & (aleatoric <= ALEATORIC_AT_MOST)


def classifier_discard(train: pd.DataFrame, *, frame: FrameSpec) -> Discard:
    """The curator fit on `train`, as a function of the added rows it reads.

    Args:
        train: The holdout's training rows, in `frame`'s columns.
        frame: The frame whose features and label the curator reads.

    Returns:
        A `Discard` that marks the rows `discards` drops from their `classifier_scores`, indexed
        as the rows given.

    Raises:
        ValueError: As `classifier_scores`.
    """
    scores = classifier_scores(train, frame=frame)

    def discard(rows: pd.DataFrame) -> pd.Series:
        read = scores(rows)
        dropped = discards(read[CONFIDENCE].to_numpy(), read[ALEATORIC].to_numpy())
        return pd.Series(dropped, index=rows.index, dtype=bool)

    return discard


def classifier_scores(train: pd.DataFrame, *, frame: FrameSpec) -> Scores:
    """The curator fit on `train`, as the reading of the added rows it scores.

    The fit and every read run on one OpenMP thread, as `bko.evaluate.classifier` does: the
    classifier is small, and threads beside other work cost far more than they save.

    Args:
        train: The holdout's training rows, in `frame`'s columns.
        frame: The frame whose features and label the curator reads.

    Returns:
        A `Scores` that reads each row's confidence and aleatoric uncertainty over the first
        `STAGES` stages.

    Logs one `curator` line: the training rows, their count per label, the stages read and both
    thresholds.

    Raises:
        ValueError: If `train` holds one label only, which no classifier separates, or the fit
            stops before every stage it reads.
    """
    features = list(frame.features)
    labels = train[frame.label].to_numpy()
    if len(np.unique(labels)) < 2:
        msg = f"the curator needs both labels in the training rows; got {np.unique(labels)}"
        raise ValueError(msg)
    with threadpool_limits(limits=1, user_api="openmp"):
        model = curator().fit(train.loc[:, features], labels)
    if model.n_iter_ <= STAGES:
        msg = f"the curator stopped after {model.n_iter_} stages; it reads {STAGES}"
        raise ValueError(msg)
    counts = dict(zip(*np.unique(labels, return_counts=True), strict=True))
    logger.info(
        "curator train=%d labels=%s stages=%d confidence_at_most=%s aleatoric_at_most=%s",
        len(train),
        ",".join(f"{label}:{count}" for label, count in sorted(counts.items())),
        STAGES,
        CONFIDENCE_AT_MOST,
        ALEATORIC_AT_MOST,
    )
    classes = list(model.classes_)

    def scores(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return pd.DataFrame({CONFIDENCE: [], ALEATORIC: []}, index=rows.index, dtype=float)
        own = np.array([classes.index(label) for label in rows[frame.label]])
        with threadpool_limits(limits=1, user_api="openmp"):
            stages = np.stack(
                [
                    staged[np.arange(len(rows)), own]
                    for staged in islice(model.staged_predict_proba(rows.loc[:, features]), STAGES)
                ]
            )
        confidence = stages.mean(axis=0)
        aleatoric = (stages * (1 - stages)).mean(axis=0)
        return pd.DataFrame({CONFIDENCE: confidence, ALEATORIC: aleatoric}, index=rows.index)

    return scores
