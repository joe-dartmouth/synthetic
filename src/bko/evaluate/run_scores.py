"""An arm's scores on one holdout: its added rows, its copies and distances, and its utility.

`_arm_scores` scores an arm's augmented training set under every model, the regression first, and
`_replica_scores` scores a replica fill alone. Each fit goes through `probabilities`, and a
fixed-epoch model's fit is logged under the harness's logger name, `bko.evaluate.run`, which
its log readers and tests filter on.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Final

import pandas as pd

from bko.baselines.records import Baseline, training_set
from bko.baselines.request import Stratum
from bko.datasets.frames import FrameSpec
from bko.datasets.split import Split
from bko.evaluate.classifier import MODELS, Factory, Model, probabilities
from bko.evaluate.copies import copies
from bko.evaluate.errors import EvaluationError
from bko.evaluate.fidelity import MEASURES, fidelity
from bko.evaluate.privacy import distances
from bko.evaluate.scores import Measure, Scope, Score, Source, suffixed
from bko.evaluate.utility import undefined, utility


logger = logging.getLogger("bko.evaluate.run")

MIN_LABELS: Final = 2
"""The labels a replica's rows must hold for a classifier fit on them alone to be scored."""


def _arm_scores(
    seed: int,
    name: str,
    baseline: Baseline,
    parts: Split,
    clean: pd.DataFrame,
    cells: tuple[Stratum, ...],
    models: Mapping[Model, Factory],
    frame: FrameSpec,
    cutoff: float,
    calibration: bool,
) -> list[Score]:
    """ADDED rows, then copies, then distances when rows were added, then AUGMENTED utility.

    The regression's utility comes first, then each other model's in `models`' order, each with
    its calibration measures when `calibration` holds.
    """
    train, added = parts.train, baseline.added
    counts = {
        Measure.ROWS: float(len(added)),
        Measure.TEST_COPIES: float(copies(added, parts.test, frame=frame)),
    }
    scores = _rows(seed, name, Source.ADDED, counts)
    if len(added):
        scores += _rows(seed, name, Source.ADDED, distances(added, train, frame=frame))
    training = training_set(train, baseline, frame=frame)
    ordered = [Model.LR, *(model for model in models if model is not Model.LR)]
    for model in ordered:
        fitted = _fitted(
            seed, name, training, clean, factory=models[model], frame=frame, model=model
        )
        measured = utility(
            clean, fitted, cells, edges=frame.edges, cutoff=cutoff, calibration=calibration
        )
        scores += _scoped(seed, name, Source.AUGMENTED, _suffixed(measured, model))
    return scores


def _fitted(
    seed: int,
    arm: str,
    training: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    factory: Factory,
    frame: FrameSpec,
    model: Model,
) -> pd.Series:
    """`probabilities` of `arm`'s fit at holdout `seed`, logging a fixed-epoch model's fit.

    The log line is `fit holdout=%d arm=%s model=%s n_iter=%d converged=true|false`. A model
    outside `MODELS` that does not converge is refused naming the holdout and the arm; a `MODELS`
    model's refusal keeps its message.

    Raises:
        EvaluationError: As `probabilities`.
    """

    def on_fit(n_iter: int, converged: bool) -> None:
        logger.info(
            "fit holdout=%d arm=%s model=%s n_iter=%d converged=%s",
            seed,
            arm,
            model,
            n_iter,
            "true" if converged else "false",
        )

    try:
        return probabilities(
            training, clean, factory=factory, frame=frame, model=model, on_fit=on_fit
        )
    except EvaluationError as exc:
        if model in MODELS:
            raise
        msg = f"holdout {seed}'s {arm} arm: {exc}"
        raise EvaluationError(msg) from exc


def _replica_scores(
    seed: int,
    name: str,
    replica: pd.DataFrame,
    parts: Split,
    clean: pd.DataFrame,
    cells: tuple[Stratum, ...],
    factory: Factory,
    frame: FrameSpec,
    cutoff: float,
) -> list[Score]:
    """REPLICA utility of a fit on the replica alone, then fidelity, then copies.

    A replica holding rows of fewer than 2 labels, as a partial fill may, fits no classifier: its
    utility is NaN in every measure and scope, under the keys a fit writes, and its WARNING names
    the labels it holds. A replica of fewer than 2 rows has no fidelity either, so its fidelity is
    NaN too.
    """
    labels = sorted(replica[frame.label].unique().tolist())
    if len(labels) < MIN_LABELS:
        logger.warning(
            "undefined holdout=%d arm=%s rows=%d labels=%d held=%s",
            seed,
            name,
            len(replica),
            len(labels),
            ",".join(map(str, labels)) or "none",
        )
        measured = undefined()
    else:
        fitted = probabilities(replica.assign(weight=1.0), clean, factory=factory, frame=frame)
        measured = utility(clean, fitted, cells, edges=frame.edges, cutoff=cutoff)
    found = (
        fidelity(parts.train, replica, frame=frame)
        if len(replica) >= MIN_LABELS
        else dict.fromkeys(MEASURES, math.nan)
    )
    copied = {Measure.TEST_COPIES: float(copies(replica, parts.test, frame=frame))}
    return [
        *_scoped(seed, name, Source.REPLICA, measured),
        *_rows(seed, name, Source.REPLICA, found),
        *_rows(seed, name, Source.REPLICA, copied),
    ]


def _require_regression(models: Mapping[Model, Factory]) -> None:
    """Refuse `models` without `Model.LR`, which the criteria and replica utility read.

    Raises:
        EvaluationError: If `models` lacks `Model.LR`.
    """
    if Model.LR not in models:
        msg = f"the models {[str(model) for model in models]} lack the regression {Model.LR!s}"
        raise EvaluationError(msg)


def _suffixed(
    values: Mapping[tuple[Measure, Scope], float], model: Model
) -> Mapping[tuple[Measure, Scope], float]:
    """`values` under `model`'s measure names: the regression's as they are, others suffixed."""
    return {(suffixed(measure, model), scope): value for (measure, scope), value in values.items()}


def _rows(seed: int, arm: str, source: Source, values: Mapping[Measure, float]) -> list[Score]:
    return [
        Score(seed, arm, source, measure, Scope.ALL, value) for measure, value in values.items()
    ]


def _scoped(
    seed: int, arm: str, source: Source, values: Mapping[tuple[Measure, Scope], float]
) -> list[Score]:
    return [
        Score(seed, arm, source, measure, scope, value)
        for (measure, scope), value in values.items()
    ]
