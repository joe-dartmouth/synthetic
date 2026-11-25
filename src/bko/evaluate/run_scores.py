"""An arm's scores on one holdout: its added rows, its copies and distances, and its utility.

`_arm_scores` scores an arm's augmented training set under every model, the regression first, and
`_replica_scores` scores a replica fill alone. Each fit goes through `probabilities`, and a
fixed-epoch model's fit is logged under the harness's logger name, `bko.evaluate.run`, which
its log readers and tests filter on. `value_shares` and `code_share_error` read how far an arm's
added rows' codes sit from the removed rows' in a run that holds `bko-tree`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, cast

import pandas as pd

from bko.baselines.records import Baseline, training_set
from bko.baselines.request import Stratum
from bko.datasets.frames import FrameSpec
from bko.datasets.split import Split
from bko.evaluate.classifier import MODELS, Factory, Model, probabilities
from bko.evaluate.copies import copies
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import PRIOR_SHIFT_ARM
from bko.evaluate.fidelity import MEASURES, fidelity
from bko.evaluate.privacy import distances
from bko.evaluate.scores import Measure, Scope, Score, Source, suffixed
from bko.evaluate.shift import out_of_fold, shifted
from bko.evaluate.utility import undefined, utility


logger = logging.getLogger("bko.evaluate.run")

MIN_LABELS: Final = 2
"""The labels a replica's rows must hold for a classifier fit on them alone to be scored."""


def _arm_scores(  # ruff: ignore[too-many-arguments]  (the share error beside the fit's inputs)
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
    *,
    share_error: Mapping[str, pd.Series] = MappingProxyType({}),
    roc: bool = False,
) -> list[Score]:
    """ADDED rows, then copies, then distances when rows were added, then AUGMENTED utility.

    With `share_error`, each tree field's value shares in the removed training rows
    (`value_shares`), the arm writes `Measure.CODE_SHARE_ERROR` at the flagged scope after its
    distances, NaN when it added no rows, so every holdout of the run holds the same keys.

    The regression's utility comes first, then each other model's in `models`' order, each with
    its calibration measures when `calibration` holds and its ROC AUC when `roc` holds.
    `PRIOR_SHIFT_ARM`'s probabilities are `shifted` per model by offsets solved on the model's
    `out_of_fold` predictions of the training split.
    """
    train, added = parts.train, baseline.added
    counts = {
        Measure.ROWS: float(len(added)),
        Measure.TEST_COPIES: float(copies(added, parts.test, frame=frame)),
    }
    scores = _rows(seed, name, Source.ADDED, counts)
    if len(added):
        scores += _rows(seed, name, Source.ADDED, distances(added, train, frame=frame))
    if share_error:
        error = code_share_error(added, share_error) if len(added) else math.nan
        scores.append(
            Score(seed, name, Source.ADDED, Measure.CODE_SHARE_ERROR, Scope.FLAGGED, error)
        )
    training = training_set(train, baseline, frame=frame)
    ordered = [Model.LR, *(model for model in models if model is not Model.LR)]
    shift_counts = (
        cast("Mapping[str, Sequence[int]]", baseline.params["shift"])
        if name == PRIOR_SHIFT_ARM
        else None
    )
    for model in ordered:
        fitted = _fitted(
            seed, name, training, clean, factory=models[model], frame=frame, model=model
        )
        if shift_counts is not None:
            held, folds = out_of_fold(
                training, factory=models[model], frame=frame, model=model, seed=seed
            )
            fitted = shifted(
                fitted,
                held,
                train,
                clean,
                shift_counts,
                edges=frame.edges,
                holdout=seed,
                model=model,
                folds=folds,
            )
        measured = utility(
            clean,
            fitted,
            cells,
            edges=frame.edges,
            cutoff=cutoff,
            calibration=calibration,
            roc=roc,
        )
        scores += _scoped(seed, name, Source.AUGMENTED, _suffixed(measured, model))
    return scores


def value_shares(values: pd.Series) -> pd.Series:
    """Each value's share of `values`, indexed by the value as a float; empty for no value."""
    return values.astype(float).value_counts(normalize=True, sort=False)


def code_share_error(added: pd.DataFrame, removed: Mapping[str, pd.Series]) -> float:
    """The mean over `removed`'s columns of the total variation distance from `added`'s shares.

    Each column's distance is half the sum, over every value either side holds, of the absolute
    difference between `added`'s share (`value_shares`) and `removed`'s. A column `removed` holds
    no value of reads NaN, and so does the mean.
    """
    distances = []
    for column, held in removed.items():
        if held.empty:
            distances.append(math.nan)
            continue
        drawn = value_shares(added[column])
        values = drawn.index.union(held.index)
        gaps = drawn.reindex(values, fill_value=0.0) - held.reindex(values, fill_value=0.0)
        distances.append(math.fsum(gaps.abs()) / 2)
    return math.fsum(distances) / len(distances)


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
