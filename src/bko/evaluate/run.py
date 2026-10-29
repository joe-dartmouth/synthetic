"""The harness: every arm on every holdout of an experiment, scored into one list.

Each holdout splits the records with its seed, under the frame's band edges, and an experiment's
removal then thins its training rows (`prepared`); the test rows are never thinned. An arm gets a
copy of the split's training rows and the request, which carries the frame, never the test rows,
so no generator sees a test record. A runner returns a checked
`Baseline`; `augment` checks every method's. The reference arm `real` has no runner: its added
rows are the rows the removal took.

Each holdout fills every arm, then scores each on the test split, with every model; under E1's
rule (`Experiment.decontaminate`) the split first loses the records any arm's rows copy, and an
experiment that keeps copies only counts them. A model's rows follow the regression's, its measures
suffixed with its name, and replica utility reads the regression only. A BKO arm is drained after
the gap fills and again after the replica fills, so each fill's candidates enter the scores under
their own source.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from hashlib import sha256
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.errors import BaselineError, OutputContractError
from bko.baselines.methods import METHODS, Runner, augment
from bko.baselines.records import Baseline, derived_seed, number, training_set, unit_weights
from bko.baselines.request import Method, Request, Stratum
from bko.datasets.ablation import thinned
from bko.datasets.cells import cell_counts
from bko.datasets.frames import FrameSpec
from bko.datasets.split import Split, split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.classifier import MODELS, Factory, Model, probabilities
from bko.evaluate.copies import copies, decontaminated
from bko.evaluate.data import digest
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import (
    REAL_ARM,
    Experiment,
    Removal,
    flagged,
    gap_request,
    replica_request,
)
from bko.evaluate.fidelity import fidelity
from bko.evaluate.privacy import distances
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source
from bko.evaluate.utility import utility


logger = logging.getLogger(__name__)


def method_arms(methods: Mapping[Method, Runner] = METHODS) -> dict[str, Runner]:
    """Every method in `methods` as an arm.

    A control gets an empty request of the request's frame, as `augment` needs.
    """
    return {method.value: partial(_method_arm, method, methods) for method in methods}


def _method_arm(
    method: Method, methods: Mapping[Method, Runner], train: pd.DataFrame, request: Request
) -> Baseline:
    asked = request if method.adds_rows else Request(frame=request.frame)
    return augment(train, method, asked, methods=methods)


ARMS: Final[Mapping[str, Runner]] = MappingProxyType(method_arms())
"""Every comparison method of `bko.baselines`, by name."""

type Drain = Callable[[int, Source], Mapping[Measure, float]]
"""A BKO arm's counts since the last call, given the holdout's seed and the source they score as.
The harness drains each arm after the gap fills and, for a replica arm, after the replica fills,
and writes them per holdout as `Source.CANDIDATES` and `Source.REPLICA_CANDIDATES` scores."""


def prepared(
    parts: Split, removal: Removal | None, *, edges: Sequence[int]
) -> tuple[Split, pd.DataFrame]:
    """`parts` with `removal`'s rows gone from its training rows, and the rows removed.

    The draw takes the split's seed (`bko.datasets.ablation.thinned`), and the test rows are
    `parts`' own. With no removal, `parts` itself and no row.

    Raises:
        DatasetError: As `thinned`, if the removal's cell holds no training row.
    """
    if removal is None:
        return parts, parts.train.iloc[:0]
    kept = thinned(
        parts.train, cell=removal.cell, edges=edges, fraction=removal.fraction, seed=parts.seed
    )
    removed = parts.train.loc[~parts.train.index.isin(kept.index)]
    return Split(kept, parts.test, parts.seed), removed


def held_out_ratio(parts: Split, *, edges: Sequence[int]) -> float:
    """Nadeau and Bengio's n2/n1: the largest ratio of held-out to training records in a cell.

    Taking the largest keeps every scope's interval at least as wide as its own cells need. The
    harness reads it from the split before any removal, so a thinned cell does not set it.
    """
    kept = cell_counts(parts.train, edges=edges)["count"].tolist()
    held = cell_counts(parts.test, edges=edges)["count"].tolist()
    return max(out / into for out, into in zip(held, kept, strict=True) if out)


def fingerprint(cells: Sequence[Stratum]) -> float:
    """`cells` as a number: the first 12 hex digits of the sha256 of their sorted list's `repr`.

    The value is below 2**48, so a float holds it exactly.
    """
    return float(int(sha256(repr(sorted(cells)).encode()).hexdigest()[:12], 16))


def run(
    records: pd.DataFrame,
    experiment: Experiment,
    arms: Mapping[str, Runner] = ARMS,
    *,
    models: Mapping[Model, Factory] = MODELS,
    drains: Mapping[str, Drain] = MappingProxyType({}),
    on_holdout: Callable[[], None] | None = None,
    frame: FrameSpec = UCI_FRAME,
) -> list[Score]:
    """Score every arm of `experiment` on each of its holdouts of `records`, a dataset of `frame`.

    `on_holdout` is called after each holdout, so a caller can stop the run between holdouts.
    `models` must hold `Model.LR`, which the criteria and replica utility read. `REAL_ARM` needs
    no runner.

    Raises:
        EvaluationError: If an arm other than `REAL_ARM` has no runner, no cell needs filling, or
            a fit does not converge.
        BaselineError: If an arm cannot fill a request.
    """
    missing = sorted(set(experiment.arms) - set(arms) - {REAL_ARM})
    if missing:
        msg = f"no runner for arms {missing}"
        raise EvaluationError(msg)
    scores: list[Score] = []
    total = len(experiment.holdout_seeds)
    for done, seed in enumerate(experiment.holdout_seeds, start=1):
        started = time.perf_counter()
        parts = split(records, seed=seed, edges=frame.edges)
        scores += score_holdout(parts, experiment, arms, models=models, drains=drains, frame=frame)
        elapsed = time.perf_counter() - started
        logger.info("holdout=%d done=%d/%d seconds=%.1f", seed, done, total, elapsed)
        if on_holdout is not None:
            on_holdout()
    return scores


def score_holdout(
    parts: Split,
    experiment: Experiment,
    arms: Mapping[str, Runner],
    *,
    models: Mapping[Model, Factory] = MODELS,
    drains: Mapping[str, Drain] = MappingProxyType({}),
    frame: FrameSpec = UCI_FRAME,
) -> list[Score]:
    """Every score of one holdout of `frame`'s data: the held-out rows' own, then each arm's.

    The holdout's training rows are `parts`' after the experiment's removal (`prepared`); the
    held-out ratio reads `parts` itself. `REAL_ARM` adds the removed rows, with no runner. The
    held-out rows end with the flagged cells' `fingerprint`.

    Raises:
        EvaluationError: If `models` lacks `Model.LR`, or a fit does not converge.
        BaselineError: If an arm cannot fill a request.
    """
    _require_regression(models)
    whole = parts
    parts, removed = prepared(whole, experiment.removal, edges=frame.edges)
    train, seed = parts.train, parts.seed
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        seed,
        len(train),
        len(parts.test),
        parts.fingerprint,
    )
    gap = gap_request(train, experiment, seed=seed, frame=frame)
    _log_quota(seed, gap)
    cells = flagged(train, experiment, edges=frame.edges)
    fills = {
        name: _real(removed, train, holdout=seed, seed=gap.seed, frame=frame)
        if name == REAL_ARM
        else _fill(arms[name], train, gap, holdout=seed, attempts=experiment.attempts, name=name)
        for name in experiment.arms
    }
    gap_counts = {
        name: drains[name](seed, Source.CANDIDATES) for name in experiment.arms if name in drains
    }
    replicas = {
        name: _fill(
            arms[name],
            train,
            replica_request(train, seed=seed, frame=frame),
            holdout=seed,
            attempts=experiment.attempts,
            name=f"{name}-replica",
        ).added
        for name in experiment.replica_arms
    }
    replica_counts = {
        name: drains[name](seed, Source.REPLICA_CANDIDATES) for name in replicas if name in drains
    }
    clean = _scored_split(
        parts.test,
        [*(fill.added for fill in fills.values()), *replicas.values()],
        keep_copies=not experiment.decontaminate,
        holdout=seed,
        frame=frame,
    )
    scores = _rows(
        seed,
        HELD_OUT_ARM,
        Source.HELD_OUT,
        {
            Measure.HELD_OUT_RATIO: held_out_ratio(whole, edges=frame.edges),
            **distances(parts.test, train, frame=frame),
            Measure.FLAGGED_FINGERPRINT: fingerprint(cells),
        },
    )
    for name in experiment.arms:
        fill = fills[name]
        scores += _arm_scores(seed, name, fill, parts, clean, cells, models=models, frame=frame)
        if name in replicas:
            replica = replicas[name]
            scores += _replica_scores(
                seed, name, replica, parts, clean, cells, factory=models[Model.LR], frame=frame
            )
        if name in gap_counts:
            scores += _rows(seed, name, Source.CANDIDATES, gap_counts[name])
        if name in replica_counts:
            scores += _rows(seed, name, Source.REPLICA_CANDIDATES, replica_counts[name])
    return scores


def _scored_split(
    test: pd.DataFrame,
    added: Sequence[pd.DataFrame],
    *,
    keep_copies: bool,
    holdout: int,
    frame: FrameSpec,
) -> pd.DataFrame:
    """The test records every arm is scored on, after logging how many some arm's rows copy.

    E1's rule drops each copied record; with `keep_copies` the whole split is scored.
    """
    clean = decontaminated(test, added, frame=frame)
    logger.info("test holdout=%d records=%d copied=%d", holdout, len(test), len(test) - len(clean))
    return test if keep_copies else clean


def _log_quota(seed: int, request: Request) -> None:
    for stratum, rows in request.quota.items():
        logger.info("quota holdout=%d stratum=%s rows=%d", seed, "/".join(map(str, stratum)), rows)


def _fill(
    runner: Runner,
    train: pd.DataFrame,
    request: Request,
    *,
    holdout: int,
    attempts: int,
    name: str,
) -> Baseline:
    """Fill `request`, retrying a refusal with a derived seed; a contract break is never retried.

    The runner gets a copy of `train` on every attempt, so one that changes its input changes
    nothing the harness reads.

    Raises:
        BaselineError: The last refusal, if every attempt is refused, or a contract break.
        EvaluationError: If `attempts` is below 1.
    """
    first = request.seed
    for attempt in range(1, attempts + 1):
        try:
            baseline = runner(train.copy(), request)
        except OutputContractError:
            # The context only; the caller logs the error itself, without a traceback.
            logger.error(  # ruff: ignore[error-instead-of-exception]
                "contract holdout=%d arm=%s attempt=%d", holdout, name, attempt
            )
            raise
        except BaselineError as exc:
            if attempt == attempts:
                logger.error(  # ruff: ignore[error-instead-of-exception]
                    "refused holdout=%d arm=%s attempts=%d", holdout, name, attempts
                )
                raise
            logger.warning(
                "retry holdout=%d arm=%s attempt=%d reason=%s", holdout, name, attempt, exc
            )
            request = dataclasses.replace(request, seed=derived_seed(first, "attempt", attempt))
            continue
        _log_arm(baseline, holdout=holdout, name=name, attempt=attempt, seed=request.seed)
        return baseline
    msg = f"attempts {attempts}: an arm needs at least one attempt"
    raise EvaluationError(msg)


def _real(
    removed: pd.DataFrame, train: pd.DataFrame, *, holdout: int, seed: int, frame: FrameSpec
) -> Baseline:
    """`REAL_ARM`'s fill: the removed rows as added rows, every training row at weight 1.0.

    No runner fills it and no `check` reads it: its rows are real, so no quota or range applies.
    Its `arm` line reads attempt 1 and the gap request's seed.
    """
    added = number(removed.loc[:, list(frame.columns)])
    baseline = Baseline(added=added, weights=unit_weights(train), params={})
    _log_arm(baseline, holdout=holdout, name=REAL_ARM, attempt=1, seed=seed)
    return baseline


def _log_arm(baseline: Baseline, *, holdout: int, name: str, attempt: int, seed: int) -> None:
    logger.info(
        "arm holdout=%d arm=%s attempt=%d seed=%d added=%d sha256=%s params=%s",
        holdout,
        name,
        attempt,
        seed,
        len(baseline.added),
        digest(baseline.added),
        json.dumps(baseline.params, sort_keys=True, default=str),
    )


def _arm_scores(
    seed: int,
    name: str,
    baseline: Baseline,
    parts: Split,
    clean: pd.DataFrame,
    cells: tuple[Stratum, ...],
    *,
    models: Mapping[Model, Factory],
    frame: FrameSpec,
) -> list[Score]:
    """ADDED rows, then copies, then distances when rows were added, then AUGMENTED utility.

    The regression's utility comes first, then each other model's in `models`' order.
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
        fitted = probabilities(training, clean, factory=models[model], frame=frame)
        measured = utility(clean, fitted, cells, edges=frame.edges)
        scores += _scoped(seed, name, Source.AUGMENTED, _suffixed(measured, model))
    return scores


def _replica_scores(
    seed: int,
    name: str,
    replica: pd.DataFrame,
    parts: Split,
    clean: pd.DataFrame,
    cells: tuple[Stratum, ...],
    *,
    factory: Factory,
    frame: FrameSpec,
) -> list[Score]:
    """REPLICA utility of a fit on the replica alone, then fidelity, then copies."""
    fitted = probabilities(replica.assign(weight=1.0), clean, factory=factory, frame=frame)
    copied = {Measure.TEST_COPIES: float(copies(replica, parts.test, frame=frame))}
    return [
        *_scoped(seed, name, Source.REPLICA, utility(clean, fitted, cells, edges=frame.edges)),
        *_rows(seed, name, Source.REPLICA, fidelity(parts.train, replica, frame=frame)),
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
    if model is Model.LR:
        return values
    return {
        (Measure(f"{measure}-{model}"), scope): value for (measure, scope), value in values.items()
    }


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
