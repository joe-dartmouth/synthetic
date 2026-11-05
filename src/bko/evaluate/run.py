"""The harness: every arm on every holdout of an experiment, scored into one list.

Each holdout splits the records with its seed, under the frame's band edges, and an experiment's
removal then thins its training rows (`bko.evaluate.prepare.prepared`); the test rows are never
thinned. An arm gets a copy of the split's training rows and the request, which carries the frame,
never the test rows, so no generator sees a test record. A runner returns a checked `Baseline`;
`augment` checks every method's. The reference arm `real` has no runner: its added rows are the
rows the removal took.

Each holdout fills every arm, then scores each on the test split, with every model; under E1's
rule (`Experiment.decontaminate`) the split first loses the records any arm's rows copy, and an
experiment that keeps copies only counts them. A model's rows follow the regression's, its measures
suffixed with its name, and replica utility reads the regression only. A BKO arm is drained after
the gap fills and again after the replica fills, so each fill's candidates enter the scores under
their own source.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.methods import METHODS, Runner, augment
from bko.baselines.records import Baseline, number, training_set, unit_weights
from bko.baselines.request import Method, Request, Stratum
from bko.datasets.frames import FrameSpec
from bko.datasets.split import Split, split
from bko.evaluate.classifier import MODELS, Factory, Model, probabilities
from bko.evaluate.copies import copies, decontaminated
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import REAL_ARM, Experiment, flagged, gap_request, replica_request
from bko.evaluate.fidelity import MEASURES, fidelity
from bko.evaluate.filling import Filled, Logged, fill_arm, log_arm
from bko.evaluate.prepare import fingerprint, held_out_ratio, prepared, require_cell, split_line
from bko.evaluate.privacy import distances
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source
from bko.evaluate.shared_split import marked, marker
from bko.evaluate.utility import undefined, utility


logger = logging.getLogger(__name__)

MIN_LABELS: Final = 2
"""The labels a replica's rows must hold for a classifier fit on them alone to be scored."""


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


def run(
    records: pd.DataFrame,
    experiment: Experiment,
    arms: Mapping[str, Runner] = ARMS,
    *,
    models: Mapping[Model, Factory] = MODELS,
    drains: Mapping[str, Drain] = MappingProxyType({}),
    on_holdout: Callable[[], None] | None = None,
    frame: FrameSpec,
) -> list[Score]:
    """Score every arm of `experiment` on each of its holdouts of `records`, a dataset of `frame`.

    `on_holdout` is called after each holdout, so a caller can stop the run between holdouts.
    `models` must hold `Model.LR`, which the criteria and replica utility read. `REAL_ARM` needs
    no runner.

    Raises:
        EvaluationError: If the removal's cell is not a cell of `frame`, an arm other than
            `REAL_ARM` has no runner, no cell needs filling, or a fit does not converge.
        BaselineError: If an arm cannot fill a request.
        DatasetError: As `prepared`, if the removal's cell holds no training row of a holdout.
    """
    require_cell(experiment.removal, frame)
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
    _tally_undefined(scores, total)
    return scores


def _tally_undefined(scores: Sequence[Score], total: int) -> None:
    """Log, per arm, how many of the run's `total` holdouts scored its replica undefined."""
    undefined_in = Counter(
        s.arm
        for s in scores
        if s.source is Source.REPLICA
        and s.measure is Measure.AUC_PR
        and s.scope is Scope.ALL
        and math.isnan(s.value)
    )
    for arm, count in undefined_in.items():
        logger.warning("undefined replicas arm=%s holdouts=%d of=%d", arm, count, total)


def score_holdout(
    parts: Split,
    experiment: Experiment,
    arms: Mapping[str, Runner],
    *,
    models: Mapping[Model, Factory] = MODELS,
    drains: Mapping[str, Drain] = MappingProxyType({}),
    frame: FrameSpec,
    exclude: Sequence[pd.DataFrame] | None = None,
    logged: Logged | None = None,
) -> list[Score]:
    """Every score of one holdout of `frame`'s data: the held-out rows' own, then each arm's.

    The holdout's training rows are `parts`' sized by the experiment's training fraction, then
    thinned by its removal (`prepared`); the held-out ratio reads the sized split. `REAL_ARM`
    adds the removed rows, with no runner. The held-out rows end with the flagged cells'
    `fingerprint`, then, for an experiment that keeps copies, `Measure.COPIES_KEPT`, then, with
    `exclude` set, `Measure.SHARED_SPLIT` (`bko.evaluate.shared_split.marker`). A given `logged`
    receives the `split` line and every fill's `Filled` as they are logged.

    Raises:
        EvaluationError: If `models` lacks `Model.LR`, or a fit does not converge.
        BaselineError: If an arm cannot fill a request.
        DatasetError: As `prepared`, if the removal's cell holds no training row.
    """
    _require_regression(models)
    ready = prepared(parts, experiment, edges=frame.edges)
    ablated = ready.ablated
    train, test, seed = ablated.train, ablated.test, ablated.seed
    logged = Logged() if logged is None else logged
    logged.split = split_line(ablated)
    logger.info("%s", logged.split)
    gap = gap_request(train, experiment, seed=seed, frame=frame)
    _log_quota(seed, gap)
    cells = flagged(train, experiment, edges=frame.edges)
    logger.info("flagged holdout=%d cells=%s", seed, ",".join("/".join(map(str, c)) for c in cells))
    fills = {
        name: _collect(
            logged,
            _real(ready.removed, train, holdout=seed, seed=gap.seed, frame=frame)
            if name == REAL_ARM
            else fill_arm(
                arms[name],
                train,
                gap,
                holdout=seed,
                attempts=experiment.attempts,
                name=name,
                kind="gap",
            ),
        )
        for name in experiment.arms
    }
    gap_counts = {
        name: drains[name](seed, Source.CANDIDATES) for name in experiment.arms if name in drains
    }
    replicas = {
        name: _collect(
            logged,
            fill_arm(
                arms[name],
                train,
                replica_request(train, experiment, seed=seed, frame=frame),
                holdout=seed,
                attempts=experiment.attempts,
                name=f"{name}-replica",
                kind="replica",
            ),
        ).added
        for name in experiment.replica_arms
    }
    replica_counts = {
        name: drains[name](seed, Source.REPLICA_CANDIDATES) for name in replicas if name in drains
    }
    clean = _scored_split(
        test,
        [*(fill.added for fill in fills.values()), *replicas.values()],
        keep_copies=not experiment.decontaminate,
        holdout=seed,
        frame=frame,
        exclude=exclude,
    )
    for name in experiment.arms:
        _log_training(seed, name, "gap", fills[name].added, train, frame=frame)
        if name in replicas:
            _log_training(seed, name, "replica", replicas[name], train, frame=frame)
    scores = _rows(
        seed,
        HELD_OUT_ARM,
        Source.HELD_OUT,
        {
            Measure.HELD_OUT_RATIO: held_out_ratio(ready.sized, edges=frame.edges),
            **distances(test, train, frame=frame),
            Measure.FLAGGED_FINGERPRINT: fingerprint(cells),
            **({} if experiment.decontaminate else {Measure.COPIES_KEPT: 1.0}),
            **({} if exclude is None else marker(clean)),
        },
    )
    cutoff = experiment.cutoff
    for name in experiment.arms:
        scores += _arm_scores(seed, name, fills[name], ablated, clean, cells, models, frame, cutoff)
        if name in replicas:
            scores += _replica_scores(
                seed, name, replicas[name], ablated, clean, cells, models[Model.LR], frame, cutoff
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
    exclude: Sequence[pd.DataFrame] | None,
) -> pd.DataFrame:
    """The test records every arm is scored on, after logging how many some arm's rows copy.

    A test record is copied when some arm's added or replica row equals it in every one of
    `frame`'s columns. E1's rule drops each copied record; with `keep_copies` the whole split is
    scored, and the `test` line reads `kept=true`. A set `exclude` goes through `marked`.
    """
    clean = decontaminated(test, added, frame=frame)
    logger.info(
        "test holdout=%d records=%d copied=%d kept=%s",
        holdout,
        len(test),
        len(test) - len(clean),
        "true" if keep_copies else "false",
    )
    scored = test if keep_copies else clean
    if exclude is None:
        return scored
    copied = len(test) - len(clean)
    return marked(scored, exclude, records=len(test), copied=copied, holdout=holdout, frame=frame)


def _log_training(
    holdout: int,
    name: str,
    fill: str,
    added: pd.DataFrame,
    train: pd.DataFrame,
    *,
    frame: FrameSpec,
) -> None:
    """Log how many of an arm's `added` rows equal some training row in every frame column.

    A log line only, never a score: the scores' columns stay as they are.
    """
    logger.info(
        "training holdout=%d arm=%s fill=%s rows=%d copies=%d",
        holdout,
        name,
        fill,
        len(added),
        copies(added, train, frame=frame),
    )


def _log_quota(seed: int, request: Request) -> None:
    for stratum, rows in request.quota.items():
        logger.info("quota holdout=%d stratum=%s rows=%d", seed, "/".join(map(str, stratum)), rows)


def _collect(logged: Logged, filled: Filled) -> Baseline:
    """Record `filled` in `logged` and return its rows."""
    logged.fills.append(filled)
    return filled.baseline


def _real(
    removed: pd.DataFrame, train: pd.DataFrame, *, holdout: int, seed: int, frame: FrameSpec
) -> Filled:
    """`REAL_ARM`'s fill: the removed rows as added rows, every training row at weight 1.0.

    No runner fills it and no `check` reads it: its rows are real, so no quota or range applies.
    Its `arm` line reads attempt 1 and the gap request's seed.
    """
    added = number(removed.loc[:, list(frame.columns)])
    baseline = Baseline(added=added, weights=unit_weights(train), params={})
    line = log_arm(baseline, holdout=holdout, name=REAL_ARM, attempt=1, seed=seed)
    return Filled(REAL_ARM, "gap", 1, baseline, line)


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
        measured = utility(clean, fitted, cells, edges=frame.edges, cutoff=cutoff)
        scores += _scoped(seed, name, Source.AUGMENTED, _suffixed(measured, model))
    return scores


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
