"""The harness: every arm on every holdout of an experiment, scored into one list.

Each holdout splits the frame with its seed. An arm gets a copy of the split's training rows and
the request, never the test rows, so no generator sees a test record. A runner returns a checked
`Baseline`; `augment` checks every method's.

Each holdout fills every arm, then scores each on the test split without the records any arm's
rows copy. A BKO arm is drained after the gap fills and again after the replica fills, so each
fill's candidates enter the scores under their own source.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.errors import BaselineError, OutputContractError
from bko.baselines.methods import METHODS, Runner, augment
from bko.baselines.records import Baseline, derived_seed, training_set
from bko.baselines.request import Method, Request, Stratum
from bko.datasets.cells import cell_counts
from bko.datasets.split import Split, split
from bko.evaluate.classifier import Factory, build, probabilities
from bko.evaluate.copies import copies, decontaminated
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment, flagged, gap_request, replica_request
from bko.evaluate.fidelity import fidelity
from bko.evaluate.privacy import distances
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source
from bko.evaluate.utility import utility


logger = logging.getLogger(__name__)


def method_arms(methods: Mapping[Method, Runner] = METHODS) -> dict[str, Runner]:
    """Every method in `methods` as an arm; a control ignores the request, as `augment` needs."""
    return {method.value: partial(_method_arm, method, methods) for method in methods}


def _method_arm(
    method: Method, methods: Mapping[Method, Runner], train: pd.DataFrame, request: Request
) -> Baseline:
    return augment(train, method, request if method.adds_rows else Request(), methods=methods)


ARMS: Final[Mapping[str, Runner]] = MappingProxyType(method_arms())
"""Every comparison method of `bko.baselines`, by name."""

type Drain = Callable[[], Mapping[Measure, float]]
"""A BKO arm's counts since the last call. The harness drains each arm after the gap fills and,
for a replica arm, after the replica fills, and writes them per holdout as `Source.CANDIDATES` and
`Source.REPLICA_CANDIDATES` scores."""


def held_out_ratio(parts: Split) -> float:
    """Nadeau and Bengio's n2/n1: the largest ratio of held-out to training records in a cell.

    Taking the largest keeps every scope's interval at least as wide as its own cells need.
    """
    kept = cell_counts(parts.train)["count"].tolist()
    held = cell_counts(parts.test)["count"].tolist()
    return max(out / into for out, into in zip(held, kept, strict=True) if out)


def run(
    frame: pd.DataFrame,
    experiment: Experiment,
    arms: Mapping[str, Runner] = ARMS,
    *,
    factory: Factory = build,
    drains: Mapping[str, Drain] = MappingProxyType({}),
) -> list[Score]:
    """Score every arm of `experiment` on each of its holdouts of `frame`.

    Raises:
        EvaluationError: If an arm has no runner, no cell needs filling, or a fit does not
            converge.
        BaselineError: If an arm cannot fill a request.
    """
    missing = sorted(set(experiment.arms) - set(arms))
    if missing:
        msg = f"no runner for arms {missing}"
        raise EvaluationError(msg)
    scores: list[Score] = []
    total = len(experiment.holdout_seeds)
    for done, seed in enumerate(experiment.holdout_seeds, start=1):
        started = time.perf_counter()
        parts = split(frame, seed=seed)
        scores += score_holdout(parts, experiment, arms, factory=factory, drains=drains)
        elapsed = time.perf_counter() - started
        logger.info("holdout=%d done=%d/%d seconds=%.1f", seed, done, total, elapsed)
    return scores


def score_holdout(
    parts: Split,
    experiment: Experiment,
    arms: Mapping[str, Runner],
    *,
    factory: Factory = build,
    drains: Mapping[str, Drain] = MappingProxyType({}),
) -> list[Score]:
    """Every score of one holdout: the held-out rows' own, then each arm's."""
    train, test, seed = parts.train, parts.test, parts.seed
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        seed,
        len(train),
        len(test),
        parts.fingerprint,
    )
    gap = gap_request(train, experiment, seed=seed)
    for stratum, rows in gap.quota.items():
        logger.info("quota holdout=%d stratum=%s rows=%d", seed, "/".join(map(str, stratum)), rows)
    cells = flagged(train, experiment)
    fills = {
        name: _fill(arms[name], train, gap, holdout=seed, attempts=experiment.attempts, name=name)
        for name in experiment.arms
    }
    gap_counts = {name: drains[name]() for name in experiment.arms if name in drains}
    replicas = {
        name: _fill(
            arms[name],
            train,
            replica_request(train, seed=seed),
            holdout=seed,
            attempts=experiment.attempts,
            name=f"{name}-replica",
        ).added
        for name in experiment.replica_arms
    }
    replica_counts = {name: drains[name]() for name in replicas if name in drains}
    clean = decontaminated(test, [*(fill.added for fill in fills.values()), *replicas.values()])
    logger.info("test holdout=%d records=%d copied=%d", seed, len(test), len(test) - len(clean))
    held_out = {Measure.HELD_OUT_RATIO: held_out_ratio(parts), **distances(test, train)}
    scores = _rows(seed, HELD_OUT_ARM, Source.HELD_OUT, held_out)
    for name in experiment.arms:
        scores += _arm_scores(seed, name, fills[name], parts, clean, cells, factory=factory)
        if name in replicas:
            replica = replicas[name]
            scores += _replica_scores(seed, name, replica, parts, clean, cells, factory=factory)
        if name in gap_counts:
            scores += _rows(seed, name, Source.CANDIDATES, gap_counts[name])
        if name in replica_counts:
            scores += _rows(seed, name, Source.REPLICA_CANDIDATES, replica_counts[name])
    return scores


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
        digest = hashlib.sha256(baseline.added.to_csv(lineterminator="\n").encode()).hexdigest()
        logger.info(
            "arm holdout=%d arm=%s attempt=%d seed=%d added=%d sha256=%s params=%s",
            holdout,
            name,
            attempt,
            request.seed,
            len(baseline.added),
            digest,
            json.dumps(baseline.params, sort_keys=True, default=str),
        )
        return baseline
    msg = f"attempts {attempts}: an arm needs at least one attempt"
    raise EvaluationError(msg)


def _arm_scores(
    seed: int,
    name: str,
    baseline: Baseline,
    parts: Split,
    clean: pd.DataFrame,
    cells: tuple[Stratum, ...],
    *,
    factory: Factory,
) -> list[Score]:
    """ADDED rows, then copies, then distances when rows were added, then AUGMENTED utility."""
    train, added = parts.train, baseline.added
    counts = {
        Measure.ROWS: float(len(added)),
        Measure.TEST_COPIES: float(copies(added, parts.test)),
    }
    scores = _rows(seed, name, Source.ADDED, counts)
    if len(added):
        scores += _rows(seed, name, Source.ADDED, distances(added, train))
    fitted = probabilities(training_set(train, baseline), clean, factory=factory)
    return scores + _scoped(seed, name, Source.AUGMENTED, utility(clean, fitted, cells))


def _replica_scores(
    seed: int,
    name: str,
    replica: pd.DataFrame,
    parts: Split,
    clean: pd.DataFrame,
    cells: tuple[Stratum, ...],
    *,
    factory: Factory,
) -> list[Score]:
    """REPLICA utility of a fit on the replica alone, then fidelity, then copies."""
    fitted = probabilities(replica.assign(weight=1.0), clean, factory=factory)
    copied = {Measure.TEST_COPIES: float(copies(replica, parts.test))}
    return [
        *_scoped(seed, name, Source.REPLICA, utility(clean, fitted, cells)),
        *_rows(seed, name, Source.REPLICA, fidelity(parts.train, replica)),
        *_rows(seed, name, Source.REPLICA, copied),
    ]


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
