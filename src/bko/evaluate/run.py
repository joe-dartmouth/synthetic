"""The harness: every arm on every holdout of an experiment, scored into one list.

Each holdout splits the frame with its seed. An arm gets a copy of the split's training rows and
the request, never the test rows, so no generator sees a test record. A runner returns a checked
`Baseline`; `augment` checks every method's.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
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
    for seed in experiment.holdout_seeds:
        started = time.perf_counter()
        scores += score_holdout(split(frame, seed=seed), experiment, arms, factory=factory)
        logger.info("holdout=%d seconds=%.1f", seed, time.perf_counter() - started)
    return scores


def score_holdout(
    parts: Split, experiment: Experiment, arms: Mapping[str, Runner], *, factory: Factory = build
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
    held_out = {Measure.HELD_OUT_RATIO: held_out_ratio(parts), **distances(test, train)}
    scores = _rows(seed, HELD_OUT_ARM, Source.HELD_OUT, held_out)
    for name in experiment.arms:
        baseline = _fill(arms[name], train, gap, attempts=experiment.attempts, name=name)
        scores += _rows(seed, name, Source.ADDED, {Measure.ROWS: float(len(baseline.added))})
        if len(baseline.added):
            scores += _rows(seed, name, Source.ADDED, distances(baseline.added, train))
        fitted = probabilities(training_set(train, baseline), test, factory=factory)
        scores += _scoped(seed, name, Source.AUGMENTED, utility(test, fitted, cells))
        if name in experiment.replica_arms:
            scores += _replica(
                arms[name], parts, cells, attempts=experiment.attempts, name=name, factory=factory
            )
    return scores


def _fill(
    runner: Runner, train: pd.DataFrame, request: Request, *, attempts: int, name: str
) -> Baseline:
    """Fill `request`, retrying a refusal with a derived seed; a contract break is never retried.

    Raises:
        BaselineError: The last refusal, if every attempt is refused.
    """
    seed = request.seed
    for attempt in range(1, attempts):
        try:
            return _logged(runner(train.copy(), request), seed=seed, name=name)
        except OutputContractError:
            raise
        except BaselineError as exc:
            logger.warning("retry holdout=%d arm=%s attempt=%d reason=%s", seed, name, attempt, exc)
        request = Request(
            quota=request.quota, strata=request.strata, seed=derived_seed(seed, "attempt", attempt)
        )
    return _logged(runner(train.copy(), request), seed=seed, name=name)


def _logged(baseline: Baseline, *, seed: int, name: str) -> Baseline:
    digest = hashlib.sha256(baseline.added.to_csv(lineterminator="\n").encode()).hexdigest()
    logger.info("arm holdout=%d arm=%s added=%d sha256=%s", seed, name, len(baseline.added), digest)
    return baseline


def _replica(
    runner: Runner,
    parts: Split,
    cells: tuple[Stratum, ...],
    *,
    attempts: int,
    name: str,
    factory: Factory,
) -> list[Score]:
    request = replica_request(parts.train, seed=parts.seed)
    replica = _fill(runner, parts.train, request, attempts=attempts, name=f"{name}-replica").added
    fitted = probabilities(replica.assign(weight=1.0), parts.test, factory=factory)
    return [
        *_scoped(parts.seed, name, Source.REPLICA, utility(parts.test, fitted, cells)),
        *_rows(parts.seed, name, Source.REPLICA, fidelity(parts.train, replica)),
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
