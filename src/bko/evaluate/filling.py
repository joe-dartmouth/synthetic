"""One arm's fill of one request, retried on a refusal, and the `arm` line each fill logs.

`bko.evaluate.run` fills every arm's gap and replica requests through `fill_arm`; a shortfall
becomes a partial fill as `bko.evaluate.shortfall` decides. Each fill comes back as a `Filled`,
which holds its `arm` line byte for byte, and a holdout's `Logged` collects them with its `split`
line, so `bko.evaluate.shared` checks what the harness logged without reading its logger.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from bko.baselines.errors import BaselineError, OutputContractError
from bko.baselines.fills import CURRENT_FILL, Fill, FillKind
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, derived_seed
from bko.baselines.request import Request
from bko.evaluate.data import digest
from bko.evaluate.errors import EvaluationError
from bko.evaluate.shortfall import partial_fill, shortfall_kept, structural


logger = logging.getLogger("bko.evaluate.run")


@dataclass(frozen=True, slots=True)
class Filled:
    """One returned fill and the `arm` line it logged.

    Attributes:
        name: The `arm` line's arm: the arm's name, or `<arm>-replica` for a replica fill.
        kind: Whether it filled the gap or the replica request.
        attempt: The attempt that returned.
        baseline: The rows and params it returned.
        line: The `arm` line's message, byte for byte as logged.
    """

    name: str
    kind: FillKind
    attempt: int
    baseline: Baseline
    line: str


@dataclass(slots=True)
class Logged:
    """One holdout's `split` line and every fill's `Filled`, in the order the harness logs them.

    Attributes:
        split: The `split` line's message; None until the harness logs it.
        fills: Each returned fill, in fill order.
    """

    split: str | None = None
    fills: list[Filled] = field(default_factory=list)


def fill_arm(
    runner: Runner,
    train: pd.DataFrame,
    request: Request,
    *,
    holdout: int,
    attempts: int,
    name: str,
    kind: FillKind,
) -> Filled:
    """Fill `request`, retrying a refusal with a derived seed; a contract break is never retried.

    On a gap fill, a generator still short at its row limit on the last attempt returns the rows
    it kept (`bko.evaluate.shortfall.partial_fill`) rather than refusing; a replica fill still
    refuses, since its scores read it as a copy of every training cell. On a gap fill whose
    shortfall falls on strata of fewer than 2 training rows alone (`structural`), it fills
    partially on any attempt, since no retry changes those rows; a shortfall that also falls on
    another stratum, and a plain refusal, are retried as any other.

    The runner gets a copy of `train` on every attempt, so one that changes its input changes
    nothing the harness reads. `CURRENT_FILL` names the holdout and `kind` around each call.

    Returns:
        The returned fill, with the `arm` line it logged.

    Raises:
        BaselineError: The last refusal, if every attempt is refused, or a contract break.
        EvaluationError: If `attempts` is below 1.
    """
    first = request.seed
    thin = structural(train, request) if kind == "gap" else frozenset()
    for attempt in range(1, attempts + 1):
        try:
            baseline = _call(runner, train, request, Fill(holdout, kind))
        except OutputContractError:
            # The context only; the caller logs the error itself, without a traceback.
            logger.error(  # ruff: ignore[error-instead-of-exception]
                "contract holdout=%d arm=%s attempt=%d", holdout, name, attempt
            )
            raise
        except BaselineError as exc:
            last = attempt == attempts and kind == "gap"
            kept = shortfall_kept(exc)
            if kept is not None and (last or (thin and set(kept[1]) <= thin)):
                baseline = partial_fill(*kept, train, request, fill=(holdout, name, attempt))
                line = log_arm(
                    baseline, holdout=holdout, name=name, attempt=attempt, seed=request.seed
                )
                return Filled(name, kind, attempt, baseline, line)
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
        line = log_arm(baseline, holdout=holdout, name=name, attempt=attempt, seed=request.seed)
        return Filled(name, kind, attempt, baseline, line)
    msg = f"attempts {attempts}: an arm needs at least one attempt"
    raise EvaluationError(msg)


def _call(runner: Runner, train: pd.DataFrame, request: Request, fill: Fill) -> Baseline:
    """`runner` on a copy of `train`, with `CURRENT_FILL` set to `fill` for the call alone."""
    token = CURRENT_FILL.set(fill)
    try:
        return runner(train.copy(), request)
    finally:
        CURRENT_FILL.reset(token)


def arm_line(baseline: Baseline, *, holdout: int, name: str, attempt: int, seed: int) -> str:
    """A fill's `arm` line: its holdout, arm, attempt, seed, rows, sha256 and params."""
    params = json.dumps(baseline.params, sort_keys=True, default=str)
    return (
        f"arm holdout={holdout:d} arm={name} attempt={attempt:d} seed={seed:d}"
        f" added={len(baseline.added):d} sha256={digest(baseline.added)} params={params}"
    )


def log_arm(baseline: Baseline, *, holdout: int, name: str, attempt: int, seed: int) -> str:
    """Log a fill's `arm_line` and return it."""
    line = arm_line(baseline, holdout=holdout, name=name, attempt=attempt, seed=seed)
    logger.info("%s", line)
    return line
