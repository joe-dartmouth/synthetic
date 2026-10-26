"""The stop rules of a paid run: its spend projection and its idle limit.

ADR `research/e1-evaluation-protocol` records the ceiling, and `docs/DEVELOPMENT.md` (Evaluation)
the protocol these rules belong to. A `Watch` reads two things a paid run can get wrong before the
ceiling is reached: a spend that, scaled from the first holdouts to every holdout, passes the
ceiling, and a gap between completed calls longer than the idle limit. Each rule's values are the
run's `Experiment`'s, each firing logs the measure that fired, and each refuses with an
`EvaluationError`, so the command exits 2 and the call log ends `failed` (`bko.generate.log`). The
idle rule is read before each call, so a call still in flight is bounded by the client's own
timeout, not by this rule. A replay spends nothing live, so its projection never fires.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment


logger = logging.getLogger(__name__)


class Watch:
    """The stop rules of one run, read at each call and each finished holdout."""

    def __init__(
        self,
        experiment: Experiment,
        spend: Callable[[], float],
        *,
        projecting: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Hold the rules' values; the idle clock starts now.

        Args:
            experiment: The run's configuration: its holdouts, ceiling and rule values.
            spend: The run's live spend in USD.
            projecting: Whether the projection rule applies; a run that resumes another counts the
                holdouts it replays, so its live spend cannot be scaled by them.
            clock: Seconds on a monotonic clock.
        """
        self._holdouts = len(experiment.holdout_seeds)
        self._after = experiment.projection_after
        self._ceiling = experiment.spend_ceiling
        self._idle = experiment.idle_limit
        self._spend = spend
        self._projecting = projecting
        self._clock = clock
        self._done = 0
        self._last = clock()

    def before_call(self) -> None:
        """Refuse the next call if none completed within the idle limit.

        Raises:
            EvaluationError: If more than `idle_limit` seconds passed since the last completed
                call, or since the run began.
        """
        idle = self._clock() - self._last
        if idle > self._idle:
            logger.warning("stop rule=idle idle_seconds=%.1f limit_seconds=%.1f", idle, self._idle)
            msg = f"no call completed in {idle:.0f} s, past the idle limit of {self._idle:.0f} s"
            raise EvaluationError(msg)

    def call_done(self) -> None:
        """Record a completed call."""
        self._last = self._clock()

    def holdout_done(self) -> None:
        """Count a finished holdout; refuse the run if its spend projects past the ceiling.

        The projection is read once, after `projection_after` holdouts: the live spend over those
        holdouts, scaled to every holdout.

        Raises:
            EvaluationError: If the projection exceeds the spend ceiling.
        """
        self._done += 1
        if not self._projecting or self._done != self._after or self._after >= self._holdouts:
            return
        spent = self._spend()
        projected = spent / self._done * self._holdouts
        logger.info(
            "projection holdouts=%d of=%d usd=%.4f projected_usd=%.4f ceiling=%.2f",
            self._done,
            self._holdouts,
            spent,
            projected,
            self._ceiling,
        )
        if projected > self._ceiling:
            logger.warning(
                "stop rule=projection projected_usd=%.4f ceiling=%.2f", projected, self._ceiling
            )
            msg = (
                f"the run's spend of {spent:.4f} USD over {self._done} holdouts projects to "
                f"{projected:.4f} USD over {self._holdouts}, past its ceiling of "
                f"{self._ceiling} USD"
            )
            raise EvaluationError(msg)
