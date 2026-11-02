"""The spend projection of a paid run, one of its stop rules.

ADR `research/e1-evaluation-protocol` records the ceiling, and `docs/DEVELOPMENT.md` (Evaluation)
the protocol these rules belong to. A `Watch` reads a spend that, scaled from the first holdouts to
every holdout, passes the ceiling, before the ceiling is reached. Its values are the run's
`Experiment`'s; when it fires it logs the measure that fired and refuses with a `StopRuleError`, so
the command exits 2 and the call log ends `failed` (`bko.generate.log`). It logs the run's idle
limit too, which `bko.generate.flight` holds each call to. The spend is the run's whole spend, so a
resume, which replays the stopped run's holdouts and counts their spend again, projects as the
fresh run would; a replay, which spends nothing, is not projected.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from bko.evaluate.errors import StopRuleError
from bko.evaluate.experiment import Experiment


logger = logging.getLogger(__name__)


class Watch:
    """The spend projection of one run, read at each finished holdout.

    Attributes:
        projecting: Whether the projection rule applies.
    """

    def __init__(
        self,
        experiment: Experiment,
        spend: Callable[[], float],
        *,
        projecting: bool = True,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """Hold the rules' values.

        Args:
            experiment: The run's configuration: its holdouts, ceiling and rule values.
            spend: The run's whole spend in USD: carried, replayed and live, so a resume scales
                the spend of every holdout it has done, replayed ones included.
            projecting: Whether the projection rule applies; a replay's does not.
            on_done: Called at each finished holdout before the projection is read, so the call
                log counts the holdouts a ceiling or projection stop records.
        """
        self._holdouts = len(experiment.holdout_seeds)
        self._after = experiment.projection_after
        self._ceiling = experiment.spend_ceiling
        self._idle = experiment.idle_limit
        self._spend = spend
        self._on_done = on_done
        self.projecting = projecting and self._after < self._holdouts
        self._done = 0
        logger.info(
            "stop rules projecting=%s projection_after=%d idle_limit_seconds=%.1f",
            self.projecting,
            self._after,
            self._idle,
        )

    def holdout_done(self) -> None:
        """Count a finished holdout; refuse the run if its spend projects to the ceiling.

        The projection is read once, after `projection_after` holdouts: the run's whole spend over
        those holdouts, scaled to every holdout. A projection that reaches the ceiling refuses, as
        the ceiling itself stops a run that reaches it.

        Raises:
            StopRuleError: If the projection reaches the spend ceiling.
        """
        self._done += 1
        if self._on_done is not None:
            self._on_done()
        if not self.projecting or self._done != self._after:
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
        if projected >= self._ceiling:
            logger.warning(
                "stop rule=projection projected_usd=%.4f ceiling=%.2f", projected, self._ceiling
            )
            msg = (
                f"the run's spend of {spent:.4f} USD over {self._done} holdouts projects to "
                f"{projected:.4f} USD over {self._holdouts}, reaching its ceiling of "
                f"{self._ceiling} USD"
            )
            raise StopRuleError(msg, "projection")
