"""The stop rules of a paid run: its spend projection and its idle limit.

ADR `research/e1-evaluation-protocol` records the ceiling, and `docs/DEVELOPMENT.md` (Evaluation)
the protocol these rules belong to. A `Watch` reads two things a paid run can get wrong before the
ceiling is reached: a spend that, scaled from the first holdouts to every holdout, passes the
ceiling, and a call that gets no reply within the idle limit, its retries included. Each rule's
values are the run's `Experiment`'s, each firing logs the measure that fired, and each refuses with
an `EvaluationError`, so the command exits 2 and the call log ends `failed` (`bko.generate.log`).
The idle rule is a timer armed for each call: when it runs out it interrupts the main thread, which
breaks a stalled read, and `call` turns that interrupt into the refusal. A replay spends nothing
live, so its projection never fires.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Protocol

from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment


logger = logging.getLogger(__name__)


class Timer(Protocol):
    """A one-shot timer, as `threading.Timer` is."""

    daemon: bool

    def start(self) -> None:
        """Start counting."""

    def cancel(self) -> None:
        """Stop counting; the action never runs if it has not."""


type Timers = Callable[[float, Callable[[], None]], Timer]
"""Build a timer that runs an action after some seconds."""


def interrupt_main() -> None:
    """Send SIGINT to the main thread, so a read blocked there returns and raises."""
    signal.pthread_kill(threading.main_thread().ident or 0, signal.SIGINT)


class Watch:
    """The stop rules of one run, read at each call and each finished holdout.

    Attributes:
        projecting: Whether the projection rule applies.
        fired: Whether the idle rule's timer ran out.
    """

    def __init__(
        self,
        experiment: Experiment,
        spend: Callable[[], float],
        *,
        projecting: bool = True,
        timers: Timers = threading.Timer,
        interrupt: Callable[[], None] = interrupt_main,
    ) -> None:
        """Hold the rules' values.

        Args:
            experiment: The run's configuration: its holdouts, ceiling and rule values.
            spend: The run's live spend in USD.
            projecting: Whether the projection rule applies; a run that resumes another counts the
                holdouts it replays, so its live spend cannot be scaled by them.
            timers: Builds the idle rule's timer for each call.
            interrupt: What the timer runs when the limit passes; it interrupts the main thread.
        """
        self._holdouts = len(experiment.holdout_seeds)
        self._after = experiment.projection_after
        self._ceiling = experiment.spend_ceiling
        self._idle = experiment.idle_limit
        self._spend = spend
        self.projecting = projecting and self._after < self._holdouts
        self._timers = timers
        self._interrupt = interrupt
        self.fired = False
        self._done = 0
        logger.info(
            "stop rules projecting=%s projection_after=%d idle_limit_seconds=%.1f",
            self.projecting,
            self._after,
            self._idle,
        )

    @contextmanager
    def call(self) -> Generator[None]:
        """Run one call under the idle limit.

        Raises:
            EvaluationError: If the call got no reply within `idle_limit` seconds.
        """
        timer = self._timers(self._idle, self._fire)
        timer.daemon = True
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            if not self.fired:
                raise
            msg = f"a call got no reply within the idle limit of {self._idle:.0f} s"
            raise EvaluationError(msg) from None
        finally:
            timer.cancel()

    def _fire(self) -> None:
        self.fired = True
        logger.warning("stop rule=idle limit_seconds=%.1f", self._idle)
        self._interrupt()

    def holdout_done(self) -> None:
        """Count a finished holdout; refuse the run if its spend projects to the ceiling.

        The projection is read once, after `projection_after` holdouts: the live spend over those
        holdouts, scaled to every holdout. A projection that reaches the ceiling refuses, as the
        ceiling itself stops a run that reaches it.

        Raises:
            EvaluationError: If the projection reaches the spend ceiling.
        """
        self._done += 1
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
            raise EvaluationError(msg)
