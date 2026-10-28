"""An operator's stop of a supervised run: a signal to the supervisor, or a `STOP` file.

`bko.evaluate.supervise` installs `OperatorStop.handle` for `SIGTERM`, `SIGINT` and `SIGHUP` while
it runs. The first signal marks the run stopping and is forwarded, once, as `SIGINT` to the
segment's child, which `score` turns into an `interrupted` end line (`bko.evaluate.scoring`); the
supervisor keeps waiting for the child, with no kill timeout, and resumes nothing after it. Each
child runs in its own process group, so a signal to the supervisor's group reaches the child only
through that one forward. A `STOP` file in the run's folder marks the run stopping too, read when a
segment ends and through each backoff, so a child killed alone (`kill -9`, which reads as an
out-of-memory kill) is not resumed. `docs/DEVELOPMENT.md` (Evaluation, Supervision) holds the
operator's commands.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from collections.abc import Generator
from pathlib import Path
from types import FrameType
from typing import Final


STOP_FILE: Final = "STOP"
"""The file in a run's folder whose presence stops its supervisor before the next resume."""

SIGNALS: Final = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
"""The signals that stop a supervised run."""

STEP_SECONDS: Final = 0.2
"""How often a backoff rereads the stop: a stop lands within this long of its signal or file."""


class OperatorStop:
    """Whether the operator stopped the run, and the child the stop is forwarded to.

    Attributes:
        by: What stopped the run: the signal's name, `STOP` for the file, or None.
    """

    def __init__(self, stop_file: Path) -> None:
        """Watch `stop_file`; nothing has stopped the run yet."""
        self.by: str | None = None
        self._stop_file = stop_file
        self._child: int | None = None
        self._forwarded = False

    @contextlib.contextmanager
    def installed(self) -> Generator[OperatorStop]:
        """Handle `SIGNALS` while the body runs, then restore the handlers before it."""
        previous = {number: signal.signal(number, self.handle) for number in SIGNALS}
        try:
            yield self
        finally:
            for number, handler in previous.items():
                signal.signal(number, signal.SIG_DFL if handler is None else handler)

    def handle(self, number: int, frame: FrameType | None) -> None:
        """Mark the run stopping by the signal `number` and forward it to the child, once."""
        del frame
        if self.by is None:
            self.by = signal.Signals(number).name
        self._forward()

    def started(self, pid: int) -> None:
        """Record the child `pid`; a stop that came before it started is forwarded now."""
        self._child = pid
        if self.by is not None:
            self._forward()

    def ended(self) -> None:
        """Forget the child once it has exited, so no stop is forwarded to its reused pid."""
        self._child = None

    @property
    def stopping(self) -> bool:
        """Whether a signal or the `STOP` file stopped the run; reading it checks the file."""
        if self.by is None and self._stop_file.exists():
            self.by = STOP_FILE
        return self.by is not None

    def wait(self, seconds: float) -> None:
        """Wait `seconds`, or less once the run is stopping."""
        deadline = time.monotonic() + seconds
        while not self.stopping and (left := deadline - time.monotonic()) > 0:
            time.sleep(min(STEP_SECONDS, left))

    def _forward(self) -> None:
        """Send the child one `SIGINT`, if a child runs and none was sent."""
        if self._child is None or self._forwarded:
            return
        self._forwarded = True
        with contextlib.suppress(ProcessLookupError):
            os.kill(self._child, signal.SIGINT)
