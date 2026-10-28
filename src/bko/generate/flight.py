"""Calls in flight: a wave's calls go out in plan order and their outcomes come back in it.

A `Flight` sends one wave of planned calls at a time, with at most `Limits.in_flight` in flight.
It numbers and logs each call's request in plan order (`bko.generate.calls.request`), so call
numbers, and a replay's answers, follow the plan whatever order replies land in. Each reply or
failure is written as it lands, on the event loop's thread. The wave returns its batches in call
order, or raises the failure of the lowest failing call once every call already sent has landed;
no call is sent after one fails. Which calls form a wave is the caller's rule (ADR
`research/concurrent-call-waves`). A `Flight` used in a `with` block runs every wave on one event
loop, which an async client's connections need; outside one, each wave gets its own loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Self

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.generate.calls import Batch, Pending, answer, fail, request
from bko.generate.guidance import Guidance
from bko.generate.log import CallLog
from bko.generate.messages import AnySend, Reply


logger = logging.getLogger(__name__)

IN_FLIGHT: Final = 8
"""The calls a run keeps in flight at once unless told otherwise: its rate budget at the API."""


@dataclass(frozen=True, slots=True)
class Limits:
    """How a run's calls are paced.

    Attributes:
        in_flight: The most calls in flight at once.
        idle: The seconds a call may go without a reply, or None for no limit.
    """

    in_flight: int
    idle: float | None

    def __post_init__(self) -> None:
        """Refuse fewer than one call in flight.

        Raises:
            ValueError: If `in_flight` is below 1.
        """
        if self.in_flight < 1:
            msg = f"in_flight must be at least 1, not {self.in_flight}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Planned:
    """One planned call.

    Attributes:
        guidance: What the call says about its cell.
        rows: The records it asks for.
        skeleton: Reserved for a record skeleton; it must be None.
    """

    guidance: Guidance
    rows: int
    skeleton: object | None = None


class Flight:
    """Sends a run's waves of calls, at most `limits.in_flight` at once.

    Attributes:
        log: The run's call log.
        limits: How the calls are paced.
    """

    def __init__(
        self,
        send: AnySend,
        log: CallLog,
        limits: Limits,
        *,
        spendable: Callable[[], None],
    ) -> None:
        """Hold what each wave needs; nothing is sent until a wave is.

        Args:
            send: Makes each call; its reply is awaited when it is awaitable.
            log: The run's call log.
            limits: How the calls are paced.
            spendable: Called before each call is requested; it raises to refuse the call.
        """
        self.log = log
        self.limits = limits
        self._send = send
        self._spendable = spendable
        self._runner: asyncio.Runner | None = None

    def __enter__(self) -> Self:
        """Start the event loop every wave runs on."""
        self._runner = asyncio.Runner()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the event loop; never suppress."""
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def wave(
        self,
        planned: Sequence[Planned],
        *,
        frame: FrameSpec = UCI_FRAME,
        on_batch: Callable[[Batch], None] | None = None,
    ) -> tuple[Batch, ...]:
        """Send `planned`'s calls, requested in its order, and return their batches in that order.

        Args:
            planned: The wave's calls, in plan order.
            frame: The frame whose records each call asks for.
            on_batch: Called once every call sent has landed, with each batch in call order up
                to the first call that failed or was not sent, as a one-call-at-a-time run would
                have taken them before it stopped.

        Returns:
            Each call's outcome, in call order.

        Raises:
            ValueError: As `bko.generate.calls.request`, before that call is requested.
            GenerationError: The lowest failing call's failure, once every call sent has landed.
            Exception: Anything `spendable` or `on_batch` raises, once every call sent has landed.
            KeyboardInterrupt: Propagated unchanged; each call still in flight is cancelled first
                and writes its `error` line.
        """
        landed, error = self._run(planned, frame)
        for batch in landed:
            if on_batch is not None:
                on_batch(batch)
        if error is not None:
            raise error
        return landed

    def _run(
        self, planned: Sequence[Planned], frame: FrameSpec
    ) -> tuple[tuple[Batch, ...], Exception | None]:
        """The wave's batches before its first failure, and that failure or None."""
        if self._runner is None:
            with asyncio.Runner() as runner:
                return runner.run(self._wave(planned, frame))
        try:
            return self._runner.run(self._wave(planned, frame))
        except BaseException:
            # An interrupt raised inside a call leaves the others on the loop: closing it cancels
            # them, so their `error` lines are written before the caller's next line.
            self._runner.close()
            self._runner = asyncio.Runner()
            raise

    async def _wave(
        self, planned: Sequence[Planned], frame: FrameSpec
    ) -> tuple[tuple[Batch, ...], Exception | None]:
        slots = asyncio.Semaphore(self.limits.in_flight)
        failed = asyncio.Event()
        flying: list[asyncio.Task[Batch]] = []
        stopped: Exception | None = None
        try:
            await self._dispatch(planned, frame, slots, failed, flying)
        except Exception as exc:
            # Calls already sent are paid for: let them land and be logged before stopping.
            stopped = exc
            await asyncio.gather(*flying, return_exceptions=True)
        except BaseException:
            for task in flying:
                task.cancel()
            await asyncio.gather(*flying, return_exceptions=True)
            raise
        # Every task's exception is read, so none is reported as never retrieved.
        errors = [task.exception() for task in flying]
        landed: list[Batch] = []
        for task, error in zip(flying, errors, strict=True):
            if error is not None:
                if not isinstance(error, Exception):
                    raise error
                return tuple(landed), error
            landed.append(task.result())
        return tuple(landed), stopped

    async def _dispatch(
        self,
        planned: Sequence[Planned],
        frame: FrameSpec,
        slots: asyncio.Semaphore,
        failed: asyncio.Event,
        flying: list[asyncio.Task[Batch]],
    ) -> None:
        """Request each planned call in order as a slot frees, until one fails; await them all."""
        for item in planned:
            await slots.acquire()
            if failed.is_set():
                slots.release()
                break
            try:
                self._spendable()
                pending = request(
                    item.guidance, item.rows, log=self.log, frame=frame, skeleton=item.skeleton
                )
            except BaseException:
                slots.release()
                raise
            flying.append(asyncio.create_task(self._fly(pending, slots, failed)))
        if flying:
            await asyncio.wait(flying)

    async def _fly(
        self, pending: Pending, slots: asyncio.Semaphore, failed: asyncio.Event
    ) -> Batch:
        try:
            try:
                sent = self._send(pending.call)
                reply = sent if isinstance(sent, Reply) else await sent
                return answer(pending, reply, log=self.log)
            except BaseException as exc:
                fail(pending, exc, log=self.log)
                failed.set()
                raise
        finally:
            slots.release()
