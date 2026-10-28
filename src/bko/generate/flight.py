"""Calls in flight: a wave's calls go out in plan order and their outcomes come back in it.

A `Flight` sends one wave of planned calls at a time, with at most `Limits.in_flight` in flight.
It numbers and logs each call's request in plan order (`bko.generate.calls.request`), so call
numbers, and a replay's answers, follow the plan whatever order replies land in. Each reply or
failure is written as it lands, on the event loop's thread. The wave returns its batches in call
order, or raises the failure of the lowest failing call once every call already sent has landed;
no call is sent after one fails. Which calls form a wave is the caller's rule (ADR
`research/concurrent-call-waves`). A `Flight` used in a `with` block runs every wave on one event
loop, which an async client's connections need; outside one, each wave gets its own loop.

The stop rules hold as they do one call at a time. Before each call, every call still in flight
reserves its most: `max_tokens` of output and its request's JSON length plus `OVERHEAD_TOKENS` of
input, at the model's `bko.generate.spend.PRICES`. When the log's spend plus those reservations
reaches the ceiling, the call waits for them to land, and the caller's exact check then reads the
spend a one-call-at-a-time run reads there. Each awaited send runs under `asyncio.timeout` at the
idle limit, its retries included: on expiry the call raises `IdleError`, and the calls beside it are
cancelled, each writing its `error` line, in call order. Each call's task sets
`bko.generate.messages.CALL_NUMBER`, so its retry lines name it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Final, Self

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.generate.calls import Batch, Pending, answer, fail, request
from bko.generate.errors import IdleError
from bko.generate.guidance import Guidance
from bko.generate.log import CallLog
from bko.generate.messages import CALL_NUMBER, AnySend, Reply
from bko.generate.spend import price_of


logger = logging.getLogger(__name__)

IN_FLIGHT: Final = 8
"""The calls a run keeps in flight at once unless told otherwise: its rate budget at the API."""

OVERHEAD_TOKENS: Final = 2000
"""Input tokens a call's reservation adds to its request's JSON length, for what the API adds
around it: the output schema's grammar and the message framing."""


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


@dataclass(eq=False, slots=True)
class _Wave:
    """One wave's state on the event loop.

    Attributes:
        slots: The free places in flight.
        failed: Set once a call fails, so no later call is sent.
        flying: Each call sent, by its number, in call order.
        reserved: Each call still in flight's reservation in USD, by its number.
        idle: The first call's `IdleError`, once one has fired.
    """

    slots: asyncio.Semaphore
    failed: asyncio.Event = field(default_factory=asyncio.Event)
    flying: dict[int, asyncio.Task[Batch]] = field(default_factory=dict)
    reserved: dict[int, float] = field(default_factory=dict)
    idle: IdleError | None = None


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
        ceiling: float = math.inf,
    ) -> None:
        """Hold what each wave needs; nothing is sent until a wave is.

        Args:
            send: Makes each call; its reply is awaited when it is awaitable.
            log: The run's call log.
            limits: How the calls are paced.
            spendable: The exact spend check, called before each call is requested, once the
                calls in flight whose reservations could reach `ceiling` have landed; it raises
                to refuse the call.
            ceiling: The spend in USD at which `spendable` refuses; the reservations are read
                against it.
        """
        self.log = log
        self.limits = limits
        self._send = send
        self._spendable = spendable
        self._ceiling = ceiling
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
            IdleError: If a call got no reply within `limits.idle` seconds, once the calls beside
                it are cancelled.
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
        wave = _Wave(asyncio.Semaphore(self.limits.in_flight))
        stopped: Exception | None = None
        try:
            await self._dispatch(planned, frame, wave)
        except Exception as exc:
            # Calls already sent are paid for: let them land and be logged before stopping.
            stopped = exc
            await asyncio.gather(*wave.flying.values(), return_exceptions=True)
        except BaseException:
            for task in wave.flying.values():
                task.cancel()
            await asyncio.gather(*wave.flying.values(), return_exceptions=True)
            raise
        landed: list[Batch] = []
        # Every task's exception is read, so none is reported as never retrieved.
        outcomes = [_outcome(task, wave) for task in wave.flying.values()]
        for outcome in outcomes:
            if isinstance(outcome, Batch):
                landed.append(outcome)
                continue
            if not isinstance(outcome, Exception):
                raise outcome
            return tuple(landed), outcome
        return tuple(landed), stopped

    async def _dispatch(self, planned: Sequence[Planned], frame: FrameSpec, wave: _Wave) -> None:
        """Request each planned call in order as a slot frees, until one fails; await them all."""
        for item in planned:
            await wave.slots.acquire()
            try:
                pending = await self._request(item, frame, wave)
            except BaseException:
                wave.slots.release()
                raise
            if pending is None:
                wave.slots.release()
                break
            wave.reserved[pending.index] = _reservation(pending)
            wave.flying[pending.index] = asyncio.create_task(self._fly(pending, wave))
        if wave.flying:
            await asyncio.wait(wave.flying.values())

    async def _request(self, item: Planned, frame: FrameSpec, wave: _Wave) -> Pending | None:
        """`item`'s call, requested once the ceiling allows it, or None once a call has failed."""
        if not wave.failed.is_set():
            await self._afford(wave)
        if wave.failed.is_set():
            return None
        self._spendable()
        return request(item.guidance, item.rows, log=self.log, frame=frame, skeleton=item.skeleton)

    async def _afford(self, wave: _Wave) -> None:
        """Wait for the calls in flight if the spend plus their reservations reaches the ceiling."""
        reserved = sum(wave.reserved.values())
        if reserved and self.log.usd + reserved >= self._ceiling:
            await asyncio.wait([wave.flying[index] for index in wave.reserved])

    async def _fly(self, pending: Pending, wave: _Wave) -> Batch:
        CALL_NUMBER.set(pending.index)
        try:
            try:
                reply = await self._reply(pending)
                return answer(pending, reply, log=self.log)
            except BaseException as exc:
                fail(pending, exc, log=self.log)
                wave.failed.set()
                if isinstance(exc, IdleError) and wave.idle is None:
                    wave.idle = exc
                    await _cancel_beside(pending.index, wave)
                raise
        finally:
            wave.reserved.pop(pending.index, None)
            wave.slots.release()

    async def _reply(self, pending: Pending) -> Reply:
        """`pending`'s reply, awaited under the idle limit when the send returns an awaitable.

        Raises:
            IdleError: If no reply came within `limits.idle` seconds.
        """
        sent = self._send(pending.call)
        if isinstance(sent, Reply):
            return sent
        idle = self.limits.idle
        scope = asyncio.timeout(idle)
        try:
            async with scope:
                return await sent
        except TimeoutError:
            if not scope.expired() or idle is None:
                raise
            logger.warning("stop rule=idle call=%d limit_seconds=%.1f", pending.index, idle)
            msg = f"call {pending.index} got no reply within the idle limit of {idle:g} s"
            raise IdleError(msg) from None


def _reservation(pending: Pending) -> float:
    """The most `pending` can cost in USD: all its output tokens, and its request's input."""
    settings = pending.call.settings
    price = price_of(settings.model)
    text = len(json.dumps(pending.call.params()))
    return (settings.max_tokens * price.output + (text + OVERHEAD_TOKENS) * price.input) / 1e6


async def _cancel_beside(index: int, wave: _Wave) -> None:
    """Cancel each call in flight but call `index`, in call order, and wait for each to land."""
    for other, task in list(wave.flying.items()):
        if other != index and not task.done():
            task.cancel()
            await asyncio.wait([task])


def _outcome(task: asyncio.Task[Batch], wave: _Wave) -> Batch | BaseException:
    """A landed task's batch or exception; a call the idle rule cancelled gives its `IdleError`."""
    if task.cancelled():
        return wave.idle if wave.idle is not None else asyncio.CancelledError()
    error = task.exception()
    return task.result() if error is None else error
