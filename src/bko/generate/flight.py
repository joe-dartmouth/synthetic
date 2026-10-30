"""Calls in flight: a wave's calls go out in plan order and their outcomes come back in it.

A `Flight` sends one wave of planned calls at a time, with at most `Limits.in_flight` in flight.
It numbers and logs each call's request in plan order (`bko.generate.calls.request`), so call
numbers, and a replay's answers, follow the plan whatever order replies land in. Each reply or
failure is written as it lands, on the event loop's thread. The wave returns its batches in call
order, or raises the failure of the lowest failing call once every call already sent has landed,
unless a stop rule's error outranks it; no call is sent after one fails. Which calls form a wave
is the caller's rule (ADR `research/concurrent-call-waves`). A `Flight` used in a `with` block
runs every wave on one event loop, which an async client's connections need; outside one, each
wave gets its own loop.

The stop rules hold as they do one call at a time. Before each call, every call still in flight
reserves its most: `max_tokens` of output and its request's JSON length plus `OVERHEAD_TOKENS` of
input, at the model's `bko.generate.spend.PRICES`; the input is priced at the higher of `input` and
`cache_write`, since a call may bill its prompt as a cache write. When the log's spend plus those
reservations reach the ceiling, the call waits for them to land, and the caller's exact check then
reads the spend a one-call-at-a-time run reads there. Each awaited send runs under
`asyncio.timeout` at the idle limit, its retries included: on expiry the call raises `IdleError`
carrying its attempt count and whether a retry met a 429, no later call is sent, and the calls
beside it land under their own idle limits, so their replies and spend are recorded, before the
lowest idle call's `IdleError` is raised. An interrupt cancels the calls in flight; each requested
call still gets its `error` line, recording `KeyboardInterrupt`, before the call log can close, a
further interrupt notwithstanding. Each call's task sets `bko.generate.messages.CALL_NUMBER`, so
its retry lines name it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Final, Self

from bko.datasets.frames import FrameSpec
from bko.generate.calls import Batch, Pending, answer, fail, request
from bko.generate.errors import IdleError
from bko.generate.guidance import Guidance
from bko.generate.log import CallLog
from bko.generate.messages import ATTEMPTS, CALL_NUMBER, AnySend, Reply, Tries
from bko.generate.prompt import Shown
from bko.generate.skeleton import Skeleton
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
        skeleton: A prescribing request's lines, one per record asked; None for an arm that
            prescribes none.
        shown: The training records the call shows; None for an arm that shows none.
    """

    guidance: Guidance
    rows: int
    skeleton: Skeleton | None = None
    shown: Shown | None = None


@dataclass(eq=False, slots=True)
class _Wave:
    """One wave's state on the event loop.

    Attributes:
        slots: The free places in flight.
        failed: Set once a call fails, so no later call is sent.
        flying: Each call sent, by its number, in call order.
        pending: Each call requested, by its number.
        answered: The numbers of the calls whose `reply` or `error` line is written.
        reserved: Each call still in flight's reservation in USD, by its number.
        idle: Each call's `IdleError`, by its number, once it has fired.
        cause: What first stopped the wave's loop, or None while nothing has. An interrupt is
            recorded as `KeyboardInterrupt`, whether raised as one or delivered as a cancellation
            of the wave's own task, which `asyncio.Runner`'s SIGINT handler makes on the main
            thread; any other `BaseException` is recorded as itself.
    """

    slots: asyncio.Semaphore
    failed: asyncio.Event = field(default_factory=asyncio.Event)
    flying: dict[int, asyncio.Task[Batch]] = field(default_factory=dict)
    pending: dict[int, Pending] = field(default_factory=dict)
    answered: set[int] = field(default_factory=set)
    reserved: dict[int, float] = field(default_factory=dict)
    idle: dict[int, IdleError] = field(default_factory=dict)
    cause: BaseException | None = None

    def stop(self, cause: BaseException) -> None:
        """Record `cause` as what stopped the wave, unless something already has."""
        if self.cause is None:
            self.cause = cause

    def cancelled(self, error: BaseException) -> BaseException:
        """What a call's `error` line records for `error`.

        A cancellation an interrupt caused records `KeyboardInterrupt`; anything else, itself.
        """
        interrupted = isinstance(self.cause, KeyboardInterrupt)
        if isinstance(error, asyncio.CancelledError) and interrupted:
            return KeyboardInterrupt()
        return error


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
        frame: FrameSpec,
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
            IdleError: The lowest idle call's, if a call got no reply within `limits.idle`
                seconds, once the calls beside it have landed under their own idle limits.
            GenerationError: The lowest failing call's failure, once every call sent has landed.
            Exception: Anything `spendable` or `on_batch` raises, once every call sent has landed;
                what `spendable` raises outranks a call's failure when it carries a `stop_rule`.
            KeyboardInterrupt: Propagated unchanged; each call still in flight is cancelled first,
                and every requested call has its `error` line.
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
        wave = _Wave(asyncio.Semaphore(self.limits.in_flight))
        if self._runner is None:
            runner = asyncio.Runner()
            try:
                return runner.run(self._wave(planned, frame, wave))
            except BaseException as exc:
                wave.stop(exc)
                raise
            finally:
                self._settle(runner, wave)
        try:
            return self._runner.run(self._wave(planned, frame, wave))
        except BaseException as exc:
            # An interrupt raised inside a call leaves the others on the loop: they are
            # cancelled and landed, so their `error` lines are written before the caller's next.
            wave.stop(exc)
            self._settle(self._runner, wave)
            self._runner = asyncio.Runner()
            raise

    def _settle(self, runner: asyncio.Runner, wave: _Wave) -> None:
        """Land every call still in flight, close `runner`, and log each call that never ran.

        A further `KeyboardInterrupt`, as a repeated Ctrl-C raises once the loop's own handler is
        gone, is absorbed until every call has landed, so each requested call gets its `error`
        line before the call log can close; the first interrupt still propagates.
        """
        # The wave's own task is among them when an interrupt left it waiting on its calls.
        others = asyncio.all_tasks(runner.get_loop()) if wave.flying else set()
        tasks = [*wave.flying.values(), *(others - set(wave.flying.values()))]
        # Cancelled in call order, so their `error` lines are written in it.
        for task in tasks:
            task.cancel()
        while any(not task.done() for task in tasks):
            try:
                runner.get_loop().run_until_complete(asyncio.wait(tasks))
            except KeyboardInterrupt:
                continue
        for task in tasks:
            # Read, so no exception is reported as never retrieved; the first one propagates.
            if not task.cancelled():
                task.exception()
        while True:
            try:
                runner.close()
                break
            except KeyboardInterrupt:
                continue
        # A call cancelled before its first step never ran the code that writes its line.
        for index, pending in wave.pending.items():
            if index not in wave.answered:
                fail(pending, wave.cancelled(asyncio.CancelledError()), log=self.log)
                wave.answered.add(index)

    async def _wave(
        self, planned: Sequence[Planned], frame: FrameSpec, wave: _Wave
    ) -> tuple[tuple[Batch, ...], Exception | None]:
        stopped: Exception | None = None
        try:
            await self._dispatch(planned, frame, wave)
        except Exception as exc:
            # Calls already sent are paid for: let them land and be logged before stopping.
            stopped = exc
            await asyncio.gather(*wave.flying.values(), return_exceptions=True)
        except BaseException as exc:
            # A cancellation of this task itself is an interrupt: the flight cancels it only in
            # `_settle`, after recording what stopped the loop, so a first one is the SIGINT
            # handler's. A call's own `CancelledError` re-raised below is not one.
            current = asyncio.current_task()
            external = current is not None and current.cancelling() > 0
            cancelled = isinstance(exc, asyncio.CancelledError) and external
            wave.stop(KeyboardInterrupt() if cancelled else exc)
            for task in wave.flying.values():
                task.cancel()
            await asyncio.gather(*wave.flying.values(), return_exceptions=True)
            raise
        landed: list[Batch] = []
        # Every task's exception is read, so none is reported as never retrieved.
        outcomes = [_outcome(task) for task in wave.flying.values()]
        for outcome in outcomes:
            if isinstance(outcome, Batch):
                landed.append(outcome)
                continue
            if not isinstance(outcome, Exception):
                raise outcome
            return tuple(landed), self._stop(stopped, outcome, wave)
        return tuple(landed), stopped

    def _stop(self, stopped: Exception | None, failure: Exception, wave: _Wave) -> Exception:
        """What a failed wave raises: a stop rule's error before a call's failure.

        The log's `end` line then still names the rule: the dispatch's stop first, then the
        lowest idle call's `IdleError`, which logs its `stop rule=idle` line.
        """
        if _rule(stopped) and stopped is not None:
            return stopped
        if wave.idle:
            index = min(wave.idle)
            idle = self.limits.idle or 0.0
            logger.warning("stop rule=idle call=%d limit_seconds=%.1f", index, idle)
            return wave.idle[index]
        return failure

    async def _dispatch(self, planned: Sequence[Planned], frame: FrameSpec, wave: _Wave) -> None:
        """Request each planned call in order as a slot frees, until one fails; await them all."""
        for sent, item in enumerate(planned):
            await wave.slots.acquire()
            try:
                pending = await self._request(item, frame, wave)
            except BaseException:
                wave.slots.release()
                raise
            if pending is None:
                wave.slots.release()
                logger.info("halt call=%d unsent=%d", self.log.calls + 1, len(planned) - sent)
                break
            wave.reserved[pending.index] = _reservation(pending)
            wave.pending[pending.index] = pending
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
        return request(
            item.guidance,
            item.rows,
            log=self.log,
            frame=frame,
            skeleton=item.skeleton,
            shown=item.shown,
        )

    async def _afford(self, wave: _Wave) -> None:
        """Wait for the calls in flight if the spend plus their reservations reaches the ceiling."""
        reserved = sum(wave.reserved.values())
        if reserved and self.log.usd + reserved >= self._ceiling:
            logger.info(
                "ceiling wait call=%d reserved_usd=%.4f in_flight=%d usd=%.4f ceiling=%.2f",
                self.log.calls + 1,
                reserved,
                len(wave.reserved),
                self.log.usd,
                self._ceiling,
            )
            await asyncio.wait([wave.flying[index] for index in wave.reserved])

    async def _fly(self, pending: Pending, wave: _Wave) -> Batch:
        token = CALL_NUMBER.set(pending.index)
        try:
            try:
                started = time.monotonic()
                reply = await self._reply(pending)
                batch = answer(
                    pending,
                    reply,
                    log=self.log,
                    elapsed_s=time.monotonic() - started,
                    in_flight=len(wave.reserved),
                )
            except BaseException as exc:
                fail(pending, wave.cancelled(exc), log=self.log)
                wave.answered.add(pending.index)
                # No later call is sent; the calls in flight land under their own idle limits.
                wave.failed.set()
                if isinstance(exc, IdleError):
                    wave.idle[pending.index] = exc
                raise
            wave.answered.add(pending.index)
            return batch
        finally:
            wave.reserved.pop(pending.index, None)
            wave.slots.release()
            CALL_NUMBER.reset(token)

    async def _reply(self, pending: Pending) -> Reply:
        """`pending`'s reply, awaited under the idle limit when the send returns an awaitable.

        Raises:
            IdleError: If no reply came within `limits.idle` seconds.
        """
        sent = self._send(pending.call)
        if not inspect.isawaitable(sent):
            return sent
        idle = self.limits.idle
        scope = asyncio.timeout(idle)
        # The send counts its attempts here, so a call the idle rule stops still reports them.
        holder = Tries()
        token = ATTEMPTS.set(holder)
        try:
            async with scope:
                return await sent
        except TimeoutError:
            if not scope.expired() or idle is None:
                raise
            msg = f"call {pending.index} got no reply within the idle limit of {idle:g} s"
            raise IdleError(
                msg, attempts=holder.count or None, rate_limited=holder.rate_limited
            ) from None
        finally:
            ATTEMPTS.reset(token)


def _reservation(pending: Pending) -> float:
    """The most `pending` can cost in USD, its input at the higher of `input` and `cache_write`."""
    settings = pending.call.settings
    price = price_of(settings.model)
    text = len(json.dumps(pending.call.params()))
    rate = max(price.input, price.cache_write)
    return (settings.max_tokens * price.output + (text + OVERHEAD_TOKENS) * rate) / 1e6


def _rule(error: BaseException | None) -> bool:
    """Whether `error` is a stop rule's: it carries a `stop_rule`, as the call log reads it."""
    return getattr(error, "stop_rule", None) is not None


def _outcome(task: asyncio.Task[Batch]) -> Batch | BaseException:
    """A landed task's batch or exception."""
    if task.cancelled():
        return asyncio.CancelledError()
    error = task.exception()
    return task.result() if error is None else error
