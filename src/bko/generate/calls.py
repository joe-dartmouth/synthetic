"""Generation calls: `ask` makes one logged call; `generate` fills a quota in a fixed call count.

`ask` renders the prompt at the settings' guidance level, sends it, decodes the reply into
candidates and placeholders, prices it and logs it. It is built from `request`, which numbers and
logs the call, `answer`, which reads and logs its reply, and `fail`, which logs a call with no
reply, so `bko.generate.flight` can keep several calls in flight between them. `generate` checks
the whole quota, then asks each cell's rows in chunks of `Settings.per_call`, cells in quota
order, so a run makes
`sum(ceil(rows / per_call))` calls. Neither retries or tops up; a call with no response raises
`GenerationError` after its `error` line is written, and so does anything else raised before its
`reply` line. ADR `research/record-generation-design` records the design.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.gaps.observe import Cell, cell_key
from bko.generate.decode import Undecoded, decode, undecoded_in
from bko.generate.guidance import Guidance
from bko.generate.log import CallLog
from bko.generate.messages import CALL_NUMBER, Call, Reply, Send, Settings
from bko.generate.prompt import render
from bko.generate.spend import cost, price_of


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Batch:
    """One call's outcome.

    Attributes:
        index: The call's index in the run, counted from 1.
        cell: The cell it asked for.
        asked: The records it asked for.
        candidates: The decoded items, then one `Undecoded` per asked record short.
        reply: The reply.
        usd: Its spend.
    """

    index: int
    cell: Cell
    asked: int
    candidates: tuple[object, ...]
    reply: Reply
    usd: float

    @property
    def undecoded(self) -> int:
        """The placeholders among the candidates."""
        return undecoded_in(self.candidates)


@dataclass(frozen=True, slots=True)
class Generation:
    """A run's calls, in the order they were made.

    Attributes:
        batches: Each call's outcome.
    """

    batches: tuple[Batch, ...]

    @property
    def usd(self) -> float:
        """The run's spend: every call's, summed."""
        return sum((batch.usd for batch in self.batches), 0.0)

    def candidates(self, cell: Cell) -> tuple[object, ...]:
        """Return every candidate of `cell`'s calls, in call order."""
        return tuple(
            candidate
            for batch in self.batches
            if batch.cell == cell
            for candidate in batch.candidates
        )


def _settings(log: CallLog, settings: Settings | None) -> Settings:
    """The log's settings, refusing others: the log priced and recorded its own.

    Raises:
        ValueError: If `settings` is given and differs from `log.settings`.
    """
    if settings is None or settings == log.settings:
        return log.settings
    msg = f"settings {settings!r} differ from the settings the call log records, {log.settings!r}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Pending:
    """A call whose `request` line is written and whose `reply` or `error` line is not.

    Attributes:
        index: The call's index in the run, counted from 1.
        cell: The cell it asks for.
        asked: The records it asks for.
        call: The request to send.
    """

    index: int
    cell: Cell
    asked: int
    call: Call


def request(
    guidance: Guidance,
    rows: int,
    *,
    log: CallLog,
    settings: Settings | None = None,
    frame: FrameSpec = UCI_FRAME,
    skeleton: object | None = None,
) -> Pending:
    """Render one call for `rows` records of `guidance`'s cell and write its `request` line.

    Args:
        guidance: What the request says about the cell.
        rows: The records the call asks for.
        log: The run's call log, which numbers the call.
        settings: How the call is made, its guidance level included; the log's when None.
        frame: The frame whose records the call asks for.
        skeleton: Reserved for a record skeleton; it must be None.

    Returns:
        The call, numbered and logged, to send.

    Raises:
        ValueError: Before the `request` line, if `settings` differ from the log's or a skeleton
            is given.
    """
    if skeleton is not None:
        msg = f"a skeleton is not taken yet, not {skeleton!r}"
        raise ValueError(msg)
    settings = _settings(log, settings)
    system, user = render(guidance, rows, settings.guidance, frame=frame)
    call = Call(settings, guidance.sex, guidance.label, system, user, frame=frame)
    index = log.request(guidance.cell, rows, call)
    CALL_NUMBER.set(index)
    # The SDK logs its own retries at INFO; this line lets a reader tie them to the call.
    logger.info("send call=%d cell=%s asked=%d", index, cell_key(guidance.cell), rows)
    return Pending(index, guidance.cell, rows, call)


def answer(pending: Pending, reply: Reply, *, log: CallLog) -> Batch:
    """Decode and price `pending`'s reply, and write its `reply` line.

    Args:
        pending: The call, as `request` returned it.
        reply: Its reply.
        log: The run's call log.

    Returns:
        The call's outcome.
    """
    settings = log.settings
    index, key, rows = pending.index, cell_key(pending.cell), pending.asked
    candidates = decode(reply, rows)
    usd = cost(reply.usage, settings.model)
    log.reply(index, reply, candidates, usd)
    batch = Batch(index, pending.cell, rows, candidates, reply, usd)
    if reply.model != settings.model:
        logger.warning("served call=%d model=%s settings=%s", index, reply.model, settings.model)
    logger.info(
        "generate call=%d cell=%s level=%s asked=%d decoded=%d undecoded=%d stop_reason=%s "
        "input_tokens=%d output_tokens=%d usd=%.6f request_id=%s",
        index,
        key,
        settings.guidance,
        rows,
        len(candidates) - batch.undecoded,
        batch.undecoded,
        reply.stop_reason,
        reply.usage.input_tokens,
        reply.usage.output_tokens,
        usd,
        reply.request_id,
    )
    if batch.undecoded:
        reason = next(c.reason for c in candidates if isinstance(c, Undecoded))
        logger.warning(
            "short call=%d cell=%s asked=%d undecoded=%d reason=%s",
            index,
            key,
            rows,
            batch.undecoded,
            reason,
        )
    return batch


def fail(pending: Pending, error: BaseException, *, log: CallLog) -> str:
    """Write `pending`'s `error` line and an ERROR log line, for a call with no `reply` line.

    Args:
        pending: The call, as `request` returned it.
        error: Why it has no reply: no response, an interrupt, or a failure reading the reply.
        log: The run's call log.

    Returns:
        The text the `error` line holds.
    """
    text = log.failure(pending.index, error)
    # No traceback: an exception's text may hold the credential, and the caller re-raises.
    key = cell_key(pending.cell)
    logger.error("failed call=%d cell=%s error=%s", pending.index, key, text)
    return text


def ask(
    guidance: Guidance,
    rows: int,
    *,
    send: Send,
    log: CallLog,
    settings: Settings | None = None,
    frame: FrameSpec = UCI_FRAME,
) -> Batch:
    """Make one call for `rows` records of `guidance`'s cell, and log it.

    Args:
        guidance: What the request says about the cell.
        rows: The records the call asks for.
        send: Makes the call.
        log: The run's call log.
        settings: How the call is made, its guidance level included; the log's when None.
        frame: The frame whose records the call asks for.

    Returns:
        The call's outcome.

    Raises:
        ValueError: Before the `request` line, if `settings` differ from the log's.
        GenerationError: If no response arrives. Its `error` line and an ERROR log line are
            written first, as they are for anything else raised between the `request` line
            and the `reply` line, a `KeyboardInterrupt` included, which propagates unchanged.
    """
    pending = request(guidance, rows, log=log, settings=settings, frame=frame)
    try:
        return answer(pending, send(pending.call), log=log)
    except BaseException as exc:
        fail(pending, exc, log=log)
        raise


def generate(
    quota: Mapping[Cell, int],
    guidance: Mapping[Cell, Guidance],
    *,
    send: Send,
    log: CallLog,
    settings: Settings | None = None,
    on_batch: Callable[[Batch], None] | None = None,
    frame: FrameSpec = UCI_FRAME,
) -> Generation:
    """Ask for each cell's quota in chunks of `settings.per_call`, cells in quota order.

    Args:
        quota: The records to generate per cell.
        guidance: Each cell's guidance, as `bko.generate.guidance.guide` returns it.
        send: Makes each call.
        log: The run's call log.
        settings: How each call is made; the log's when None.
        on_batch: Called with each call's outcome as it returns, before the next call, so a
            caller can write it out before a later call fails.
        frame: The frame whose records each call asks for.

    Returns:
        Every call's outcome, in call order.

    Raises:
        ValueError: Before any call, if `settings` differ from the log's, or a quota entry is
            not an `int` of at least 1 (a `bool` is refused) or names a cell `guidance` does
            not hold.
        GenerationError: As `ask`.
    """
    settings = _settings(log, settings)
    for cell, rows in quota.items():
        if cell not in guidance:
            msg = f"quota cell {cell!r} has no guidance"
            raise ValueError(msg)
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            msg = f"quota of cell {cell!r} must be an int of at least 1, not {rows!r}"
            raise ValueError(msg)
    planned = chunks(quota, settings.per_call)
    calls = len(planned)
    ceiling = calls * settings.max_tokens * price_of(settings.model).output / 1e6
    logger.info(
        "plan calls=%d level=%s max_tokens=%d max_output_usd=%.2f",
        calls,
        settings.guidance,
        settings.max_tokens,
        ceiling,
    )
    batches: list[Batch] = []
    for cell, chunk in planned:
        batch = ask(guidance[cell], chunk, send=send, log=log, settings=settings, frame=frame)
        batches.append(batch)
        if on_batch is not None:
            on_batch(batch)
    return Generation(tuple(batches))


def chunks(quota: Mapping[Cell, int], per_call: int) -> tuple[tuple[Cell, int], ...]:
    """Each call `generate` makes for `quota`: its cell and records asked, in call order."""
    return tuple(
        (cell, min(per_call, rows - start))
        for cell, rows in quota.items()
        for start in range(0, rows, per_call)
    )
