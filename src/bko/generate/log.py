"""The call log: an append-only JSONL record of a run's every request and response, with its spend.

A run writes one `run` line, then per call a `request` line before the call and a `reply` or `error`
line after it, then one `end` line holding the run's status and totals. The `run` and `end` lines
name the log a replay answers from under `replay` and the stopped log a resume continues under
`resume`, and say under `rehearsal` whether canned replies answered every call; the `run` line
records under `in_flight` the most calls the run keeps in flight at once, and a resume's under
`carried_usd` the spend of the stopped log's calls it dropped, and the `end` line names under `stop`
the stop rule that ended a failed run, or null, under `retryable` whether the API caused the stop
(an idle stop, or a `GenerationError` marked `retryable`), so a supervised run resumes it, and under
`rate_limited` whether any call's retry met a 429; its totals split `live_usd`, what the run paid
or a rehearsal billed, from `usd`. Each line is flushed as it is written, so the lines of every call
made survive a failure, an interrupt or a killed process. The file is created exclusively, so a
rerun never overwrites a paid log, with mode 0o600, so only its owner reads it. ADR
`research/record-generation-design` records the format.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Self, TextIO

from bko.gaps.observe import Cell, cell_key
from bko.generate.decode import undecoded_in
from bko.generate.errors import GenerationError
from bko.generate.messages import Call, Reply, Settings
from bko.generate.spend import price_of


logger = logging.getLogger(__name__)

SCHEMA: Final = 2
"""The log format's version, written in the `run` line."""

STOP_RULES: Final = ("ceiling", "projection", "idle")
"""The stop rules a failed run's `end` line can name under `stop`."""


def utc_now() -> datetime:
    """The current time in UTC, the call log's default clock."""
    return datetime.now(UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class Answered:
    """The call log a run's first calls are answered from.

    Attributes:
        kind: `replay`, which answers every call, or `resume`, which answers the first `calls`
            and leaves the rest to the API.
        sha256: That log's sha256.
        calls: The calls it answers.
        carried_usd: For a resume, the spend of the logged calls it dropped and sends again, so
            the run's spend still counts them; 0 for a replay.
    """

    kind: Literal["replay", "resume"]
    sha256: str
    calls: int
    carried_usd: float = 0.0


class CallLog:
    """One run's call log, a context manager that writes the `end` line on exit.

    Construct it in a `with` statement: only `__exit__` closes the file once the `run` line is
    written.

    Attributes:
        path: The log file.
        answered: The call log the first calls are answered from, or None for a live run.
        replay: The sha256 of the call log a replay answers from, or None.
        resume: The sha256 and answered calls of the stopped log a resume continues, or None.
        rehearsal: Whether canned replies answer every call (`bko.evaluate.rehearse`).
        settings: How each call is made; its model's price is the one every spend uses.
        run: The named run whose calls the log holds, or None outside a named run.
        in_flight: The most calls the run keeps in flight at once (`bko.generate.flight`).
        calls: Calls requested.
        failed: Calls that wrote an `error` line rather than a `reply` line.
        retried: Calls that took more than one HTTP attempt.
        asked: Records requested, those of failed calls included.
        decoded: Candidates decoded from replies.
        undecoded: Placeholders for records a reply did not decode.
        input_tokens: Uncached input tokens billed.
        output_tokens: Output tokens billed.
        usd: Spend, priced at the settings model's rates; a resumed log's counts its replayed
            calls and starts at the spend it carried.
        live_usd: The spend of calls past the answered ones: what the run paid.
        rate_limited: Whether any call's retry met a 429, from its reply or its failure.
    """

    def __init__(
        self,
        path: Path,
        settings: Settings,
        *,
        answered: Answered | None = None,
        rehearsal: bool = False,
        run: str | None = None,
        in_flight: int = 1,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        """Price the settings model, then create the log and write its `run` line.

        Args:
            path: The log file; it must not exist. It is created readable by its owner only.
            settings: How each call is made.
            answered: The call log the first calls are answered from, or None for a live run.
            rehearsal: Whether canned replies answer every call.
            run: The named run whose calls the log holds, or None outside a named run.
            in_flight: The most calls the run keeps in flight at once; the `run` line records it.
            now: The clock each line's time is read from.

        Raises:
            GenerationError: As `bko.generate.spend.price_of`, before the file is created.
            importlib.metadata.PackageNotFoundError: If `anthropic` is not installed, before the
                file is created.
            FileExistsError: If `path` exists.
            OSError: If the file cannot be opened or its `run` line written; the file is then
                removed.
        """
        price = price_of(settings.model)
        # Read before the file exists, so a failure here leaves no log to block a rerun.
        anthropic = importlib.metadata.version("anthropic")
        self.path = path
        self.answered = answered
        self.replay = (
            answered.sha256 if answered is not None and answered.kind == "replay" else None
        )
        self.resume = (
            {"sha256": answered.sha256, "calls": answered.calls}
            if answered is not None and answered.kind == "resume"
            else None
        )
        self.rehearsal = rehearsal
        self.settings = settings
        self.run = run
        self.in_flight = in_flight
        self._now = now
        self.calls = 0
        self.failed = 0
        self.retried = 0
        self.asked = 0
        self.decoded = 0
        self.undecoded = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usd = 0.0 if answered is None else answered.carried_usd
        self.live_usd = 0.0
        self.rate_limited = False
        run_line: dict[str, object] = {
            "kind": "run",
            "schema": SCHEMA,
            "started": self._at(),
            "settings": dataclasses.asdict(settings),
            "price": dataclasses.asdict(price),
            "anthropic": anthropic,
            "replay": self.replay,
            "resume": self.resume,
            "rehearsal": rehearsal,
            "in_flight": in_flight,
        }
        if self.resume is not None:
            run_line["carried_usd"] = self.usd
        # Exclusive, and owner-only whatever the umask: the log holds every prompt and response.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        opened: TextIO | None = None
        try:
            opened = os.fdopen(descriptor, "w", encoding="utf-8")
            self._file: TextIO = opened
            self._write(run_line)
        except BaseException:
            # No log of a run that never started, so nothing blocks a rerun.
            if opened is None:
                os.close(descriptor)
            else:
                opened.close()
            path.unlink(missing_ok=True)
            raise
        logger.info(
            "log path=%s schema=%d model=%s effort=%s level=%s per_call=%d max_tokens=%d "
            "replay=%s resume=%s rehearsal=%s in_flight=%d",
            path,
            SCHEMA,
            settings.model,
            settings.effort,
            settings.guidance,
            settings.per_call,
            settings.max_tokens,
            self.replay or "none",
            "none" if self.resume is None else self.resume["sha256"],
            str(rehearsal).lower(),
            in_flight,
        )

    def __enter__(self) -> Self:
        """Return the log."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Write the `end` line, log the run's totals and close the file; never suppress.

        The totals line names the replayed log's sha256 under `replay` and the resumed log's under
        `resume`, or `none`, and splits `live_usd` from `usd`, so replayed spend never reads as
        paid.

        The status is `complete` with no exception, `interrupted` for `KeyboardInterrupt`, and
        `failed` for any other. `stop` names the stop rule that ended a failed run, read from the
        exception's `stop_rule` attribute when it is one of `STOP_RULES` (`bko.evaluate` raises
        it), and is null otherwise. `retryable` is true for a failed run stopped by the idle rule
        or by a `GenerationError` marked `retryable`, and false otherwise, other stop rules
        included. `rate_limited` is true when any call's retry met a 429.
        """
        if kind is None:
            status = "complete"
        elif issubclass(kind, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        rule = getattr(error, "stop_rule", None) if status == "failed" else None
        stop = rule if rule in STOP_RULES else None
        retryable = status == "failed" and (
            stop == "idle"
            or (stop is None and isinstance(error, GenerationError) and error.retryable)
        )
        try:
            self._write(
                {
                    "kind": "end",
                    "status": status,
                    "ended": self._at(),
                    "replay": self.replay,
                    "resume": self.resume,
                    "rehearsal": self.rehearsal,
                    "stop": stop,
                    "retryable": retryable,
                    "rate_limited": self.rate_limited,
                    **self._totals(),
                }
            )
            logger.info(
                "run status=%s run=%s level=%s in_flight=%d calls=%d failed=%d retried=%d asked=%d "
                "decoded=%d undecoded=%d input_tokens=%d output_tokens=%d usd=%.6f live_usd=%.6f "
                "replay=%s resume=%s rehearsal=%s log=%s",
                status,
                self.run or "none",
                self.settings.guidance,
                self.in_flight,
                self.calls,
                self.failed,
                self.retried,
                self.asked,
                self.decoded,
                self.undecoded,
                self.input_tokens,
                self.output_tokens,
                self.usd,
                self.live_usd,
                self.replay or "none",
                "none" if self.resume is None else self.resume["sha256"],
                str(self.rehearsal).lower(),
                self.path,
            )
        finally:
            self._file.close()

    def discard(self) -> None:
        """Close and remove the log, for a run that failed before its first call."""
        self._file.close()
        self.path.unlink(missing_ok=True)

    def request(self, cell: Cell, asked: int, call: Call) -> int:
        """Write a call's `request` line, before it is sent.

        Args:
            cell: The cell the call asks for.
            asked: The records it asks for.
            call: The request.

        Returns:
            The call's index in the run, counted from 1.
        """
        self.calls += 1
        self.asked += asked
        self._write(
            {
                "kind": "request",
                "call": self.calls,
                "cell": cell_key(cell),
                "asked": asked,
                "at": self._at(),
                "params": call.params(),
            }
        )
        return self.calls

    def reply(self, index: int, reply: Reply, candidates: Sequence[object], usd: float) -> None:
        """Write a call's `reply` line: its counts, usage and spend, and the whole response.

        Args:
            index: The call's index, as `request` returned it.
            reply: Its reply.
            candidates: What the reply decoded to, placeholders included.
            usd: Its spend.
        """
        undecoded = undecoded_in(candidates)
        decoded = len(candidates) - undecoded
        self.decoded += decoded
        self.undecoded += undecoded
        self.input_tokens += reply.usage.input_tokens
        self.output_tokens += reply.usage.output_tokens
        self.usd += usd
        if self.answered is None or index > self.answered.calls:
            self.live_usd += usd
        self.retried += (reply.attempts or 0) > 1
        self.rate_limited |= reply.rate_limited
        self._write(
            {
                "kind": "reply",
                "call": index,
                "at": self._at(),
                "request_id": reply.request_id,
                "attempts": reply.attempts,
                "model": reply.model,
                "stop_reason": reply.stop_reason,
                "decoded": decoded,
                "undecoded": undecoded,
                "usage": dataclasses.asdict(reply.usage),
                "usd": usd,
                "response": dict(reply.raw),
            }
        )

    def failure(self, index: int, error: BaseException) -> str:
        """Write a call's `error` line, for a call whose `reply` line was not written.

        A `GenerationError` is logged by its text, which `bko.generate.api` builds without the
        response body, so it holds no credential. Any other exception is logged by its type name
        only, since its text may hold one.

        Args:
            index: The call's index, as `request` returned it.
            error: The failure: no response, an interrupt, or a failure reading the reply.

        Returns:
            The text the line holds.
        """
        self.failed += 1
        attempts = error.attempts if isinstance(error, GenerationError) else None
        self.retried += (attempts or 0) > 1
        self.rate_limited |= isinstance(error, GenerationError) and error.rate_limited
        text = str(error) if isinstance(error, GenerationError) else type(error).__name__
        self._write(
            {"kind": "error", "call": index, "at": self._at(), "error": text, "attempts": attempts}
        )
        return text

    def _totals(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "failed": self.failed,
            "retried": self.retried,
            "asked": self.asked,
            "decoded": self.decoded,
            "undecoded": self.undecoded,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": self.usd,
            "live_usd": self.live_usd,
        }

    def _at(self) -> str:
        return self._now().isoformat()

    def _write(self, line: dict[str, object]) -> None:
        self._file.write(json.dumps(line, sort_keys=True) + "\n")
        self._file.flush()
