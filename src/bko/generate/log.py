"""The call log: an append-only JSONL record of a run's every request and response, with its spend.

A run writes one `run` line, then per call a `request` line before the call and a `reply` or
`error` line after it, then one `end` line holding the run's status and totals. Each line is
flushed as it is written, so the lines of every call made survive a failure, an interrupt or a
killed process. The file is created exclusively, so a rerun never overwrites a paid log, with mode
0o600, so only its owner reads it. ADR
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
from typing import Final, Self, TextIO

from bko.gaps.observe import Cell, cell_key
from bko.generate.decode import undecoded_in
from bko.generate.errors import GenerationError
from bko.generate.messages import Call, Reply, Settings
from bko.generate.spend import price_of


logger = logging.getLogger(__name__)

SCHEMA: Final = 1
"""The log format's version, written in the `run` line."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CallLog:
    """One run's call log, a context manager that writes the `end` line on exit.

    Construct it in a `with` statement: only `__exit__` closes the file once the `run` line is
    written.

    Attributes:
        path: The log file.
        settings: How each call is made; its model's price is the one every spend uses.
        calls: Calls requested.
        failed: Calls that wrote an `error` line rather than a `reply` line.
        asked: Records requested, those of failed calls included.
        decoded: Candidates decoded from replies.
        undecoded: Placeholders for records a reply did not decode.
        input_tokens: Uncached input tokens billed.
        output_tokens: Output tokens billed.
        usd: Spend, priced at the settings model's rates.
    """

    def __init__(
        self, path: Path, settings: Settings, *, now: Callable[[], datetime] = _utc_now
    ) -> None:
        """Price the settings model, then create the log and write its `run` line.

        Args:
            path: The log file; it must not exist. It is created readable by its owner only.
            settings: How each call is made.
            now: The clock each line's time is read from.

        Raises:
            GenerationError: As `bko.generate.spend.price_of`, before the file is created.
            FileExistsError: If `path` exists.
        """
        price = price_of(settings.model)
        self.path = path
        self.settings = settings
        self._now = now
        self.calls = 0
        self.failed = 0
        self.asked = 0
        self.decoded = 0
        self.undecoded = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usd = 0.0
        # Exclusive, and owner-only whatever the umask: the log holds every prompt and response.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file: TextIO = os.fdopen(descriptor, "w", encoding="utf-8")
        try:
            self._write(
                {
                    "kind": "run",
                    "schema": SCHEMA,
                    "started": self._at(),
                    "settings": dataclasses.asdict(settings),
                    "price": dataclasses.asdict(price),
                    "anthropic": importlib.metadata.version("anthropic"),
                }
            )
        except BaseException:
            self._file.close()
            raise

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

        The status is `complete` with no exception, `interrupted` for `KeyboardInterrupt`, and
        `failed` for any other.
        """
        if kind is None:
            status = "complete"
        elif issubclass(kind, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        try:
            self._write({"kind": "end", "status": status, "ended": self._at(), **self._totals()})
            logger.info(
                "run status=%s calls=%d failed=%d asked=%d decoded=%d undecoded=%d "
                "input_tokens=%d output_tokens=%d usd=%.4f log=%s",
                status,
                self.calls,
                self.failed,
                self.asked,
                self.decoded,
                self.undecoded,
                self.input_tokens,
                self.output_tokens,
                self.usd,
                self.path,
            )
        finally:
            self._file.close()

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
        self._write(
            {
                "kind": "reply",
                "call": index,
                "at": self._at(),
                "request_id": reply.request_id,
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
        text = str(error) if isinstance(error, GenerationError) else type(error).__name__
        self._write({"kind": "error", "call": index, "at": self._at(), "error": text})
        return text

    def _totals(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "failed": self.failed,
            "asked": self.asked,
            "decoded": self.decoded,
            "undecoded": self.undecoded,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": self.usd,
        }

    def _at(self) -> str:
        return self._now().isoformat()

    def _write(self, line: dict[str, object]) -> None:
        self._file.write(json.dumps(line, sort_keys=True) + "\n")
        self._file.flush()
