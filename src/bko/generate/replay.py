"""Replay: a `Send` that answers each call from a call log, so a paid run's reading costs nothing.

`Replay` reads a schema-2 call log and answers a rerun's calls in the order the log made them.
Each call must send the logged request's parameters. Its answer is the logged reply, rebuilt as a
`Reply`, or the logged failure, raised again as `GenerationError`. Order, not content, pairs them,
because one request can be logged many times with different replies. ADR
`research/call-log-replay` records the design.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

from bko.generate.errors import GenerationError, ReplayError
from bko.generate.log import SCHEMA
from bko.generate.messages import Call, Reply, Usage, text_of


class Replay:
    """A `Send` over a call log's calls, in their order.

    Attributes:
        digest: The log's sha256, which the rerun's own log records under `replay`.
    """

    def __init__(self, path: Path) -> None:
        """Read the call log at `path`.

        Raises:
            ReplayError: If a line is not a JSON object, the log is not schema `SCHEMA`, or a
                request has no reply or error line.
            OSError: If the log cannot be read.
        """
        data = path.read_bytes()
        self.digest = hashlib.sha256(data).hexdigest()
        lines = _lines(data, path)
        if not lines or lines[0].get("kind") != "run" or lines[0].get("schema") != SCHEMA:
            msg = f"{path} is not a schema {SCHEMA} call log"
            raise ReplayError(msg)
        requests = {line["call"]: line for line in lines if line.get("kind") == "request"}
        outcomes = {line["call"]: line for line in lines if line.get("kind") in {"reply", "error"}}
        unanswered = sorted(set(requests) - set(outcomes))
        if unanswered:
            msg = f"call {unanswered[0]} of {path} has no reply or error line"
            raise ReplayError(msg)
        self._calls = deque((requests[index], outcomes[index]) for index in sorted(requests))

    @property
    def remaining(self) -> int:
        """The logged calls not yet answered."""
        return len(self._calls)

    def __call__(self, call: Call) -> Reply:
        """Answer `call` with the next logged call's reply, or raise its logged failure.

        Raises:
            ReplayError: If no logged call is left, or `call` differs from the logged request.
            GenerationError: The logged call's failure.
        """
        if not self._calls:
            msg = "the call log holds no call left to answer this one"
            raise ReplayError(msg)
        request, outcome = self._calls.popleft()
        if json.loads(json.dumps(call.params(), sort_keys=True)) != request["params"]:
            msg = f"the call differs from logged call {request['call']}'s request"
            raise ReplayError(msg)
        if outcome["kind"] == "error":
            raise GenerationError(outcome["error"], attempts=outcome["attempts"])
        return Reply(
            request_id=outcome["request_id"],
            model=outcome["model"],
            stop_reason=outcome["stop_reason"],
            text=text_of(outcome["response"]),
            usage=Usage(**outcome["usage"]),
            raw=outcome["response"],
            attempts=outcome["attempts"],
        )


def _lines(data: bytes, path: Path) -> list[dict[str, Any]]:
    try:
        lines = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, ValueError) as exc:
        msg = f"{path} is not JSON lines: {exc}"
        raise ReplayError(msg) from exc
    if not all(isinstance(line, dict) for line in lines):
        msg = f"{path} holds a line that is not a JSON object"
        raise ReplayError(msg)
    return lines
