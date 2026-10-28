"""Replay: a `Send` that answers each call from a call log, so a paid run's reading costs nothing.

`Replay` reads a schema-2 call log and answers a rerun's calls in the order the log made them.
Each call must send the logged request's parameters. Its answer is the logged reply, rebuilt as a
`Reply`, or the logged failure, raised again as `GenerationError`. Order, not content, pairs them,
because one request can be logged many times with different replies.

`Resume` reads a stopped run's log the same way, then sends the calls past its answered ones live.
It drops a final segment with no newline, which a killed process can leave, and a last call whose
answer is not exactly one `reply` line. A log whose `run` line holds `in_flight` above 1 had calls
in flight together, so it drops every call from the first one not answered by exactly one `reply`
line, and carries the dropped calls' spend into the resumed run's (`Answered.carried_usd`). At
any in-flight, what the stopped log itself carried is carried too, so a resume of a resume keeps
the whole run's spend. It refuses a log with a failed call before its last, a replay's log, a
rehearsal's log, whose replies are canned, a run the projection or the ceiling stopped, which is
reported rather than resumed, and a resumed log that stopped before its replayed calls ran out.
`refuse_unresumable` holds the checks on the `run` and `end` lines, and `log_ends` reads those two
lines alone, so `launch --resume` refuses before it claims a folder. ADR `research/call-log-replay`
records the design.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import deque
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NamedTuple

from bko.generate.errors import GenerationError, ReplayError
from bko.generate.log import SCHEMA, Answered
from bko.generate.messages import AnySend, Call, Reply, Usage, text_of


logger = logging.getLogger(__name__)


class _Log(NamedTuple):
    """A call log as read: sha256, `run` line, calls, `end` line, carried spend, dropped calls."""

    digest: str
    run: dict[str, Any]
    calls: tuple[tuple[dict[str, Any], dict[str, Any]], ...]
    end: dict[str, Any] | None
    carried_usd: float = 0.0
    dropped: int = 0


UNRESUMABLE: Final = frozenset({"projection", "ceiling"})
"""The stop rules whose runs are reported, never resumed: resuming would spend past the rule."""

TAIL: Final = 1 << 16
"""Bytes read from a call log's end to find its `end` line."""


def refuse_unresumable(path: Path, run: Mapping[str, Any], end: Mapping[str, Any] | None) -> None:
    """Refuse a stopped log whose `run` or `end` line says it must not be resumed.

    Raises:
        ReplayError: If the log replays another, is a rehearsal's, or ended on a stop rule in
            `UNRESUMABLE`.
    """
    source = run.get("replay")
    if source is not None:
        msg = f"{path} replays the log with sha256 {source}; replay that log instead"
        raise ReplayError(msg)
    if run.get("rehearsal"):
        msg = f"{path} is a rehearsal's log, whose replies are canned; a resume sends live"
        raise ReplayError(msg)
    stop = end.get("stop") if end is not None else None
    if stop in UNRESUMABLE:
        msg = (
            f"{path} was stopped by the {stop} rule, so it is reported, not resumed; "
            "a rerun needs a new run id"
        )
        raise ReplayError(msg)


def log_ends(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The `run` line of the call log at `path` and its `end` line, or None when it has none.

    Only the first line and the last `TAIL` bytes are read, so a paid log of hundreds of MB costs
    nothing to check.

    Raises:
        ReplayError: If the first line is not a schema `SCHEMA` `run` line.
        OSError: If the log cannot be read.
    """
    with path.open("rb") as handle:
        first = handle.readline()
        size = handle.seek(0, 2)
        handle.seek(max(0, size - TAIL))
        tail = [line for line in handle.read().splitlines() if line.strip()]
    try:
        run = json.loads(first)
    except ValueError:
        run = None
    if not isinstance(run, dict) or run.get("kind") != "run" or run.get("schema") != SCHEMA:
        msg = f"{path} is not a schema {SCHEMA} call log"
        raise ReplayError(msg)
    try:
        last = json.loads(tail[-1]) if tail else None
    except ValueError:
        last = None
    end = last if isinstance(last, dict) and last.get("kind") == "end" else None
    return run, end


class Replay:
    """A `Send` over a call log's calls, in their order.

    Attributes:
        digest: The log's sha256, which the rerun's own log records under `replay`.
        rehearsal: Whether the log is a rehearsal's, so the rerun's own log keeps the mark.
    """

    def __init__(self, path: Path) -> None:
        """Read the call log at `path`.

        Raises:
            ReplayError: If a line is not a JSON object; the first line is not a schema
                `SCHEMA` `run` line; a request, reply or error line has no whole-number `call`,
                or shares one with another line of its kind; the requests are not calls 1 to n;
                a request has no reply or error line, or an answer no request; or a line misses
                a field its kind needs.
            OSError: If the log cannot be read.
        """
        self._load(_read(path, stopped=False))

    def _load(self, log: _Log) -> None:
        self.digest = log.digest
        self.rehearsal = bool(log.run.get("rehearsal"))
        self._calls = deque(log.calls)
        self._total = len(log.calls)

    @property
    def answered(self) -> Answered:
        """What the rerun's own log records about this source."""
        return Answered("replay", self.digest, self._total)

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


class Resume(Replay):
    """A `Send` that answers a stopped run's answered calls from its log, then sends the rest live.

    Attributes:
        digest: The stopped log's sha256, which the resumed run's log records under `resume`.
        dropped: The logged calls it dropped, to send again.
    """

    def __init__(self, path: Path, send: AnySend) -> None:
        """Read the stopped log at `path`; calls past its answered ones go through `send`.

        Raises:
            ReplayError: As `Replay`, past a dropped last call; or if a call before the last
                failed, as `refuse_unresumable`, if its `resume` field is malformed, or if it
                resumed more calls than it holds.
            OSError: If the log cannot be read.
        """
        log = _read(path, stopped=True)
        refuse_unresumable(path, log.run, log.end)
        prior = log.run.get("resume")
        if prior is not None:
            calls = prior.get("calls") if isinstance(prior, dict) else None
            sha256 = prior.get("sha256") if isinstance(prior, dict) else None
            if isinstance(calls, bool) or not isinstance(calls, int) or not isinstance(sha256, str):
                msg = f"the run line of {path} has resume {prior!r}, not a sha256 and a call count"
                raise ReplayError(msg)
            if len(log.calls) < calls:
                msg = f"{path} stopped while it replayed; resume the log with sha256 {sha256!r}"
                raise ReplayError(msg)
        self._load(log)
        self._carried = log.carried_usd
        self.dropped = log.dropped
        self._live = send
        self._sent = False

    @property
    def answered(self) -> Answered:
        """What the resumed run's own log records about this source, the spend it carries too."""
        return Answered("resume", self.digest, self._total, self._carried)

    def __call__(self, call: Call) -> Reply | Awaitable[Reply]:  # ty: ignore[invalid-method-override]
        """Answer from the stopped log while it holds calls, then send live, awaitably or not."""
        if self._calls:
            return super().__call__(call)
        if not self._sent:
            self._sent = True
            logger.info("resume live from call %d", self._total + 1)
        return self._live(call)


def _read(path: Path, *, stopped: bool) -> _Log:
    """Read and check the call log at `path`; `stopped` reads a stopped run's log.

    Raises:
        ReplayError: As `Replay`, and for a stopped log, as `Resume`.
        OSError: If the log cannot be read.
    """
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if stopped and not data.endswith(b"\n"):
        # A killed process can leave its last line unfinished; that line answered no call.
        kept = data.rfind(b"\n") + 1
        logger.warning(
            "resume dropped a cut final line of %d bytes from %s", len(data) - kept, path
        )
        data = data[:kept]
    lines = _lines(data, path)
    if not lines or lines[0].get("kind") != "run" or lines[0].get("schema") != SCHEMA:
        msg = f"{path} is not a schema {SCHEMA} call log"
        raise ReplayError(msg)
    carried, dropped = 0.0, 0
    if stopped:
        lines, carried, dropped = _resumable(lines, path)
    requests = _by_call(lines, {"request"}, path)
    outcomes = _by_call(lines, {"reply", "error"}, path)
    if sorted(requests) != list(range(1, len(requests) + 1)):
        msg = f"the request lines of {path} are not calls 1 to {len(requests)}"
        raise ReplayError(msg)
    unanswered = sorted(set(requests) - set(outcomes))
    if unanswered:
        msg = f"call {unanswered[0]} of {path} has no reply or error line"
        raise ReplayError(msg)
    unasked = sorted(set(outcomes) - set(requests))
    if unasked:
        msg = f"call {unasked[0]} of {path} has an answer and no request line"
        raise ReplayError(msg)
    for index in requests:
        _check(requests[index], outcomes[index], path)
    failed = sorted(index for index, outcome in outcomes.items() if outcome["kind"] == "error")
    if stopped and failed:
        msg = (
            f"call {failed[0]} of {path} failed before the run's last call, "
            "so the run did not stop there"
        )
        raise ReplayError(msg)
    calls = tuple((requests[index], outcomes[index]) for index in sorted(requests))
    end = lines[-1] if lines[-1].get("kind") == "end" else None
    return _Log(digest, lines[0], calls, end, carried, dropped)


_CALL_KINDS: Final = frozenset({"request", "reply", "error"})


def _resumable(lines: list[dict[str, Any]], path: Path) -> tuple[list[dict[str, Any]], float, int]:
    """A stopped log's `lines` without the calls a resume sends again, its spend and their count.

    The spend is what the log itself carried, at any in-flight, so a resume of a resume keeps the
    whole run's spend, plus the dropped calls' spend for a concurrent log.
    """
    prior = lines[0].get("carried_usd", 0.0)
    carried = (
        float(prior) if isinstance(prior, int | float) and not isinstance(prior, bool) else 0.0
    )
    if _concurrent(lines[0]):
        kept, spend, dropped = _from_first_unanswered(lines, path)
        return kept, carried + spend, dropped
    kept, dropped = _without_unanswered_last(lines, path)
    return kept, carried, dropped


def _concurrent(run: Mapping[str, Any]) -> bool:
    """Whether a `run` line says its run kept more than one call in flight."""
    in_flight = run.get("in_flight")
    return isinstance(in_flight, int) and not isinstance(in_flight, bool) and in_flight > 1


def _from_first_unanswered(
    lines: list[dict[str, Any]], path: Path
) -> tuple[list[dict[str, Any]], float, int]:
    """`lines` without every call from the first not answered by one reply, their spend and count.

    Each dropped call gets one warning. The spend is the dropped calls' `reply` lines' `usd`.
    """
    answers: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        index = line.get("call")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if line.get("kind") == "request":
            answers.setdefault(index, [])
        elif line.get("kind") in {"reply", "error"}:
            answers.setdefault(index, []).append(line)
    first = next(
        (
            index
            for index in sorted(answers)
            if [line["kind"] for line in answers[index]] != ["reply"]
        ),
        None,
    )
    carried = 0.0
    if first is None:
        return lines, carried, 0
    dropped = [index for index in sorted(answers) if index >= first]
    for index in dropped:
        kinds = [line["kind"] for line in answers[index]]
        logger.warning(
            "resume dropped call %d of %s, whose answer lines are %s; it is sent again",
            index,
            path,
            kinds or "none",
        )
        carried += sum(
            line["usd"]
            for line in answers[index]
            if line["kind"] == "reply" and isinstance(line.get("usd"), int | float)
        )
    gone = set(dropped)
    kept = [
        line
        for line in lines
        if line.get("kind") not in _CALL_KINDS or line.get("call") not in gone
    ]
    return kept, carried, len(dropped)


def _without_unanswered_last(
    lines: list[dict[str, Any]], path: Path
) -> tuple[list[dict[str, Any]], int]:
    """`lines` without the last call when its answer is not exactly one reply, and calls dropped."""
    last = max(
        (
            line["call"]
            for line in lines
            if line.get("kind") == "request"
            and isinstance(line.get("call"), int)
            and not isinstance(line.get("call"), bool)
        ),
        default=None,
    )
    if last is None:
        return lines, 0
    answers = [
        line["kind"]
        for line in lines
        if line.get("kind") in {"reply", "error"} and line.get("call") == last
    ]
    if answers == ["reply"]:
        return lines, 0
    logger.warning(
        "resume dropped call %d of %s, whose answer lines are %s; it is sent again",
        last,
        path,
        answers or "none",
    )
    kept = [
        line for line in lines if line.get("kind") not in _CALL_KINDS or line.get("call") != last
    ]
    return kept, 1


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


_OUTCOME: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "reply": frozenset({"request_id", "model", "stop_reason", "usage", "response", "attempts"}),
        "error": frozenset({"error", "attempts"}),
    }
)
"""The fields each kind of answer line needs to be raised or rebuilt."""

_USAGE: Final = frozenset(field.name for field in dataclasses.fields(Usage))


def _by_call(
    lines: Sequence[dict[str, Any]], kinds: set[str], path: Path
) -> dict[int, dict[str, Any]]:
    """The lines of `kinds` by their `call`, refusing a missing, non-integer or repeated one."""
    found: dict[int, dict[str, Any]] = {}
    for line in lines:
        if line.get("kind") not in kinds:
            continue
        index = line.get("call")
        if isinstance(index, bool) or not isinstance(index, int):
            msg = f"a {line['kind']} line of {path} has no whole-number call: {index!r}"
            raise ReplayError(msg)
        if index in found:
            msg = f"call {index} of {path} has more than one {'/'.join(sorted(kinds))} line"
            raise ReplayError(msg)
        found[index] = line
    return found


def _check(request: dict[str, Any], outcome: dict[str, Any], path: Path) -> None:
    """Refuse a request or answer line that misses a field `Replay.__call__` reads."""
    index = request["call"]
    if not isinstance(request.get("params"), dict):
        msg = f"call {index} of {path} has a request line with no params"
        raise ReplayError(msg)
    missing = sorted(_OUTCOME[outcome["kind"]] - set(outcome))
    if missing:
        msg = f"call {index} of {path} has a {outcome['kind']} line missing {missing}"
        raise ReplayError(msg)
    attempts = outcome["attempts"]
    if attempts is not None and (isinstance(attempts, bool) or not isinstance(attempts, int)):
        msg = f"call {index} of {path} has attempts {attempts!r}, not a whole number"
        raise ReplayError(msg)
    if outcome["kind"] == "error":
        if not isinstance(outcome["error"], str):
            msg = f"call {index} of {path} has an error line whose error is not text"
            raise ReplayError(msg)
        return
    usage = outcome["usage"]
    if not isinstance(usage, dict) or set(usage) != _USAGE:
        msg = f"call {index} of {path} has usage {usage!r}, not the keys {sorted(_USAGE)}"
        raise ReplayError(msg)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in usage.values()):
        msg = f"call {index} of {path} has a usage count that is not a whole number"
        raise ReplayError(msg)
    if not isinstance(outcome["response"], dict):
        msg = f"call {index} of {path} has a reply line whose response is not an object"
        raise ReplayError(msg)
