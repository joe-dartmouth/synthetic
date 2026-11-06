"""A Messages record's request read from its run folder's call log, for `bko.evaluate.trace`.

`read_prompts` reads the user turn of each wanted call and the run line's guidance level from
`calls.jsonl`; `asked` names one record's request from them, or why it cannot (ADR
`research/record-explainability`).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from bko.evaluate.errors import EvaluationError
from bko.evaluate.holdout_candidates import Candidate
from bko.generate.prompt import sections


CALLS: Final = "calls.jsonl"
"""The run folder's call log, which a Messages arm's prompts are read from."""

UNTRACED: Final = f"no {CALLS}"
"""A Messages record's `untraced` reason in a folder without a call log."""

NO_RUN: Final = f"{CALLS} has no run line, so no guidance level"
"""A Messages record's `untraced` reason when the call log names no guidance level."""

NO_CALL: Final = "its candidates line names no call"
"""A Messages record's `untraced` reason when its candidates line has a null call."""

_MALFORMED: Final = (ValueError, LookupError, TypeError, AttributeError, RecursionError)
"""What reading a call log line that is not the shape `bko.generate.log` writes raises."""

type Json = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Prompts:
    """The call log's user turns of the wanted calls, and its guidance level.

    Attributes:
        users: Each wanted call the log asks, with its user turn.
        level: The run line's guidance level; None when the log has no run line.
    """

    users: Mapping[int, str]
    level: str | None


def read_prompts(path: Path, wanted: Collection[int]) -> Prompts:
    """The user turn of each call in `wanted` that the call log at `path` asks, and its level.

    Raises:
        OSError: If the log cannot be read.
        EvaluationError: If it is not UTF-8, or a line is not JSON or a request line not the
            shape `bko.generate.log` writes, naming the path and the line.
    """
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"{path} is not UTF-8: {exc}"
        raise EvaluationError(msg) from exc
    users: dict[int, str] = {}
    level = None
    for number, raw in enumerate(text.splitlines(), start=1):
        try:
            ran, found = _take(json.loads(raw), wanted, users)
        except _MALFORMED as exc:
            msg = f"{path} line {number} is not a call log line: {type(exc).__name__}: {exc}"
            raise EvaluationError(msg) from exc
        if ran:
            level = found
    return Prompts(users, level)


def _take(line: Any, wanted: Collection[int], users: dict[int, str]) -> tuple[bool, str | None]:
    """Put a wanted request line's user turn in `users`; a run line gives `(True, its level)`."""
    if line.get("kind") == "run":
        return True, line.get("settings", {}).get("guidance")
    if line.get("kind") == "request" and line.get("call") in wanted:
        users[int(line["call"])] = _user(line["params"]["messages"][0]["content"])
    return False, None


def _user(content: str | Sequence[Json]) -> str:
    """A message's text: the string itself, or its text blocks joined."""
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text", "")) for block in content)


def asked(line: Candidate, prompts: Prompts | None, known: Json) -> tuple[Json | None, str | None]:
    """A Messages record's request, `known` beside its cell and prompt sections, or why not.

    `prompts` is None for a folder without a call log.
    """
    if prompts is None:
        return None, UNTRACED
    if prompts.level is None:
        return None, NO_RUN
    if line.call is None:
        return None, NO_CALL
    user = prompts.users.get(line.call)
    if user is None:
        return None, f"call {line.call} is not in {CALLS}"
    try:
        read = sections(user)
    except ValueError as exc:
        return None, f"call {line.call}'s prompt: {exc}"
    shown = {part.name: getattr(read, part.name) for part in dataclasses.fields(read)}
    return {**known, "guidance": prompts.level, "cell": line.cell, "sections": shown}, None
