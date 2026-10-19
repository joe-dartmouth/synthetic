"""The candidates file: one JSON line per asked record, written by `python -m bko.generate`.

Each line holds `call` (the call's 1-based index), `cell` (`bko.gaps.observe.cell_key`),
`position` (0-based within the call) and either the decoded `candidate`, as the reply gave it, or
the `undecoded` reason, its keys sorted. `bko.fusion` reads it back.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from bko.gaps.observe import Cell, cell_key
from bko.generate.decode import Reason, Undecoded
from bko.generate.errors import CandidatesFormatError


KEYS: Final = frozenset({"call", "cell", "position"})
"""The keys every line holds."""
OUTCOMES: Final = frozenset({"candidate", "undecoded"})
"""The keys of which every line holds exactly one."""


@dataclass(frozen=True, slots=True)
class CandidateLine:
    """One asked record.

    Attributes:
        call: The call's 1-based index.
        cell: The cell's key, as `cell_key` writes it.
        position: The record's 0-based position in the call.
        candidate: The decoded item as the reply gave it, or an `Undecoded` placeholder.
    """

    call: int
    cell: str
    position: int
    candidate: object


def render_lines(index: int, cell: Cell, candidates: Sequence[object]) -> str:
    """One line per candidate of one call, each ending in a newline."""
    lines: list[str] = []
    for position, candidate in enumerate(candidates):
        line: dict[str, object] = {"call": index, "cell": cell_key(cell), "position": position}
        if isinstance(candidate, Undecoded):
            line["undecoded"] = candidate.reason.value
        else:
            line["candidate"] = candidate
        lines.append(json.dumps(line, sort_keys=True) + "\n")
    return "".join(lines)


def read(data: bytes) -> tuple[CandidateLine, ...]:
    """The lines of a candidates file, in file order.

    Raises:
        CandidatesFormatError: If the bytes are not UTF-8, the last line has no newline, a line is
            not one candidate line, a (call, position) repeats, or a call names two cells.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"the candidates file is not UTF-8: {exc}"
        raise CandidatesFormatError(msg) from exc
    if text and not text.endswith("\n"):
        msg = "the candidates file's last line has no newline, so its run stopped mid-line"
        raise CandidatesFormatError(msg)
    lines: list[CandidateLine] = []
    seen: set[tuple[int, int]] = set()
    cells: dict[int, str] = {}
    for number, raw in enumerate(text.split("\n")[:-1], start=1):
        line = _line(number, raw)
        if (line.call, line.position) in seen:
            msg = f"line {number} repeats call {line.call} position {line.position}"
            raise CandidatesFormatError(msg)
        if cells.setdefault(line.call, line.cell) != line.cell:
            msg = f"line {number} puts call {line.call} in {line.cell}, not {cells[line.call]}"
            raise CandidatesFormatError(msg)
        seen.add((line.call, line.position))
        lines.append(line)
    return tuple(lines)


def _line(number: int, raw: str) -> CandidateLine:
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        msg = f"line {number} is not JSON: {exc}"
        raise CandidatesFormatError(msg) from exc
    if (
        not isinstance(value, dict)
        or value.keys() - OUTCOMES != KEYS
        or len(OUTCOMES & value.keys()) != 1
    ):
        msg = f"line {number} is not call, cell, position and one of candidate or undecoded"
        raise CandidatesFormatError(msg)
    call, cell, position = value["call"], value["cell"], value["position"]
    if type(call) is not int or call < 1 or type(position) is not int or position < 0:
        msg = f"line {number}'s call and position are not a positive and a non-negative integer"
        raise CandidatesFormatError(msg)
    if not isinstance(cell, str):
        msg = f"line {number}'s cell is not text"
        raise CandidatesFormatError(msg)
    if "undecoded" not in value:
        return CandidateLine(call, cell, position, value["candidate"])
    try:
        reason = Reason(value["undecoded"])
    except ValueError, TypeError:
        msg = f"line {number}'s undecoded reason {value['undecoded']!r} is not a known reason"
        raise CandidatesFormatError(msg) from None
    return CandidateLine(call, cell, position, Undecoded(reason))
