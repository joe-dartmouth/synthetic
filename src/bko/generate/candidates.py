"""The candidates file: one JSON line per asked record, written by `python -m bko.generate`.

Each line holds `call` (the call's 1-based index), `cell` (`bko.gaps.observe.cell_key`),
`position` (0-based within the call) and either the decoded `candidate`, as the reply gave it, or
the `undecoded` reason, its keys sorted. `bko.fusion` reads it back.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from bko.gaps.observe import Cell, cell_key
from bko.generate.decode import Reason, Undecoded


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
    """The lines of a candidates file, in file order."""
    raws = data.decode("utf-8").split("\n")[:-1]
    return tuple(_line(raw) for raw in raws)


def _line(raw: str) -> CandidateLine:
    value = json.loads(raw)
    if "undecoded" in value:
        candidate: object = Undecoded(Reason(value["undecoded"]))
    else:
        candidate = value["candidate"]
    return CandidateLine(value["call"], value["cell"], value["position"], candidate)
