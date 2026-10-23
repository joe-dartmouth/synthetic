"""The candidates file `python -m bko.generate` writes: a header line, then one line per record.

The header names the run the file answers: `kind` `header`, `schema`, the `request`, the gap
`definition`, the `guidance` level, the UCI archive's pinned sha256 (`uci`), the split's
`fingerprint`, each loaded release's pinned sha256 by prefix (`releases`), and the call plan
(`calls`: each call's cell key and records asked, in call order from call 1). Each later line, one
per asked record, holds `call` (the call's 1-based index), `cell` (`bko.gaps.observe.cell_key`),
`position` (0-based within the call) and either the decoded `candidate`, as the reply gave it, or
the `undecoded` reason, its keys sorted. `bko.fusion` reads it back.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, NoReturn

from bko.gaps.definitions import Definition
from bko.gaps.observe import Cell, cell_key
from bko.generate.decode import Reason, Undecoded
from bko.generate.errors import CandidatesFormatError
from bko.generate.messages import Level


KEYS: Final = frozenset({"call", "cell", "position"})
"""The keys every line holds."""
OUTCOMES: Final = frozenset({"candidate", "undecoded"})
"""The keys of which every line holds exactly one."""
HEADER_SCHEMA: Final = 1
"""The header's format version."""
HEADER_KEYS: Final = frozenset(
    {
        "kind",
        "schema",
        "request",
        "definition",
        "guidance",
        "uci",
        "fingerprint",
        "releases",
        "calls",
    }
)
"""The keys the header holds."""
_SHA256: Final = re.compile(r"[0-9a-f]{64}")


class Request(StrEnum):
    """The gap request a candidates file answers."""

    E1 = "e1"
    """E1's count-based request: each count-flagged cell's shortfall below the threshold."""
    COMPLETENESS = "completeness"
    """The completeness request: the threshold for each cell holding a per-subgroup gap."""


@dataclass(frozen=True, slots=True)
class Header:
    """The run a candidates file was generated for.

    Attributes:
        request: The gap request filled.
        definition: The gap definition whose gaps guided each cell.
        guidance: The guidance level.
        uci: The pinned sha256 of the UCI archive the split was made from.
        fingerprint: The split's fingerprint (`bko.datasets.split.Split.fingerprint`).
        releases: Each loaded release's concept prefix and pinned sha256, sorted by prefix.
        calls: Each call's cell key and records asked, in call order from call 1.
    """

    request: Request
    definition: Definition
    guidance: Level
    uci: str
    fingerprint: str
    releases: tuple[tuple[str, str], ...]
    calls: tuple[tuple[str, int], ...]


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


@dataclass(frozen=True, slots=True)
class Candidates:
    """A candidates file: its header and its record lines.

    Attributes:
        header: The run the file was generated for.
        lines: The record lines, in file order.
    """

    header: Header
    lines: tuple[CandidateLine, ...]


def render_header(header: Header) -> str:
    """The header line, ending in a newline."""
    line = {
        "kind": "header",
        "schema": HEADER_SCHEMA,
        "request": header.request.value,
        "definition": header.definition.value,
        "guidance": header.guidance.value,
        "uci": header.uci,
        "fingerprint": header.fingerprint,
        "releases": dict(header.releases),
        "calls": [list(call) for call in header.calls],
    }
    return json.dumps(line, sort_keys=True) + "\n"


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


def read(data: bytes) -> Candidates:
    """A candidates file's header, then its record lines in file order.

    Raises:
        CandidatesFormatError: If the bytes are not UTF-8, the last line has no newline, the file
            is empty or its first line is not a schema `HEADER_SCHEMA` header, a later line is
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
    if not text:
        msg = "the candidates file is empty, so it has no header"
        raise CandidatesFormatError(msg)
    first, *rest = text.split("\n")[:-1]
    header = _header(first)
    lines: list[CandidateLine] = []
    seen: set[tuple[int, int]] = set()
    cells: dict[int, str] = {}
    for number, raw in enumerate(rest, start=2):
        line = _line(number, raw)
        if (line.call, line.position) in seen:
            msg = f"line {number} repeats call {line.call} position {line.position}"
            raise CandidatesFormatError(msg)
        if cells.setdefault(line.call, line.cell) != line.cell:
            msg = f"line {number} puts call {line.call} in {line.cell!r}, not {cells[line.call]!r}"
            raise CandidatesFormatError(msg)
        seen.add((line.call, line.position))
        lines.append(line)
    return Candidates(header, tuple(lines))


def _header(raw: str) -> Header:
    """The header line, refused unless it holds exactly the header's keys, each well formed."""
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        _not_a_header(f"it is not JSON: {exc}", exc)
    if not isinstance(value, dict) or value.get("kind") != "header":
        _not_a_header("it holds no kind header")
    schema = value.get("schema")
    if type(schema) is not int or schema != HEADER_SCHEMA or value.keys() != HEADER_KEYS:
        _not_a_header(f"its schema or keys differ from {sorted(HEADER_KEYS)}")
    try:
        request = Request(value["request"])
        definition = Definition(value["definition"])
        guidance = Level(value["guidance"])
    except (ValueError, TypeError) as exc:
        _not_a_header("its request, definition or guidance is not a known value", exc)
    if not all(_is_sha256(value[key]) for key in ("uci", "fingerprint")):
        _not_a_header("its uci or fingerprint is not a sha256")
    releases = value["releases"]
    if not isinstance(releases, dict) or not all(map(_is_sha256, releases.values())):
        _not_a_header("its releases are not a sha256 per prefix")
    calls = value["calls"]
    if not isinstance(calls, list) or not all(_is_planned_call(call) for call in calls):
        _not_a_header("its calls are not [cell key, records asked of at least 1] pairs")
    return Header(
        request=request,
        definition=definition,
        guidance=guidance,
        uci=value["uci"],
        fingerprint=value["fingerprint"],
        releases=tuple(sorted(releases.items())),
        calls=tuple((cell, asked) for cell, asked in calls),
    )


def _not_a_header(why: str, cause: BaseException | None = None) -> NoReturn:
    """Refuse the first line as a header, for `why`."""
    msg = f"line 1 is not a schema {HEADER_SCHEMA} header: {why}"
    raise CandidatesFormatError(msg) from cause


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_planned_call(call: object) -> bool:
    return (
        isinstance(call, list)
        and len(call) == 2
        and isinstance(call[0], str)
        and type(call[1]) is int
        and call[1] >= 1
    )


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
