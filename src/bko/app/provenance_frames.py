"""The frames behind a record's trace: read from `trace.jsonl` or from a demo fill's run folder.

Nothing here imports streamlit; `bko.app.browse` renders what these return. A trace line is
the JSON object `bko.evaluate.trace_lines.Trace.line` writes, held read-only (`frozen`); every
field but the arm and the holdout may be absent, and a frame leaves out what its line lacks (ADR
`research/record-explainability`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import pandas as pd

from bko.evaluate.call_prompts import CALLS
from bko.evaluate.holdout_candidates import read
from bko.evaluate.trace_lines import Fill, trace_messages


if TYPE_CHECKING:
    from bko.app.steps import Outcome


type Line = Mapping[str, object]

CANDIDATES: Final = "holdout-candidates.jsonl"
"""A demo fill's candidates file in its run folder."""

PROMPT_PARTS: Final = ("opening", "bounds", "profile", "summary", "prescribed", "shown")
"""The sections of a Messages prompt shown as text blocks, in prompt order."""


@dataclass(frozen=True, slots=True)
class TraceIndex:
    """A trace file's lines, by arm and holdout.

    Attributes:
        arms: Each arm, in file order.
        fills: Each arm's holdouts, in file order, and each holdout's lines.
    """

    arms: tuple[str, ...]
    fills: Mapping[str, Mapping[int, tuple[Line, ...]]]

    def holdouts(self, arm: str) -> tuple[int, ...]:
        """The holdouts `arm` has lines on, in file order."""
        return tuple(self.fills[arm])

    def lines(self, arm: str, holdout: int) -> tuple[Line, ...]:
        """The lines of `arm` on `holdout`, in file order."""
        return self.fills[arm][holdout]


def frozen(value: object) -> object:
    """`value` read-only all the way down: each mapping a `MappingProxyType`, each list a tuple."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: frozen(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(frozen(item) for item in value)
    return value


def plain(value: object) -> object:
    """A `frozen` value as JSON holds it: each mapping a dict, each tuple a list."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value


def trace_index(text: str) -> TraceIndex:
    """Index the trace lines of `text`, each held `frozen`.

    Raises:
        ValueError: If a line is not JSON.
        TypeError: If a line is not an object, or names no arm or no integer holdout.
    """
    fills: dict[str, dict[int, list[Line]]] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        line = json.loads(raw)
        if not isinstance(line, dict):
            msg = f"line {number} is not a JSON object"
            raise TypeError(msg)
        arm, holdout = line.get("arm"), line.get("holdout")
        if not isinstance(arm, str) or not isinstance(holdout, int) or isinstance(holdout, bool):
            msg = f"line {number} names no arm and integer holdout"
            raise TypeError(msg)
        fills.setdefault(arm, {}).setdefault(holdout, []).append(cast(Line, frozen(line)))
    built = {arm: {h: tuple(lines) for h, lines in held.items()} for arm, held in fills.items()}
    return TraceIndex(tuple(fills), built)


def read_index(path: Path) -> TraceIndex:
    """Index the trace file at `path`.

    Raises:
        OSError: If it cannot be read.
        ValueError: As `trace_index`, or if it is not UTF-8.
        TypeError: As `trace_index`.
    """
    return trace_index(path.read_text(encoding="utf-8"))


def _mapping(line: Line, key: str) -> Mapping[str, object]:
    value = line.get(key)
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _listed(line: Line, key: str) -> list[object]:
    value = line.get(key)
    return list(value) if isinstance(value, list | tuple) else []


def traced(line: Line) -> bool:
    """Whether `line` stores what asked for its record."""
    return bool(_mapping(line, "request"))


def record_frame(line: Line) -> pd.DataFrame:
    """The record's value per field, as text, so mixed kinds share one column."""
    record = _mapping(line, "record")
    return pd.DataFrame(
        {"field": list(map(str, record)), "value": [str(value) for value in record.values()]}
    )


def origin_frame(line: Line) -> pd.DataFrame:
    """The record's fill, its position, and its call and cell; a direct draw has no call."""
    names = ("arm", "holdout", "fill", "record_id", "call", "cell")
    return pd.DataFrame([{name: line.get(name) for name in names}])


def request_shown(line: Line) -> dict[str, object]:
    """The request, without a Messages prompt's sections, which their own sections show."""
    return {key: value for key, value in _mapping(line, "request").items() if key != "sections"}


def prompt_sections(line: Line) -> Mapping[str, object]:
    """A Messages record's prompt sections as `bko.generate.prompt.sections` read them."""
    return _mapping(_mapping(line, "request"), "sections")


def statement_frame(line: Line) -> pd.DataFrame:
    """The gap statements a Messages prompt rendered, one per row."""
    found = _listed(prompt_sections(line), "findings")
    return pd.DataFrame({"statement": [str(statement) for statement in found]})


def prompt_texts(line: Line) -> list[str]:
    """The text of each prompt section the call held, in prompt order."""
    shown = prompt_sections(line)
    return [str(shown[part]) for part in PROMPT_PARTS if isinstance(shown.get(part), str)]


CONCEPT_COLUMNS: Final = (
    "concept",
    "label",
    "present",
    "share",
    "cell_prior",
    "group_prior",
    "cell_records",
    "cell_holders",
    "group_records",
    "group_holders",
    "group_weight",
)


def concept_frame(line: Line) -> pd.DataFrame:
    """A direct record's concepts: each one's share, its parts, its label and its draw."""
    rows = [cast(Line, each) for each in _listed(line, "concepts") if isinstance(each, Mapping)]
    return pd.DataFrame(
        [{name: row.get(name) for name in CONCEPT_COLUMNS} for row in rows],
        columns=list(CONCEPT_COLUMNS),
    )


def _named(value: object) -> str:
    """An S-node head or tail component, `[name, state]`, as `name = state`."""
    if isinstance(value, list | tuple) and len(value) == 2:
        return f"{value[0]} = {value[1]}"
    return json.dumps(plain(value))


def snode_frame(line: Line) -> pd.DataFrame:
    """A direct record's S-nodes: each one's head, tail and probability."""
    rows = []
    for concept in _listed(line, "concepts"):
        if not isinstance(concept, Mapping):
            continue
        for snode in _listed(cast(Line, concept), "snodes"):
            if isinstance(snode, Mapping):
                tail = snode.get("tail")
                parts = tail if isinstance(tail, list | tuple) else []
                rows.append(
                    {
                        "head": _named(snode.get("head")),
                        "tail": ", ".join(_named(part) for part in parts),
                        "prob": snode.get("prob"),
                    }
                )
    return pd.DataFrame(rows, columns=["head", "tail", "prob"])


def source_frame(line: Line) -> pd.DataFrame:
    """A direct record's source training record per field, and whether the field fell back.

    A field falls back when it is read from its label group's records rather than its cell's.
    """
    sources = _mapping(line, "sources")
    fallback = set(map(str, _listed(line, "fallback")))
    return pd.DataFrame(
        {
            "field": list(map(str, sources)),
            "source_record": list(sources.values()),
            "fallback": [str(name) in fallback for name in sources],
        }
    )


def closed_frame(line: Line) -> pd.DataFrame:
    """A direct record's drawn concepts closed upward under `is_a`."""
    return pd.DataFrame({"closed": [str(concept) for concept in _listed(line, "closed")]})


def validation_frame(line: Line) -> pd.DataFrame:
    """The record's validation outcome and the checks it failed."""
    failed = ", ".join(map(str, _listed(line, "failed")))
    return pd.DataFrame([{"outcome": line.get("outcome"), "failed": failed}])


def demo_traces(outcome: Outcome) -> tuple[Line, ...]:
    """A demo Messages fill's trace lines, `frozen`, one per added row in its order.

    It reads only the fill's own call log and candidates file in its run folder, never the data
    root.

    Raises:
        OSError: If the candidates file cannot be read.
        CandidatesFormatError: If it is not a candidates file.
        EvaluationError: As `bko.evaluate.trace_lines.trace_messages`, `IdentityError` included.
    """
    lines = [
        line
        for line in read((outcome.folder / CANDIDATES).read_bytes()).candidates
        if line.arm == outcome.name and line.fill == "gap"
    ]
    if not lines:
        return ()
    fill = Fill(lines[0].holdout, outcome.name, "")
    traces = trace_messages(fill, lines, outcome.request, calls=outcome.folder / CALLS)
    return tuple(cast(Line, frozen(json.loads(each.line()))) for each in traces)
