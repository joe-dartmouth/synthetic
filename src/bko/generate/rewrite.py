"""The rewrite call: rows sent with a `row` index each, and their fields asked back changed.

A `Rewrite` holds one call's rows, each only the fields to rewrite, and each measured field's
range in the holdout's training split. `render_rewrite` writes its prompt: `REWRITE`'s first
sentence, each field's meaning, the ranges, then each row as a JSON object on its own line, its
`row` index first. `rewrite_schema` fixes the reply: a `records` array whose every object holds a
`row` from the indices asked and each field rewritten, and no other key, each coded field an
`enum` over its codebook and each whole-number field an `enum` over its range, since structured
outputs cannot bound a number. A fractional field is a plain `number`, bounded in the prompt
alone. `bko.generate.calls.request` sends it; `bko.evaluate.perturb` merges the reply by `row`.
ADR `research/identifier-perturbation` records the call.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from bko.datasets.frames import FrameSpec
from bko.generate.prompt import field_units


ROW: Final = "row"
"""The key that names a row's index within its call."""

REWRITE: Final = re.compile(
    r"Rewrite these (?P<rows>\d+) records' fields: (?P<fields>\w+(?:, \w+)*)\."
)
"""The first sentence of every rewrite user turn."""

REWRITE_SYSTEM: Final = (
    "You rewrite fields of patient records so that no rewritten record can be matched to the "
    "patient it came from, while each record stays a plausible hospital encounter. You change "
    "only the fields you are given, keep every value within the codes and ranges stated, and "
    "reply with JSON that follows the schema."
)
"""The system prompt of every rewrite call."""


@dataclass(frozen=True, slots=True)
class Rewrite:
    """One rewrite call's rows.

    Attributes:
        fields: The fields rewritten, in frame order.
        rows: Each row's value of each field, in row order; a row's index is its position.
        bounds: Each measured field's lowest and highest value in the training split.
    """

    fields: tuple[str, ...]
    rows: tuple[Mapping[str, int | float], ...]
    bounds: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        """Refuse a row that lacks or adds a field.

        Raises:
            ValueError: If a row's keys are not `fields`.
        """
        for index, row in enumerate(self.rows):
            if set(row) != set(self.fields):
                msg = f"row {index} holds {sorted(row)}, not the fields {list(self.fields)}"
                raise ValueError(msg)


def render_rewrite(rewrite: Rewrite, *, frame: FrameSpec) -> tuple[str, str]:
    """The system prompt and user turn of `rewrite`'s call.

    Raises:
        ValueError: If a measured field of `rewrite` has no bounds.
    """
    units = field_units(frame)
    measured = [name for name in rewrite.fields if name not in frame.categorical]
    missing = [name for name in measured if name not in rewrite.bounds]
    if missing:
        msg = f"the measured fields {missing} have no training range"
        raise ValueError(msg)
    meanings = "\n".join(_meaning(name, frame) for name in rewrite.fields)
    ranges = "; ".join(
        f"{name} {rewrite.bounds[name][0]:g} to {rewrite.bounds[name][1]:g} {units[name]}"
        for name in measured
    )
    listed = "\n".join(
        json.dumps({ROW: index, **{name: row[name] for name in rewrite.fields}})
        for index, row in enumerate(rewrite.rows)
    )
    opening = (
        f"Rewrite these {len(rewrite.rows)} records' fields: {', '.join(rewrite.fields)}. "
        "Change each record so that it no longer matches the patient it was drawn from. Return "
        f'one object per record, in the order given, holding its "{ROW}" and every field listed, '
        "and no other key."
    )
    parts = [
        opening,
        f"The fields:\n{meanings}",
        *([f"Each measured field's range in the training data: {ranges}."] if ranges else []),
        f"The records:\n{listed}",
    ]
    return REWRITE_SYSTEM, "\n\n".join(parts)


def rewrite_schema(rewrite: Rewrite, *, frame: FrameSpec) -> dict[str, object]:
    """The structured-output schema of `rewrite`'s reply.

    Returns:
        A JSON schema whose every object is closed and requires all of its properties.
    """
    fields: dict[str, object] = {ROW: {"type": "integer", "enum": list(range(len(rewrite.rows)))}}
    for name in rewrite.fields:
        if name in frame.categorical:
            fields[name] = {"type": "integer", "enum": sorted(frame.codebook[name])}
        elif name in frame.fractional:
            fields[name] = {"type": "number"}
        else:
            low, high = rewrite.bounds[name]
            values = range(math.ceil(low), math.floor(high) + 1)
            fields[name] = {"type": "integer", "enum": list(values)}
    record = {
        "type": "object",
        "properties": fields,
        "required": [ROW, *rewrite.fields],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"records": {"type": "array", "items": record}},
        "required": ["records"],
        "additionalProperties": False,
    }


def _meaning(name: str, frame: FrameSpec) -> str:
    """`name`'s line: its description, then its codes' meanings or its unit."""
    codes = frame.codebook.get(name, {}) if name in frame.categorical else {}
    details = "".join(f"; {code} = {meaning}" for code, meaning in codes.items())
    unit = "" if codes else f"; {field_units(frame)[name]}"
    return f"- {name}: {frame.descriptions[name]}{details or unit}"
