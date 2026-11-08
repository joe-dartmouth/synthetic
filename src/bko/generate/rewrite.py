"""The rewrite call: rows sent with a `row` index each, and their fields asked back changed.

A `Rewrite` holds one call's rows, each only the fields to rewrite, never `age`, `sex` or the
label, each measured field's range in the holdout's training split, and each field's quotas: how
many of the call's rows each value or inclusive range must hold (`Share`), which
`bko.evaluate.quotas` sets from the cell's label-group training rows. `render_rewrite` writes its
prompt: `REWRITE`'s first sentence and `OPENING`, each field's meaning, the ranges, the counts
under `COUNTS_HEADER`, then each row as a JSON object on its own line, its `row` index first.
`rewrite_schema` fixes the reply: a `records` array whose every object holds a `row` from the
indices asked and each field rewritten, and no other key, each coded field an `enum` over its
codebook and each whole-number field an `enum` over its range, since structured outputs cannot
bound a number. A fractional field is a plain `number`, bounded in the prompt alone.
`bko.generate.calls.request` sends it; `bko.evaluate.perturbed` merges the reply by `row`. ADR
`research/identifier-perturbation` records the call.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
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
    "patient it came from, while each record stays a plausible, internally consistent hospital "
    "encounter. You change only the fields you are given, keep every value within the codes, "
    "ranges and counts stated, and reply with JSON that follows the schema."
)
"""The system prompt of every rewrite call."""

OPENING: Final = (
    "Change each record so that its combination of values no longer matches the patient it was "
    "drawn from; a field may keep its value. Across the records, each field must hold exactly the "
    f'counts listed below. Return one object per record, in the order given, holding its "{ROW}" '
    "and every field listed, and no other key."
)
"""What every rewrite user turn asks after `REWRITE`'s first sentence."""

COUNTS_HEADER: Final = (
    "Each field's counts across these records (a value or an inclusive range, then x and its "
    "number of records):"
)
"""The first line of the quota block, which precedes the records."""


@dataclass(frozen=True, slots=True)
class Share:
    """How many of a call's rows must hold a field's value from `low` to `high` inclusive."""

    low: int | float
    high: int | float
    count: int


@dataclass(frozen=True, slots=True)
class Rewrite:
    """One rewrite call's rows.

    Attributes:
        fields: The fields rewritten, in frame order.
        rows: Each row's value of each field, in row order; a row's index is its position.
        bounds: Each measured field's lowest and highest value in the training split.
        quotas: Each field's shares of the rows, strictly ascending and apart (each share's
            `low` above the previous share's `high`), each non-empty, together summing to the
            rows.
    """

    fields: tuple[str, ...]
    rows: tuple[Mapping[str, int | float], ...]
    bounds: Mapping[str, tuple[float, float]]
    quotas: Mapping[str, tuple[Share, ...]]

    def __post_init__(self) -> None:
        """Refuse the cell's fields, a row's other keys, and quotas that do not fit the rows.

        The quotas must be one set of shares per field, each summing to the rows.

        Raises:
            ValueError: If `fields` holds `age` or `sex`, a row's keys are not `fields`, the
                quotas' fields are not `fields`, a field's shares do not sum to the rows, a
                share's `low` exceeds its `high` or it holds no row, or a share's `low` does not
                exceed the previous share's `high`.
        """
        cell = sorted({"age", "sex"} & set(self.fields))
        if cell:
            msg = f"a rewrite may not send the cell's fields {cell}"
            raise ValueError(msg)
        for index, row in enumerate(self.rows):
            if set(row) != set(self.fields):
                msg = f"row {index} holds {sorted(row)}, not the fields {list(self.fields)}"
                raise ValueError(msg)
        if set(self.quotas) != set(self.fields):
            msg = f"the quotas name {sorted(self.quotas)}, not the fields {list(self.fields)}"
            raise ValueError(msg)
        for name in self.fields:
            shares = self.quotas[name]
            total = sum(share.count for share in shares)
            if total != len(self.rows):
                msg = f"{name}'s quotas sum to {total}, not the {len(self.rows)} rows"
                raise ValueError(msg)
            for share in shares:
                if share.low > share.high or share.count <= 0:
                    msg = f"{name}'s share {share} is inverted or holds no row"
                    raise ValueError(msg)
            for before, share in pairwise(shares):
                if share.low <= before.high:
                    msg = f"{name}'s shares do not ascend apart: {share} follows {before}"
                    raise ValueError(msg)


def render_rewrite(rewrite: Rewrite, *, frame: FrameSpec) -> tuple[str, str]:
    """The system prompt and user turn of `rewrite`'s call.

    Raises:
        ValueError: If `rewrite` sends `frame`'s label, or a measured field of it has no bounds.
    """
    if frame.label in rewrite.fields:
        msg = f"a rewrite may not send the label {frame.label}"
        raise ValueError(msg)
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
    opening = f"Rewrite these {len(rewrite.rows)} records' fields: {', '.join(rewrite.fields)}. "
    counts = "\n".join(
        f"- {name}: " + ", ".join(_share(share) for share in rewrite.quotas[name])
        for name in rewrite.fields
    )
    parts = [
        opening + OPENING,
        f"The fields:\n{meanings}",
        *([f"Each measured field's range in the training data: {ranges}."] if ranges else []),
        f"{COUNTS_HEADER}\n{counts}",
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


def _share(share: Share) -> str:
    """`share` as the prompt states it: its value or inclusive range, then x and its count.

    Each end is written in full, an `int` as its digits and a `float` as its shortest round-trip
    form, so no end is rounded.
    """
    if share.low == share.high:
        return f"{share.low} x{share.count}"
    return f"{share.low}-{share.high} x{share.count}"


def _meaning(name: str, frame: FrameSpec) -> str:
    """`name`'s line: its description, then its codes' meanings or its unit."""
    codes = frame.codebook.get(name, {}) if name in frame.categorical else {}
    details = "".join(f"; {code} = {meaning}" for code, meaning in codes.items())
    unit = "" if codes else f"; {field_units(frame)[name]}"
    return f"- {name}: {frame.descriptions[name]}{details or unit}"
