"""The record a direct draw yields and the draw log's line for it.

`Draw` holds one record drawn for a cell, with the concepts drawn present, their closure under
`is_a`, the fields read from the label group's records and each copied field's training record id.
`draw_line` writes the draw log's line for one record. `bko.generate.direct` re-exports all three.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from bko.gaps.observe import Cell, cell_key


type Record = Mapping[str, int | float]


@dataclass(frozen=True, slots=True)
class Draw:
    """One record drawn for a cell.

    Attributes:
        cell: The cell's key.
        record: The record's value per column, in column order.
        concepts: The concepts drawn present, label-only ones included.
        closed: `concepts` closed upward under `is_a` within each concept's release; recorded,
            and read into a field only by `Presence.LOGIC`.
        fallback: The fields read from the label group's records because the cell's showed no
            record matching the draw, or, for a pooled unit, by the pooled rule's label-group
            branch (`bko.generate.donors._pick_pooled`).
        sources: The training record id each copied field was read from; `sex` and the label
            follow the cell and have none. An averaged field's is its primary donor's.
        averaged: Each averaged field's donor ids, the primary's first, under a spec that sets
            `microaggregate` (`bko.generate.micro`); empty otherwise.
    """

    cell: Cell
    record: Record
    concepts: frozenset[str]
    closed: frozenset[str]
    fallback: frozenset[str]
    sources: Mapping[str, int] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    averaged: Mapping[str, tuple[int, ...]] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", MappingProxyType(dict(self.record)))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        object.__setattr__(self, "averaged", MappingProxyType(dict(self.averaged)))


def draw_line(call: int, position: int, draw: Draw) -> str:
    """The draw log's line for one record, ending in a newline, its keys sorted."""
    line = {
        "kind": "draw",
        "call": call,
        "cell": cell_key(draw.cell),
        "position": position,
        "concepts": sorted(draw.concepts),
        "closed": sorted(draw.closed),
        "fallback": sorted(draw.fallback),
    }
    return json.dumps(line, sort_keys=True) + "\n"
