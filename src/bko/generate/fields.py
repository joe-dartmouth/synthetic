"""The field groups a direct draw reads together, and the free fields no group reads.

A field group joins the fields that one concept's mapping rows read, so a draw copies them from
one training record. ADR `research/bkb-direct-generation` records the rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bko.datasets.frames import FrameSpec
from bko.gaps.observe import by_concept
from bko.ontology.schema import FieldMapping


@dataclass(frozen=True, slots=True)
class FieldGroup:
    """Fields whose values are drawn together, from one record, and the concepts they show.

    Attributes:
        fields: The fields, in column order.
        concepts: The concepts whose rows read them, sorted.
    """

    fields: tuple[str, ...]
    concepts: tuple[str, ...]


def field_groups(rows: Sequence[FieldMapping], *, frame: FrameSpec) -> tuple[FieldGroup, ...]:
    """The fields the rows read, joined when one concept's rows read both, in column order.

    The label is no field of a group: label-only rows follow the cell.
    """
    parent: dict[str, str] = {}

    def root(field: str) -> str:
        while parent[field] != field:
            field = parent[field]
        return field

    reads: dict[str, set[str]] = {}
    for concept, mapped in by_concept(rows).items():
        fields = {c.field for row in mapped for c in row.when if c.field != frame.label}
        if not fields:
            continue
        reads[concept] = fields
        for field in fields:
            parent.setdefault(field, field)
        first, *rest = sorted(fields)
        for field in rest:
            parent[root(field)] = root(first)
    order = {name: index for index, name in enumerate(frame.columns)}
    joined: dict[str, set[str]] = {}
    for field in parent:
        joined.setdefault(root(field), set()).add(field)
    groups = [
        FieldGroup(
            fields=tuple(sorted(fields, key=order.__getitem__)),
            concepts=tuple(sorted(c for c, read in reads.items() if read & fields)),
        )
        for fields in joined.values()
    ]
    return tuple(sorted(groups, key=lambda group: order[group.fields[0]]))


def free_fields(rows: Sequence[FieldMapping], *, frame: FrameSpec) -> tuple[str, ...]:
    """`frame`'s columns no field group reads, but `age`, `sex` and the label, in column order."""
    grouped = {name for group in field_groups(rows, frame=frame) for name in group.fields}
    cell = {"age", "sex", frame.label}
    return tuple(name for name in frame.columns if name not in grouped and name not in cell)
