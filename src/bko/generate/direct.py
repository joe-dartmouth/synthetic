"""Records drawn from the fused BKB's concept shares, with no API call.

The fused BKB holds, per mapped concept, a prior from the cell's dataset fragment and one from its
label group's (`bko.gaps.observe.dataset_fragment`), and every `is_a` S-node at probability 1, so
what a sampler can draw is each concept's presence. `bko.gaps.shares.shares` reads a concept's share
from the two priors; `sample` draws each concept present at its share, records the drawn set's
closure upward under `is_a`, and reads field values back from training records that show exactly the
drawn concepts of each field group. Draws come from one `numpy.random.default_rng(seed)`, consumed
in quota, record and column order, so `bko.bkb` stays standard-library only. ADR
`research/bkb-direct-generation` records each decision and its reasons.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from bko.datasets.cells import SEXES, cell_keys
from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell, Observed, by_concept, cell_key, observe_records, record_cells
from bko.gaps.shares import label_only, nested, shares
from bko.generate.errors import SamplingError
from bko.ontology.schema import FieldMapping
from bko.ontology.terms import Ontology, ancestors


logger = logging.getLogger(__name__)

type Record = Mapping[str, int | float]
type _Source = tuple[int, dict[str, float]]
"""A training record with its id."""


@dataclass(frozen=True, slots=True)
class Draw:
    """One record drawn for a cell.

    Attributes:
        cell: The cell's key.
        record: The record's value per column, in column order.
        concepts: The concepts drawn present, label-only ones included.
        closed: `concepts` closed upward under `is_a` within each concept's release; recorded,
            never read into a field.
        fallback: The fields read from the label group's records because the cell's showed no
            record matching the draw.
        sources: The training record id each copied field was read from; `sex` and the label
            follow the cell and have none.
    """

    cell: Cell
    record: Record
    concepts: frozenset[str]
    closed: frozenset[str]
    fallback: frozenset[str]
    sources: Mapping[str, int] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", MappingProxyType(dict(self.record)))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))


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


def sample(
    train: pd.DataFrame,
    quota: Mapping[Cell, int],
    *,
    rows: Sequence[FieldMapping],
    ontologies: Mapping[str, Ontology],
    name: str,
    seed: int,
    frame: FrameSpec,
    tag: tuple[int, str] | None = None,
) -> dict[Cell, tuple[Draw, ...]]:
    """Draw `quota` records per cell from the fused BKB's concept shares.

    Per record: every concept but the label-only ones is drawn present at its share, in sorted
    order, a nested concept after the one inside it (`nested`); the label-only ones follow the
    cell's label. Then, in column order: `age` from a cell record, else a label-group record in
    the cell's band; `sex` and the label from the cell; each field group from one record showing
    exactly the group's drawn concepts, the cell's, else the label group's; and every other field
    from a cell record, else a label-group record.

    Args:
        train: The training records, never the test split.
        quota: Records per cell, in the order drawn.
        rows: The field mappings.
        ontologies: Each loaded release by concept prefix, for the closure.
        name: The dataset's source name, as `bko.gaps.observe.record_cells` takes it.
        seed: The generator's seed.
        frame: The frame the records belong to.
        tag: The holdout and the fill the draw serves, which its `direct` and `fallback` lines
            name; they read `holdout=-1 fill=none` when None.

    Returns:
        Each cell's draws, in quota order.

    Raises:
        SamplingError: If a field group has no record matching a draw in the cell or its label
            group, or a free field has no record to read from.
        GapError: As `bko.gaps.observe.record_cells` and `observe_records`.
        ValueError: As `nested`, for nesting deeper or wider than one pair.
    """
    mappings = tuple(rows)
    cells = record_cells(train, mappings, name=name, edges=frame.edges)
    keys = cell_keys(train, edges=frame.edges)
    plan = _Plan(mappings, ontologies, frame)
    groups = _label_groups(train, keys, {int(cell[2]) for cell in quota}, mappings, name, frame)
    rng = np.random.default_rng(seed)
    holdout, fill = (-1, "none") if tag is None else tag
    drawn: dict[Cell, tuple[Draw, ...]] = {}
    for cell, count in quota.items():
        group = groups[int(cell[2])]
        inside = keys.eq(pd.Series(cell, index=keys.columns)).all(axis=1).to_numpy()
        sources = _Sources(
            cell=cell,
            share=shares(cells.each[cell], group.observed),
            own=_records(train.loc[inside], frame),
            group=group.records,
            banded=[
                r for r, band in zip(group.records, group.bands, strict=True) if band == cell[0]
            ],
        )
        logger.info(
            "direct holdout=%d fill=%s key=%s rows=%d records=%d group_records=%d",
            holdout,
            fill,
            cell_key(cell),
            count,
            len(sources.own),
            len(sources.group),
        )
        draws = tuple(plan.draw(sources, rng) for _ in range(count))
        for field, times in sorted(Counter(f for d in draws for f in d.fallback).items()):
            logger.info(
                "fallback holdout=%d fill=%s key=%s field=%s draws=%d",
                holdout,
                fill,
                cell_key(cell),
                field,
                times,
            )
        drawn[cell] = draws
    return drawn


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


@dataclass(frozen=True, slots=True, eq=False)
class _Group:
    """A label's training records: observed, as records, and each record's age band."""

    observed: Observed
    records: list[_Source]
    bands: list[str]


@dataclass(frozen=True, slots=True, eq=False)
class _Sources:
    """What one cell's draws read: its shares and the records each field may come from."""

    cell: Cell
    share: Mapping[str, float]
    own: list[_Source]
    group: list[_Source]
    banded: list[_Source]


class _Plan:
    """What every draw reads: the concept order, the nesting, the groups and the free fields."""

    def __init__(
        self, rows: Sequence[FieldMapping], ontologies: Mapping[str, Ontology], frame: FrameSpec
    ) -> None:
        self._frame = frame
        self._by_concept = by_concept(rows)
        self._label_only = label_only(rows, frame=frame)
        self._inner_of = {outer: inner for inner, outer in nested(rows).items()}
        self._order = sorted(
            (c for c in self._by_concept if c not in self._label_only),
            key=lambda concept: (concept in self._inner_of, concept),
        )
        groups = field_groups(rows, frame=frame)
        grouped = {field for group in groups for field in group.fields}
        self._first = {group.fields[0]: group for group in groups}
        self._free = frozenset(frame.columns) - grouped - {"age", "sex", frame.label}
        self._ontologies = ontologies

    def draw(self, sources: _Sources, rng: np.random.Generator) -> Draw:
        """One record: its concepts, then its fields in column order, then the closure."""
        present = self._concepts(sources, rng)
        record: dict[str, int | float] = {}
        fallback: set[str] = set()
        read: dict[str, int] = {}
        cell = sources.cell
        for column in self._frame.columns:
            if column == "age":
                (source, chosen), fell = _pick(cell, (column,), sources.own, sources.banded, rng)
                fields: tuple[str, ...] = (column,)
            elif column == "sex":
                record[column] = next(code for code, sex in SEXES.items() if sex == cell[1])
                continue
            elif column == self._frame.label:
                record[column] = int(cell[2])
                continue
            elif column in self._first:
                fields = self._first[column].fields
                shows = self._pattern(self._first[column], present)
                own = [r for r in sources.own if shows(r[1])]
                other = [r for r in sources.group if shows(r[1])]
                (source, chosen), fell = _pick(cell, fields, own, other, rng)
            elif column in self._free:
                fields = (column,)
                (source, chosen), fell = _pick(cell, fields, sources.own, sources.group, rng)
            else:
                continue
            for name in fields:
                record[name] = self._value(name, chosen)
                read[name] = source
            if fell:
                fallback.update(fields)
        ordered = {column: record[column] for column in self._frame.columns}
        concepts = frozenset(present)
        closed = self._closure(concepts)
        return Draw(cell, ordered, concepts, closed, frozenset(fallback), read)

    def _concepts(self, sources: _Sources, rng: np.random.Generator) -> set[str]:
        """The concepts drawn present: label-only ones by the label, each other at its share."""
        label = {self._frame.label: int(sources.cell[2])}
        present = {
            concept
            for concept in self._label_only
            if any(row.applies(label) for row in self._by_concept[concept])
        }
        share = sources.share
        for concept in self._order:
            inner = self._inner_of.get(concept)
            if inner is None:
                chance = share[concept]
            elif inner in present or share[inner] == 1:
                present.add(concept)
                continue
            else:
                chance = (share[concept] - share[inner]) / (1 - share[inner])
            if rng.random() < chance:
                present.add(concept)
        return present

    def _closure(self, concepts: frozenset[str]) -> frozenset[str]:
        closed = set(concepts)
        for concept in concepts:
            prefix = concept.partition(":")[0]
            ontology = self._ontologies.get(prefix)
            if ontology is not None:
                closed |= ancestors(ontology, concept, prefix=prefix)
        return frozenset(closed)

    def _pattern(
        self, group: FieldGroup, present: Collection[str]
    ) -> Callable[[Mapping[str, float]], bool]:
        """Whether a record shows exactly the group's drawn concepts: an absent one by no row."""

        def shows(record: Mapping[str, float]) -> bool:
            return all(
                any(row.applies(record) for row in self._by_concept[concept])
                == (concept in present)
                for concept in group.concepts
            )

        return shows

    def _value(self, field: str, record: Mapping[str, float]) -> int | float:
        value = record[field]
        return float(value) if field in self._frame.fractional else int(value)


def _pick(
    cell: Cell,
    fields: tuple[str, ...],
    own: Sequence[_Source],
    other: Sequence[_Source],
    rng: np.random.Generator,
) -> tuple[_Source, bool]:
    """A record drawn from `own`, else from `other`, with its id, and whether it fell back.

    Raises:
        SamplingError: If both are empty.
    """
    if own:
        return own[int(rng.integers(len(own)))], False
    if other:
        return other[int(rng.integers(len(other)))], True
    msg = f"{cell_key(cell)} cannot draw {'+'.join(fields)}: no training record matches the draw"
    raise SamplingError(msg)


def label_group(
    train: pd.DataFrame, label: int, rows: Sequence[FieldMapping], *, name: str, frame: FrameSpec
) -> Observed:
    """The training records of `label`, observed as the label group whose prior a share reads.

    Raises:
        GapError: As `observe_records`.
    """
    members = train.loc[train[frame.label].eq(label)]
    return observe_records(members, rows, name=f"{name}/{frame.label}={label}")


def _label_groups(
    train: pd.DataFrame,
    keys: pd.DataFrame,
    labels: Collection[int],
    rows: Sequence[FieldMapping],
    name: str,
    frame: FrameSpec,
) -> dict[int, _Group]:
    groups: dict[int, _Group] = {}
    for label in sorted(labels):
        members = train.loc[train[frame.label].eq(label)]
        observed = label_group(train, label, rows, name=name, frame=frame)
        bands = [str(band) for band in keys.loc[members.index, keys.columns[0]]]
        groups[label] = _Group(observed, _records(members, frame), bands)
    return groups


def _records(frame: pd.DataFrame, spec: FrameSpec) -> list[_Source]:
    """Each record of `frame` with its id, in row order."""
    columns = [str(column) for column in spec.columns]
    return [
        (int(index), dict(zip(columns, values, strict=True)))
        for index, *values in frame[columns].itertuples(name=None)
    ]
