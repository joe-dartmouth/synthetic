"""Records drawn from the fused BKB's concept shares, with no API call.

The fused BKB holds, per mapped concept, a prior from the cell's dataset fragment and one from its
label group's (`bko.gaps.observe.dataset_fragment`), and every `is_a` S-node at probability 1, so
what a sampler can draw is each concept's presence. `shares` reads a concept's share from the two
priors; `sample` draws each concept present at its share, records the drawn set's closure upward
under `is_a`, and reads field values back from training records that show exactly the drawn
concepts of each field group. Draws come from one `numpy.random.default_rng(seed)`, consumed in
quota, record and column order, so `bko.bkb` stays standard-library only. ADR
`research/bkb-direct-generation` records each decision and its reasons.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from bko.bkb.structure import BKB
from bko.datasets.cells import SEXES, UNDERREPRESENTED_BELOW, cell_keys
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.gaps.observe import (
    Cell,
    CellRecords,
    Observed,
    cell_key,
    dataset_fragment,
    observe_records,
    record_cells,
)
from bko.generate.errors import SamplingError
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology, ancestors


logger = logging.getLogger(__name__)

GROUP_WEIGHT: Final = UNDERREPRESENTED_BELOW
"""The label group's weight in a share, in records: a cell this small weighs as its group does."""

type Record = Mapping[str, int | float]


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
    """

    cell: Cell
    record: Record
    concepts: frozenset[str]
    closed: frozenset[str]
    fallback: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", MappingProxyType(dict(self.record)))


@dataclass(frozen=True, slots=True)
class FieldGroup:
    """Fields whose values are drawn together, from one record, and the concepts they show.

    Attributes:
        fields: The fields, in column order.
        concepts: The concepts whose rows read them, sorted.
    """

    fields: tuple[str, ...]
    concepts: tuple[str, ...]


def shares(cell: CellRecords, group: Observed) -> dict[str, float]:
    """Each mapped concept's share: `(n_c p_c + GROUP_WEIGHT p_g) / (n_c + GROUP_WEIGHT)`.

    `p_c` is the cell's dataset-fragment prior and `p_g` its label group's, each 0 where the
    fragment holds no S-node for the concept; `n_c` is the cell's record count.

    Args:
        cell: The cell's records.
        group: The records of the cell's label, observed over the same rows.

    Returns:
        The share of every concept `group` counts, in its sorted order.
    """
    mine = _priors(dataset_fragment(cell.observed))
    theirs = _priors(dataset_fragment(group))
    records = cell.observed.records
    return {
        concept: (records * mine.get(concept, 0.0) + GROUP_WEIGHT * theirs.get(concept, 0.0))
        / (records + GROUP_WEIGHT)
        for concept in group.holders
    }


def label_only(rows: Sequence[FieldMapping], *, frame: FrameSpec = UCI_FRAME) -> frozenset[str]:
    """The concepts every row of which reads only the label: they follow the cell's label."""
    return frozenset(
        concept
        for concept, mapped in _by_concept(rows).items()
        if all(clause.field == frame.label for row in mapped for clause in row.when)
    )


def nested(rows: Sequence[FieldMapping]) -> dict[str, str]:
    """Each concept whose one row's codes lie strictly inside another's, on one field, to it.

    On UCI, `HP:0001712` (`restecg` 2) to `HP:0003115` (`restecg` 1 or 2): a record showing the
    first shows the second, so the second is drawn given the first. Only one level is drawn.

    Raises:
        ValueError: If a concept lies inside more than one other, so a chain of nesting, or holds
            more than one.
    """
    single = {
        concept: mapped[0].when[0]
        for concept, mapped in _by_concept(rows).items()
        if len(mapped) == 1 and len(mapped[0].when) == 1 and isinstance(mapped[0].when[0], Codes)
    }
    pairs = [
        (inner, outer)
        for inner, small in single.items()
        for outer, large in single.items()
        if isinstance(small, Codes)
        and isinstance(large, Codes)
        and small.field == large.field
        and small.codes < large.codes
    ]
    for side, word in ((0, "lies inside"), (1, "holds")):
        counts = Counter(pair[side] for pair in pairs)
        for concept, times in sorted(counts.items()):
            if times > 1:
                msg = f"{concept} {word} {times} concepts; only one level of nesting is drawn"
                raise ValueError(msg)
    return dict(pairs)


def field_groups(
    rows: Sequence[FieldMapping], *, frame: FrameSpec = UCI_FRAME
) -> tuple[FieldGroup, ...]:
    """The fields the rows read, joined when one concept's rows read both, in column order.

    The label is no field of a group: label-only rows follow the cell.
    """
    parent: dict[str, str] = {}

    def root(field: str) -> str:
        while parent[field] != field:
            field = parent[field]
        return field

    reads: dict[str, set[str]] = {}
    for concept, mapped in _by_concept(rows).items():
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
    frame: FrameSpec = UCI_FRAME,
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
    records: list[dict[str, float]]
    bands: list[str]


@dataclass(frozen=True, slots=True, eq=False)
class _Sources:
    """What one cell's draws read: its shares and the records each field may come from."""

    cell: Cell
    share: Mapping[str, float]
    own: list[dict[str, float]]
    group: list[dict[str, float]]
    banded: list[dict[str, float]]


class _Plan:
    """What every draw reads: the concept order, the nesting, the groups and the free fields."""

    def __init__(
        self, rows: Sequence[FieldMapping], ontologies: Mapping[str, Ontology], frame: FrameSpec
    ) -> None:
        self._frame = frame
        self._by_concept = _by_concept(rows)
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
        cell = sources.cell
        for column in self._frame.columns:
            if column == "age":
                chosen, fell = _pick(cell, (column,), sources.own, sources.banded, rng)
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
                own = [r for r in sources.own if shows(r)]
                chosen, fell = _pick(cell, fields, own, [r for r in sources.group if shows(r)], rng)
            elif column in self._free:
                fields = (column,)
                chosen, fell = _pick(cell, fields, sources.own, sources.group, rng)
            else:
                continue
            for field in fields:
                record[field] = self._value(field, chosen)
            if fell:
                fallback.update(fields)
        ordered = {column: record[column] for column in self._frame.columns}
        concepts = frozenset(present)
        return Draw(cell, ordered, concepts, self._closure(concepts), frozenset(fallback))

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
    own: Sequence[dict[str, float]],
    other: Sequence[dict[str, float]],
    rng: np.random.Generator,
) -> tuple[dict[str, float], bool]:
    """A record drawn from `own`, else from `other`, and whether it fell back to `other`.

    Raises:
        SamplingError: If both are empty.
    """
    if own:
        return own[int(rng.integers(len(own)))], False
    if other:
        return other[int(rng.integers(len(other)))], True
    msg = f"{cell_key(cell)} cannot draw {'+'.join(fields)}: no training record matches the draw"
    raise SamplingError(msg)


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
        observed = observe_records(members, rows, name=f"{name}/{frame.label}={label}")
        bands = [str(band) for band in keys.loc[members.index, keys.columns[0]]]
        groups[label] = _Group(observed, _records(members, frame), bands)
    return groups


def _records(frame: pd.DataFrame, spec: FrameSpec) -> list[dict[str, float]]:
    columns = [str(column) for column in spec.columns]
    return [
        dict(zip(columns, values, strict=True))
        for values in frame[columns].itertuples(index=False, name=None)
    ]


def _priors(fragment: BKB) -> dict[str, float]:
    return {snode.head[0]: snode.prob for snode in fragment.snodes}


def _by_concept(rows: Sequence[FieldMapping]) -> dict[str, list[FieldMapping]]:
    by_concept: dict[str, list[FieldMapping]] = {}
    for row in rows:
        by_concept.setdefault(row.concept, []).append(row)
    return by_concept
