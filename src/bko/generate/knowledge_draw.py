"""Records drawn from K alone: `bko-kb`'s draw, which reads no training record but the age range.

K is the mapping table and the pinned releases its rows name. `kb_sample` draws each concept's
presence as `Presence.LOGIC` does, uniformly over the assignments each component of
`bko.generate.logic.components` allows, then adds what the drawn and label-only concepts imply.
Each field group's codes are uniform over the codebook tuples, in `itertools.product` order of
each field's sorted codes, showing exactly the group's drawn concepts; `age` is uniform over the
cell's band's integers inside the training split's age range; `sex` and the label follow the
cell. Every draw has its own generator: component k's is
`np.random.default_rng([seed, cell_ordinal, record, COMPONENT_WORD, k])` and a column's
`np.random.default_rng([seed, cell_ordinal, record, COLUMN_WORD, column_index])`, a group's
column its first field's; the distinct prefix words keep the two families apart however many
components or columns there are, so a change to one component's rows leaves every other
component's draw as it was. A frame with a free field, or a clause on a numeric field or on
`sex`, has a value K cannot give, so `kb_check` refuses it. ADR
`research/matched-knowledge-controls`, Decision 9, records the design.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from bko.datasets.cells import SEXES, age_band_labels
from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell, by_concept, cell_key
from bko.gaps.shares import label_only, nested
from bko.generate.draws import Draw
from bko.generate.errors import SamplingError
from bko.generate.fields import FieldGroup, field_groups, free_fields
from bko.generate.logic import components, logic_assignment
from bko.ontology.schema import AtLeast, FieldMapping
from bko.ontology.terms import Ontology, ancestors


logger = logging.getLogger(__name__)

COMPONENT_WORD: Final = 0
"""The seed word before a component's ordinal: `[seed, cell_ordinal, record, 0, k]`."""

COLUMN_WORD: Final = 1
"""The seed word before a column's index: `[seed, cell_ordinal, record, 1, column_index]`."""

type _Codes = tuple[int, ...]


def kb_check(rows: Sequence[FieldMapping], frame: FrameSpec) -> None:
    """Refuse a frame `kb_sample` cannot draw from `rows` alone.

    Raises:
        ValueError: Naming the field, if `frame` has a free field (`free_fields`), or a row of
            `rows` holds a clause on a numeric field or on `sex`.
    """
    free = free_fields(rows, frame=frame)
    if free:
        msg = f"frame {frame.name}: {free[0]} is a free field, which K gives no value"
        raise ValueError(msg)
    for row in rows:
        for clause in row.when:
            if isinstance(clause, AtLeast) or clause.field in {*frame.numeric, "sex"}:
                msg = f"frame {frame.name}: {row.concept} holds a numeric or cell clause on "
                msg += f"{clause.field}, which K gives no codebook tuple"
                raise ValueError(msg)


def kb_sample(
    quota: Mapping[Cell, int],
    *,
    rows: Sequence[FieldMapping],
    ontologies: Mapping[str, Ontology],
    frame: FrameSpec,
    seed: int,
    age_range: tuple[int, int],
    tag: tuple[int, str] | None = None,
) -> dict[Cell, tuple[Draw, ...]]:
    """Draw `quota` records per cell from K alone, in quota order.

    Logs one `direct holdout=%d fill=%s key=%s rows=%d records=0 group_records=0 donors=false
    presence=logic` line per cell, `tag`'s holdout and fill, `holdout=-1 fill=none` when None.

    Args:
        quota: Records per cell, in the order drawn; a cell's ordinal is its position here.
        rows: The field mappings.
        ontologies: Each loaded release by concept prefix, for the closure.
        frame: The frame the records belong to.
        seed: The first word of every generator's seed.
        age_range: The lowest and highest `age`, inclusive: the training split's observed range.
        tag: The holdout and the fill the draw serves.

    Returns:
        Each cell's draws, in quota order; no draw names a source or a fallback field.

    Raises:
        ValueError: As `kb_check`, `nested` and `bko.generate.logic.components`.
        SamplingError: If a group has no codebook tuple showing a drawn pattern, or a cell's band
            holds no integer inside `age_range`.
    """
    kb_check(rows, frame)
    plan = _Knowledge(tuple(rows), ontologies, frame)
    holdout, fill = (-1, "none") if tag is None else tag
    drawn: dict[Cell, tuple[Draw, ...]] = {}
    for ordinal, (cell, count) in enumerate(quota.items()):
        logger.info(
            "direct holdout=%d fill=%s key=%s rows=%d records=0 group_records=0 donors=false "
            "presence=logic",
            holdout,
            fill,
            cell_key(cell),
            count,
        )
        drawn[cell] = tuple(
            plan.draw(cell, (seed, ordinal, record), age_range) for record in range(count)
        )
    return drawn


class _Knowledge:
    """What every knowledge-only draw reads: the components, the groups and their tuples."""

    def __init__(
        self, rows: Sequence[FieldMapping], ontologies: Mapping[str, Ontology], frame: FrameSpec
    ) -> None:
        self._frame = frame
        self._ontologies = ontologies
        self._by_concept = by_concept(rows)
        self._label_only = label_only(rows, frame=frame)
        self._inner_of = {outer: inner for inner, outer in nested(rows).items()}
        self._drawn = frozenset(c for c in self._by_concept if c not in self._label_only)
        implies = [(inner, outer) for outer, inner in self._inner_of.items()]
        implies += [
            (c, a) for c in sorted(self._drawn) for a in self._closure(frozenset({c})) - {c}
        ]
        self._components = components(self._drawn, implies)
        self._first = {group.fields[0]: group for group in field_groups(rows, frame=frame)}
        self._bands = age_band_labels(frame.edges)
        self._showing: dict[tuple[str, int, frozenset[str]], list[_Codes]] = {}

    def draw(self, cell: Cell, stream: tuple[int, int, int], age_range: tuple[int, int]) -> Draw:
        """One record of `cell`, its generators seeded by `stream` and each draw's last word."""
        label = int(cell[2])
        present = {
            concept
            for concept in self._label_only
            if any(row.applies({self._frame.label: label}) for row in self._by_concept[concept])
        }
        for k, component in enumerate(self._components):
            rng = np.random.default_rng([*stream, COMPONENT_WORD, k])
            present |= logic_assignment(component, rng)
        present = self._implied(present)
        record: dict[str, int | float] = {}
        for index, column in enumerate(self._frame.columns):
            if column not in self._first and column not in {"age", "sex", self._frame.label}:
                continue
            rng = np.random.default_rng([*stream, COLUMN_WORD, index])
            if column == "age":
                record[column] = self._age(cell, age_range, rng)
            elif column == "sex":
                record[column] = next(code for code, sex in SEXES.items() if sex == cell[1])
            elif column == self._frame.label:
                record[column] = label
            else:
                group = self._first[column]
                options = self._tuples(group, label, frozenset(present))
                if not options:
                    msg = f"cell {cell_key(cell)}: no codebook tuple of {group.fields} shows "
                    msg += f"{sorted(present & set(group.concepts))}"
                    raise SamplingError(msg)
                chosen = options[int(rng.integers(len(options)))]
                record.update(zip(group.fields, chosen, strict=True))
        ordered = {column: record[column] for column in self._frame.columns}
        concepts = frozenset(present)
        return Draw(cell, ordered, concepts, self._closure(concepts), frozenset())

    def _age(self, cell: Cell, age_range: tuple[int, int], rng: np.random.Generator) -> int:
        """A uniform integer of `cell`'s band inside `age_range`.

        Raises:
            SamplingError: If the band and the range share no integer.
        """
        band, edges = self._bands.index(str(cell[0])), self._frame.edges
        low = max(age_range[0], edges[band - 1] if band > 0 else age_range[0])
        high = min(age_range[1], edges[band] - 1 if band < len(edges) else age_range[1])
        if low > high:
            msg = f"cell {cell_key(cell)}: its band holds no age inside {list(age_range)}"
            raise SamplingError(msg)
        return int(rng.integers(low, high + 1))

    def _tuples(self, group: FieldGroup, label: int, present: frozenset[str]) -> list[_Codes]:
        """The codebook tuples of `group`'s fields showing exactly its concepts in `present`."""
        pattern = present & frozenset(group.concepts)
        key = (group.fields[0], label, pattern)
        if key not in self._showing:
            books = [sorted(self._frame.codebook[field]) for field in group.fields]
            self._showing[key] = [
                codes
                for codes in itertools.product(*books)
                if self._shown(group, codes, label) == pattern
            ]
        return self._showing[key]

    def _shown(self, group: FieldGroup, codes: _Codes, label: int) -> frozenset[str]:
        """The concepts of `group` a record holding `codes` in its fields, at `label`, shows."""
        record = {**dict(zip(group.fields, codes, strict=True)), self._frame.label: label}
        return frozenset(
            concept
            for concept in group.concepts
            if any(row.applies(record) for row in self._by_concept[concept])
        )

    def _implied(self, present: set[str]) -> set[str]:
        """`present` with what its `is_a` closure and nested pairs imply, until nothing joins."""
        while True:
            grown = (
                present
                | (self._closure(frozenset(present)) & self._drawn)
                | {outer for outer, inner in self._inner_of.items() if inner in present}
            )
            if grown == present:
                return present
            present = grown

    def _closure(self, concepts: frozenset[str]) -> frozenset[str]:
        closed = set(concepts)
        for concept in concepts:
            prefix = concept.partition(":")[0]
            ontology = self._ontologies.get(prefix)
            if ontology is not None:
                closed |= ancestors(ontology, concept, prefix=prefix)
        return frozenset(closed)
