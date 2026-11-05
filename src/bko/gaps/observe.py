"""A dataset as BKO fragments: one prior per mapped concept its records show, at its share.

The fragment is about the ontology fragments' one individual, `INDIVIDUAL`, read as a record drawn
at random, so a concept's prior is the share of records that show it. A concept no record shows
gets no S-node, since instantiation reads structure and a prior of 0 would still ground it. A
dataset is observed whole, for population-level grounding, and in each cell of its grid, for the
per-subgroup and frequency-aware definitions of ADR `research/bko-formal-gap-definition`; a cell no
record falls in is observed with no record. ADR `research/dataset-fragment-encoding` records the
encoding.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from bko.bkb.structure import BKB
from bko.datasets.cells import (
    CELL_COLUMNS,
    LABEL,
    LABEL_CODES,
    cell_counts,
    cell_keys,
)
from bko.datasets.synthea import SYNTHEA_AGE_BAND_EDGES, Population, subjects
from bko.gaps.errors import GapError
from bko.ontology.fragments import INDIVIDUAL
from bko.ontology.schema import CodeMapping, FieldMapping, shown


type Cell = tuple[str | int, ...]
"""A cell's key, its columns in grid order: `("<55", "female", 1)` on UCI Heart Disease and
`("60-69", "male")` on Synthea, whose cells carry no label."""


@dataclass(frozen=True, slots=True, eq=False)
class Observed:
    """How many of a dataset's records show each concept its rows map.

    Attributes:
        name: The dataset's source name, which its fragment and fusion source take.
        records: How many records were read; 0 for a cell no record falls in.
        holders: Each mapped concept, in sorted order, with how many records show it; a read-only
            copy of the mapping given.
    """

    name: str
    records: int
    holders: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "holders", MappingProxyType(dict(self.holders)))

    @property
    def held(self) -> tuple[str, ...]:
        """The concepts at least one record shows, in sorted order."""
        return tuple(concept for concept, count in self.holders.items() if count > 0)


@dataclass(frozen=True, slots=True, eq=False)
class CellRecords:
    """One cell's records, observed as a dataset of their own.

    Attributes:
        observed: How many of the cell's records show each concept.
        label: The cell's label, or None where cells carry no label.
        assumed: The concepts the cell treats as held though no record of it shows them: those only
            another label's mapping rows name (ADR `research/bko-formal-gap-definition`).
    """

    observed: Observed
    label: int | None
    assumed: frozenset[str]


@dataclass(frozen=True, slots=True, eq=False)
class Cells:
    """A dataset observed whole and in each cell of its grid: what every gap definition reads.

    Attributes:
        whole: Every record, observed as one dataset.
        each: Each cell's records in grid order, a cell no record falls in included; a read-only
            copy of the mapping given.
    """

    whole: Observed
    each: Mapping[Cell, CellRecords]

    def __post_init__(self) -> None:
        object.__setattr__(self, "each", MappingProxyType(dict(self.each)))


def cell_key(cell: Cell) -> str:
    """The cell's key as text, its parts joined by `/`: what its fragment and log lines name."""
    return "/".join(str(part) for part in cell)


def observe_records(frame: pd.DataFrame, rows: Iterable[FieldMapping], *, name: str) -> Observed:
    """Count the records of `frame` that show each concept `rows` map.

    A record shows a concept when any row mapping it applies, so a record two such rows map counts
    once.

    Args:
        frame: The records: the training split, never the test split (ADR
            `research/uci-evaluation-frame`).
        rows: The field mappings.
        name: The dataset's source name.

    Raises:
        GapError: If `frame` holds no record, or a row reads a field it lacks.
    """
    return _whole(name, record_shows(frame, rows, name=name))


def record_cells(
    frame: pd.DataFrame,
    rows: Iterable[FieldMapping],
    *,
    name: str,
    edges: Sequence[int],
) -> Cells:
    """Observe the records of `frame` whole and in each age-sex-label cell.

    A cell assumes the concepts only another label's rows name: every row naming one reads the
    label and admits no record of the cell's label (ADR `research/bko-formal-gap-definition`).
    Records and their cells are paired by position, so any index will do.

    Args:
        frame: The records, with `age`, `sex` and `target` beside the fields the rows read: the
            training split, never the test split (ADR `research/uci-evaluation-frame`).
        rows: The field mappings.
        name: The dataset's source name; a cell's fragment takes it followed by the cell's key.
        edges: Age band starts.

    Raises:
        GapError: As `observe_records`, or if `frame` lacks a cell column.
        ContractError: As `bko.datasets.cells.cell_keys`, for a value outside its codes.
    """
    missing = sorted({"age", "sex", LABEL} - {str(column) for column in frame.columns})
    if missing:
        msg = f"the cells read fields {name} lacks: {', '.join(missing)}"
        raise GapError(msg)
    mappings = tuple(rows)
    shows = record_shows(frame, mappings, name=name)
    whole = _whole(name, shows)
    keys = cell_keys(frame, edges=edges)
    assumed = {label: _only_other_labels(mappings, label) for label in sorted(LABEL_CODES)}
    each: dict[Cell, CellRecords] = {
        cell: CellRecords(_member(name, shows, keys, cell), cell[2], assumed[cell[2]])
        for cell in _grid(frame, edges)
    }
    return Cells(whole=whole, each=each)


def observe_population(
    population: Population, rows: Iterable[CodeMapping], *, name: str
) -> Observed:
    """Count the patients of `population` who show each concept `rows` map.

    A patient shows a concept when they had one of its codes by the snapshot date: the case rule of
    `bko.datasets.synthea.subjects`, so every detector reads the same cases.

    Raises:
        GapError: If `population` holds no patient.
    """
    return _whole(name, _population_shows(population, rows))


def population_cells(
    population: Population,
    rows: Iterable[CodeMapping],
    *,
    name: str,
    edges: Sequence[int] = SYNTHEA_AGE_BAND_EDGES,
) -> Cells:
    """Observe the patients of `population` whole and in each age-sex cell.

    Synthea's cells carry no label, since E2's label is the ablated condition (ADR
    `research/bko-formal-gap-definition`), so no cell assumes a concept.

    Args:
        population: The population.
        rows: The code mappings.
        name: The dataset's source name; a cell's fragment takes it followed by the cell's key.
        edges: Age band starts; E2 passes its ablation's, so every detector reads one grid.

    Raises:
        GapError: If `population` holds no patient.
    """
    shows = _population_shows(population, rows)
    whole = _whole(name, shows)
    people = subjects(population, codes=frozenset())
    keys = cell_keys(people, edges=edges)[list(CELL_COLUMNS[:2])]
    grid = dict.fromkeys((band, sex) for band, sex, _ in _grid(people, edges))
    each: dict[Cell, CellRecords] = {
        cell: CellRecords(_member(name, shows, keys, cell), None, frozenset()) for cell in grid
    }
    return Cells(whole=whole, each=each)


def dataset_fragment(observed: Observed, *, held: Collection[str] | None = None) -> BKB:
    """The fragment of `observed`: P(concept) = its share of the records, per held concept.

    Args:
        observed: The records' holders.
        held: The concepts given a prior, of those a record shows; every one when omitted. A
            concept a record shows that `held` leaves out gets no S-node, since a prior grounds its
            concept whatever the anchor.

    Returns:
        One I-node and one prior S-node per concept given a prior, in sorted order, named
        `observed.name`.
    """
    fragment = BKB(observed.name)
    for concept in observed.held:
        if held is None or concept in held:
            fragment.add_inode(concept, INDIVIDUAL)
            fragment.add_snode(concept, INDIVIDUAL, observed.holders[concept] / observed.records)
    return fragment


def _observed(name: str, records: int, holders: dict[str, int]) -> Observed:
    if records == 0:
        msg = f"{name} holds no record, so no concept has a share"
        raise GapError(msg)
    return Observed(name=name, records=records, holders=MappingProxyType(holders))


def record_shows(frame: pd.DataFrame, rows: Iterable[FieldMapping], *, name: str) -> pd.DataFrame:
    """Whether each record of `frame`, by position, shows each concept `rows` map, in sorted order.

    A record shows a concept when any row mapping it applies (`bko.ontology.schema.shown`).

    Raises:
        GapError: If a row reads a field `frame` lacks.
    """
    mappings = tuple(rows)
    fields = [str(column) for column in frame.columns]
    missing = sorted({clause.field for row in mappings for clause in row.when} - set(fields))
    if missing:
        msg = f"the rows read fields {name} lacks: {', '.join(missing)}"
        raise GapError(msg)
    records = [
        dict(zip(fields, values, strict=True))
        for values in frame.itertuples(index=False, name=None)
    ]
    shows = [shown(record, mappings) for record in records]
    columns = {
        concept: [concept in one for one in shows] for concept in sorted(by_concept(mappings))
    }
    return pd.DataFrame(columns, index=pd.RangeIndex(len(records)), dtype=bool)


def _population_shows(population: Population, rows: Iterable[CodeMapping]) -> pd.DataFrame:
    """Whether each patient, by position, shows each concept `rows` map, by `subjects`' rule."""
    by_concept: dict[str, set[str]] = {}
    for row in rows:
        by_concept.setdefault(row.concept, set()).add(row.snomed)
    columns = {
        concept: subjects(population, codes=frozenset(codes))[LABEL].to_numpy(dtype=bool)
        for concept, codes in sorted(by_concept.items())
    }
    return pd.DataFrame(columns, index=pd.RangeIndex(len(population.patients)), dtype=bool)


def _whole(name: str, shows: pd.DataFrame) -> Observed:
    holders = {str(concept): int(shows[concept].sum()) for concept in shows.columns}
    return _observed(name, len(shows), holders)


def _member(name: str, shows: pd.DataFrame, keys: pd.DataFrame, cell: Cell) -> Observed:
    """The records `keys` puts in `cell`, observed; `keys` and `shows` align by position."""
    inside = np.ones(len(keys), dtype=bool)
    for column, value in zip(keys.columns, cell, strict=True):
        inside &= keys[column].eq(value).to_numpy(dtype=bool)
    members = shows.loc[inside]
    holders = {str(concept): int(members[concept].sum()) for concept in shows.columns}
    return Observed(name=f"{name}/{cell_key(cell)}", records=int(inside.sum()), holders=holders)


def _grid(people: pd.DataFrame, edges: Sequence[int]) -> tuple[tuple[str, str, int], ...]:
    """Every age-sex-label cell over `edges`, in `cell_counts`' grid order."""
    counts = cell_counts(people, edges=edges)[list(CELL_COLUMNS)]
    return tuple(
        (str(band), str(sex), int(label))
        for band, sex, label in counts.itertuples(index=False, name=None)
    )


def by_concept(rows: Iterable[FieldMapping]) -> dict[str, list[FieldMapping]]:
    """`rows` grouped by concept, in first-seen order."""
    by_concept: dict[str, list[FieldMapping]] = {}
    for row in rows:
        by_concept.setdefault(row.concept, []).append(row)
    return by_concept


def _only_other_labels(rows: Sequence[FieldMapping], label: int) -> frozenset[str]:
    """The concepts every row naming which reads the label and admits no record of `label`."""
    return frozenset(
        concept
        for concept, mapped in by_concept(rows).items()
        if all(
            any(clause.field == LABEL and not clause.holds(label) for clause in row.when)
            for row in mapped
        )
    )
