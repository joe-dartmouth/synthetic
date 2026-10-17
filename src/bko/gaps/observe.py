"""A dataset as a BKO fragment: one prior per mapped concept its records show, at its share.

The fragment is about the ontology fragments' one individual, `INDIVIDUAL`, read as a record drawn
at random, so a concept's prior is the share of records that show it. A concept no record shows
gets no S-node, since instantiation reads structure and a prior of 0 would still ground it. ADR
`research/dataset-fragment-encoding` records the encoding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from bko.bkb.structure import BKB
from bko.datasets.cells import LABEL
from bko.datasets.synthea import Population, subjects
from bko.gaps.errors import GapError
from bko.ontology.fragments import INDIVIDUAL
from bko.ontology.schema import CodeMapping, FieldMapping


@dataclass(frozen=True, slots=True, eq=False)
class Observed:
    """How many of a dataset's records show each concept its rows map.

    Attributes:
        name: The dataset's source name, which its fragment and fusion source take.
        records: How many records were read.
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
    by_concept: dict[str, list[FieldMapping]] = {}
    for row in mappings:
        by_concept.setdefault(row.concept, []).append(row)
    holders = {
        concept: sum(any(row.applies(record) for row in mapped) for record in records)
        for concept, mapped in sorted(by_concept.items())
    }
    return _observed(name, len(records), holders)


def observe_population(
    population: Population, rows: Iterable[CodeMapping], *, name: str
) -> Observed:
    """Count the patients of `population` who show each concept `rows` map.

    A patient shows a concept when they had one of its codes by the snapshot date: the case rule of
    `bko.datasets.synthea.subjects`, so both detectors read the same cases.

    Raises:
        GapError: If `population` holds no patient.
    """
    by_concept: dict[str, set[str]] = {}
    for row in rows:
        by_concept.setdefault(row.concept, set()).add(row.snomed)
    holders = {
        concept: int(subjects(population, codes=frozenset(codes))[LABEL].sum())
        for concept, codes in sorted(by_concept.items())
    }
    return _observed(name, len(population.patients), holders)


def dataset_fragment(observed: Observed) -> BKB:
    """The fragment of `observed`: P(concept) = its share of the records, per held concept.

    Returns:
        One I-node and one prior S-node per held concept, in sorted order, named `observed.name`.
    """
    fragment = BKB(observed.name)
    for concept in observed.held:
        fragment.add_inode(concept, INDIVIDUAL)
        fragment.add_snode(concept, INDIVIDUAL, observed.holders[concept] / observed.records)
    return fragment


def _observed(name: str, records: int, holders: dict[str, int]) -> Observed:
    if records == 0:
        msg = f"{name} holds no record, so no concept has a share"
        raise GapError(msg)
    return Observed(name=name, records=records, holders=MappingProxyType(holders))
