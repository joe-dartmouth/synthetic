"""What a request tells the model about one cell, as values the prompt renders.

Every level carries the cell, its training record count and each numeric field's plausible
interval, with age clipped to the cell's band. Higher levels add ontology findings, a profile of
concept shares and numeric spreads. ADR `research/record-generation-design` records the ladder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from bko.datasets.cells import AGE_BAND_EDGES, age_band_labels
from bko.datasets.uci import NUMERIC
from bko.gaps.detect import Statement
from bko.gaps.observe import Cell
from bko.ontology.mapping import RANGES
from bko.ontology.schema import FieldMapping


class Basis(StrEnum):
    """Why the findings of a cell were chosen, which sets their wording."""

    NONE = "none"
    ABSENT = "absent"
    RARE = "rare"
    SHOWN = "shown"


@dataclass(frozen=True, slots=True)
class Relationship:
    """One ontology statement, with its classes' labels.

    Attributes:
        statement: The statement.
        antecedent: The antecedent class's label.
        consequent: The consequent class's label.
    """

    statement: Statement
    antecedent: str
    consequent: str


@dataclass(frozen=True, slots=True)
class Finding:
    """The statements that a group of field states reaches.

    Attributes:
        states: The mapping rows whose concept is each statement's antecedent or descends from it.
        relationships: The statements.
    """

    states: tuple[FieldMapping, ...]
    relationships: tuple[Relationship, ...]


@dataclass(frozen=True, slots=True)
class Share:
    """The share of reference records that show a concept.

    Attributes:
        concept: The concept's curie.
        label: The concept's label.
        states: The mapping rows that show it.
        share: The share of reference records showing it.
    """

    concept: str
    label: str
    states: tuple[FieldMapping, ...]
    share: float


@dataclass(frozen=True, slots=True)
class Profile:
    """The concept shares of a cell's reference group.

    Attributes:
        records: The reference group's record count.
        shares: One share per concept.
    """

    records: int
    shares: tuple[Share, ...]


@dataclass(frozen=True, slots=True)
class Spread:
    """A numeric field's spread over the reference group.

    Attributes:
        low: The low quantile.
        median: The median.
        high: The high quantile.
    """

    low: float
    median: float
    high: float


@dataclass(frozen=True, slots=True, eq=False)
class Guidance:
    """Everything a request says about one cell.

    Attributes:
        cell: The cell's key.
        sex: The cell's `sex` code.
        label: The cell's `LABEL` code.
        records: The training records the cell holds.
        interval: Each numeric field's plausible interval, age clipped to the cell's band.
        basis: Why the findings were chosen.
        findings: The ontology findings.
        profile: The reference group's concept shares.
        summary: The reference group's numeric spreads.
    """

    cell: Cell
    sex: int
    label: int
    records: int
    interval: Mapping[str, tuple[float, float]]
    basis: Basis = Basis.NONE
    findings: tuple[Finding, ...] = ()
    profile: Profile | None = None
    summary: Mapping[str, Spread] | None = None


def intervals(cell: Cell, edges: Sequence[int] = AGE_BAND_EDGES) -> dict[str, tuple[float, float]]:
    """Return each numeric field's plausible interval for `cell`, age clipped to its band.

    Args:
        cell: The cell's key; its first item is its age band's label.
        edges: Age band starts; band i covers `edges[i-1]` to `edges[i] - 1`.

    Returns:
        Each `NUMERIC` field's `RANGES` bounds, both inclusive.

    Raises:
        ValueError: If the cell's band is not one of `edges`' bands.
    """
    labels = age_band_labels(edges)
    if cell[0] not in labels:
        msg = f"age band {cell[0]!r} is not one of {labels}"
        raise ValueError(msg)
    band = labels.index(str(cell[0]))
    bounds = {bound.field: (bound.low, bound.high) for bound in RANGES}
    low, high = bounds["age"]
    if band > 0:
        low = max(low, edges[band - 1])
    if band < len(edges):
        high = min(high, edges[band] - 1)
    bounds["age"] = (low, high)
    return {name: bounds[name] for name in NUMERIC}
