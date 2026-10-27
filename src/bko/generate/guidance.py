"""What a request tells the model about one cell, as values the prompt renders.

Every level carries the cell, its training record count and each numeric field's plausible
interval, with age clipped to the cell's band. Higher levels add ontology findings, a profile of
concept shares and numeric spreads. The numeric fields, their ranges and the band edges are the
frame's, UCI's by default. ADR `research/record-generation-design` records the ladder.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from bko.bkb.structure import BKB
from bko.datasets.cells import LABEL, SEXES, age_band_labels, cell_keys
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.gaps.definitions import Definition, StatementGap
from bko.gaps.detect import Statement, grounded
from bko.gaps.observe import Cell, Cells, cell_key, dataset_fragment
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology, ancestors


logger = logging.getLogger(__name__)


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


def intervals(
    cell: Cell, edges: Sequence[int] | None = None, *, frame: FrameSpec | None = None
) -> dict[str, tuple[float, float]]:
    """Return each numeric field's plausible interval for `cell`, age clipped to its band.

    Args:
        cell: The cell's key; its first item is its age band's label.
        edges: Age band starts; band i covers `edges[i-1]` to `edges[i] - 1`. The frame's when
            None.
        frame: The frame whose numeric fields, ranges and band edges are read; `UCI_FRAME`
            when None.

    Returns:
        Each of the frame's numeric fields' range bounds, both inclusive.

    Raises:
        ValueError: If both `edges` and `frame` are given and the edges differ from the frame's,
            or the cell's band is not one of `edges`' bands.
    """
    if edges is not None and frame is not None and tuple(edges) != frame.edges:
        msg = f"edges {tuple(edges)} differ from frame {frame.name}'s {frame.edges}"
        raise ValueError(msg)
    frame = UCI_FRAME if frame is None else frame
    edges = frame.edges if edges is None else edges
    labels = age_band_labels(edges)
    if cell[0] not in labels:
        msg = f"age band {cell[0]!r} is not one of {labels}"
        raise ValueError(msg)
    band = labels.index(str(cell[0]))
    bounds = {bound.field: (bound.low, bound.high) for bound in frame.ranges}
    low, high = bounds["age"]
    if band > 0:
        low = max(low, edges[band - 1])
    if band < len(edges):
        high = min(high, edges[band] - 1)
    bounds["age"] = (low, high)
    return {name: bounds[name] for name in frame.numeric}


def guide(
    train: pd.DataFrame,
    cells: Cells,
    found: Iterable[StatementGap],
    *,
    definition: Definition,
    fragments: Sequence[BKB],
    ontologies: Mapping[str, Ontology],
    rows: Sequence[FieldMapping] = FIELD_MAPPINGS,
    frame: FrameSpec = UCI_FRAME,
) -> dict[Cell, Guidance]:
    """Each cell's guidance: its gaps under `definition`, else what its records ground, else none.

    A statement maps to the rows of `rows` that can ground it: those whose concept is its
    antecedent or descends from it, and that admit the cell's label. Statements reached by the
    same rows form one finding; a statement no such row reaches is dropped with a WARNING. A
    statement a row reading only the label reaches is skipped, since the schema pins the label and
    every record meets that row.

    The profile and summary read the cell's reference group: the training records outside the
    cell that share its label (ADR `research/frequency-aware-gap-score`). A cell with none gets
    neither, with a WARNING.

    Args:
        train: The training split `cells` was observed from.
        cells: The training split, observed whole and per cell.
        found: The dataset's gaps under `definition`, in any order.
        definition: The definition `found` was read under; frequency-aware gaps are rare, the
            others absent.
        fragments: The ontology fragments the gaps were read against.
        ontologies: The loaded release for each concept prefix.
        rows: The field mappings.
        frame: The frame whose numeric fields, ranges and band edges are read.

    Returns:
        One guidance per cell, in grid order.

    Raises:
        ValueError: If a gap's cell is not in `cells`, a cell carries no label, a row's prefix
            has no release, or `train` holds a cell's reference group with another record count
            than `cells` does, so it is not the frame `cells` was observed from.
        ContractError: As `bko.datasets.cells.cell_keys`, for a `train` value outside its codes.
    """
    by_cell: dict[Cell, list[Statement]] = {}
    for gap in found:
        if gap.cell not in cells.each:
            msg = f"gap cell {gap.cell!r} is not in the grid"
            raise ValueError(msg)
        by_cell.setdefault(gap.cell, []).append(gap.statement)
    reach = [_reach(row, ontologies) for row in rows]
    sexes = {meaning: code for code, meaning in SEXES.items()}
    basis = Basis.RARE if definition is Definition.FREQUENCY else Basis.ABSENT
    keys = cell_keys(train, edges=frame.edges)
    guided: dict[Cell, Guidance] = {}
    for cell, records in cells.each.items():
        if records.label is None:
            msg = f"cell {cell!r} carries no label"
            raise ValueError(msg)
        if cell in by_cell:
            why, chosen = basis, set(by_cell[cell])
        elif records.observed.held:
            why, chosen = Basis.SHOWN, grounded(fragments, dataset_fragment(records.observed))
        else:
            why, chosen = Basis.NONE, ()
        profile, summary = _reference(train, keys, cells, cell, records.label, rows, frame)
        guided[cell] = Guidance(
            cell=cell,
            sex=sexes[str(cell[1])],
            label=records.label,
            records=records.observed.records,
            interval=intervals(cell, frame=frame),
            basis=why,
            findings=_findings(cell, records.label, chosen, rows, reach, ontologies),
            profile=profile,
            summary=summary,
        )
    return guided


def _reference(
    train: pd.DataFrame,
    keys: pd.DataFrame,
    cells: Cells,
    cell: Cell,
    label: int,
    rows: Sequence[FieldMapping],
    frame: FrameSpec,
) -> tuple[Profile | None, dict[str, Spread] | None]:
    """The profile and summary of the records outside `cell` that share its label.

    The profile leaves out a concept `cells` does not count (one only a row outside the rows
    `cells` was observed with maps), and a concept a row reading only the label maps.
    """
    others = [
        other.observed for key, other in cells.each.items() if key != cell and other.label == label
    ]
    total = sum(observed.records for observed in others)
    if total == 0:
        logger.warning("noreference cell=%s", cell_key(cell))
        return None, None
    by_concept: dict[str, list[FieldMapping]] = {}
    for row in rows:
        if _admits(row, label):
            by_concept.setdefault(row.concept, []).append(row)
    shares = tuple(
        Share(
            concept=concept,
            label=mapped[0].label,
            states=tuple(mapped),
            share=sum(observed.holders[concept] for observed in others) / total,
        )
        for concept, mapped in by_concept.items()
        if concept in cells.whole.holders and not any(_label_only(row) for row in mapped)
    )
    inside = keys[LABEL].eq(label) & ~(keys["age_band"].eq(cell[0]) & keys["sex"].eq(cell[1]))
    reference = train.loc[inside.to_numpy(dtype=bool)]
    if len(reference) != total:
        msg = (
            f"cell {cell_key(cell)}: train holds {len(reference)} reference rows but cells hold "
            f"{total} reference records"
        )
        raise ValueError(msg)
    summary: dict[str, Spread] = {}
    for name in frame.numeric:
        low, median, high = (float(q) for q in reference[name].quantile([0.25, 0.5, 0.75]))
        summary[name] = Spread(low, median, high)
    return Profile(records=total, shares=shares), summary


def _reach(row: FieldMapping, ontologies: Mapping[str, Ontology]) -> frozenset[str]:
    """The classes a record meeting `row` shows: its concept and the concept's ancestors."""
    prefix = row.concept.partition(":")[0]
    if prefix not in ontologies:
        msg = f"row {row.concept} has no release for prefix {prefix}"
        raise ValueError(msg)
    return frozenset({row.concept}) | ancestors(ontologies[prefix], row.concept, prefix=prefix)


def _admits(row: FieldMapping, label: int) -> bool:
    return all(
        clause.holds(label)
        for clause in row.when
        if isinstance(clause, Codes) and clause.field == LABEL
    )


def _label_only(row: FieldMapping) -> bool:
    """Whether `row` reads only the label, which the schema pins, so every record meets it."""
    return all(isinstance(clause, Codes) and clause.field == LABEL for clause in row.when)


def _findings(
    cell: Cell,
    label: int,
    chosen: Collection[Statement],
    rows: Sequence[FieldMapping],
    reach: Sequence[frozenset[str]],
    ontologies: Mapping[str, Ontology],
) -> tuple[Finding, ...]:
    """Group `chosen` by the rows admitting `label` that reach each, in `rows` order."""
    grouped: dict[tuple[int, ...], list[Statement]] = {}
    for statement in sorted(chosen):
        reached = tuple(
            index
            for index, row in enumerate(rows)
            if statement.antecedent in reach[index] and _admits(row, label)
        )
        if not reached:
            logger.warning("unreached cell=%s statement=%s", cell_key(cell), statement)
            continue
        if any(_label_only(rows[index]) for index in reached):
            logger.debug("pinned cell=%s statement=%s", cell_key(cell), statement)
            continue
        grouped.setdefault(reached, []).append(statement)
    return tuple(
        Finding(
            states=tuple(rows[index] for index in reached),
            relationships=tuple(
                Relationship(
                    statement,
                    _label(statement.antecedent, ontologies),
                    _label(statement.consequent, ontologies),
                )
                for statement in statements
            ),
        )
        for reached, statements in sorted(grouped.items())
    )


def _label(curie: str, ontologies: Mapping[str, Ontology]) -> str:
    """The class's name in its release, or its curie when the release has none."""
    ontology = ontologies.get(curie.partition(":")[0])
    term = ontology.terms.get(curie) if ontology is not None else None
    return term.name if term is not None and term.name else curie
