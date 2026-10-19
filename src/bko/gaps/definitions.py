"""The three gap definitions: a dataset's (cell, statement) gaps under each.

ADR `research/bko-formal-gap-definition` defines them over the cells of `bko.gaps.observe.Cells`:
per-subgroup grounding, the primary one, grounds each cell's statements against its own records and
the concepts it assumes; population-level grounding grounds them against every record, its gaps
holding in every cell; the frequency-aware definition takes the statements a cell's
under-represented concepts ground by themselves, a concept being under-represented by the rule of
ADR `research/frequency-aware-gap-score`. Every definition grounds through
`bko.gaps.detect.ungrounded_from`, so they differ only in the concepts they hold. Nothing here
logs: a caller logs the `Flag`s it wants to show.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

from scipy.stats import beta as beta_distribution

from bko.bkb.structure import BKB
from bko.datasets.cells import UNDERREPRESENTED_BELOW
from bko.gaps.detect import Statement, ungrounded_from
from bko.gaps.errors import GapError
from bko.gaps.observe import Cell, Cells


SHARE_RATIO: Final = 0.8
"""A concept is under-represented credibly below this fraction of its reference share."""

CREDIBILITY: Final = 0.95
"""The posterior mass below `SHARE_RATIO` of the reference share that flags a concept."""

JEFFREYS: Final = 0.5
"""Each parameter of Jeffreys' Beta prior for a binomial share."""


class Definition(StrEnum):
    """A gap definition of ADR `research/bko-formal-gap-definition`, in report order."""

    SUBGROUP = "per-subgroup"
    """Each cell grounded against its own records: the primary definition."""
    POPULATION = "population-level"
    """Every record grounded together, its gaps holding in every cell: an ablation."""
    FREQUENCY = "frequency-aware"
    """The statements a cell's under-represented concepts ground by themselves: an ablation."""


@dataclass(frozen=True, slots=True)
class StatementGap:
    """A statement one cell's records leave ungrounded under a definition.

    Attributes:
        cell: The cell's key.
        statement: The statement.
    """

    cell: Cell
    statement: Statement


@dataclass(frozen=True, slots=True)
class Flag:
    """A concept a cell shows credibly under `SHARE_RATIO` of its reference share.

    Attributes:
        cell: The cell's key.
        concept: The concept.
        shown: How many of the cell's records show it.
        records: The cell's records.
        reference_shown: How many reference records show it.
        reference_records: The reference records: those outside the cell that share its label.
        mass: The posterior mass below `SHARE_RATIO` of the reference share.
    """

    cell: Cell
    concept: str
    shown: int
    records: int
    reference_shown: int
    reference_records: int
    mass: float


def posterior_below(shown: int, records: int, share: float) -> float:
    """The posterior mass of a cell's share below `share`, under Jeffreys' prior.

    With theta ~ Beta(shown + 1/2, records - shown + 1/2), this is P(theta < share).
    """
    return float(beta_distribution.cdf(share, shown + JEFFREYS, records - shown + JEFFREYS))


def under_represented(cells: Cells) -> tuple[Flag, ...]:
    """Each concept a cell shows credibly under `SHARE_RATIO` of its reference share.

    The reference is the records outside the cell that share its label, or every record outside
    it where cells carry no label. No reference record, or no reference record showing the
    concept, flags nothing.

    Returns:
        The flags, in grid order, each cell's in its concepts' order.
    """
    flags: list[Flag] = []
    for cell, records in cells.each.items():
        reference = [
            other.observed
            for key, other in cells.each.items()
            if key != cell and other.label == records.label
        ]
        outside = sum(observed.records for observed in reference)
        if outside == 0:
            continue
        for concept, shown in records.observed.holders.items():
            held = sum(observed.holders[concept] for observed in reference)
            if held == 0:
                continue
            mass = posterior_below(shown, records.observed.records, SHARE_RATIO * held / outside)
            if mass >= CREDIBILITY:
                flags.append(
                    Flag(cell, concept, shown, records.observed.records, held, outside, mass)
                )
    return tuple(flags)


def gaps(
    definition: Definition, fragments: Sequence[BKB], cells: Cells
) -> tuple[StatementGap, ...]:
    """A dataset's gaps under `definition`, in grid order, each cell's statements sorted.

    Args:
        definition: The definition.
        fragments: The ontology fragments over the concepts the dataset's rows map.
        cells: The dataset, observed whole and per cell.

    Raises:
        QueryError: If a cell holds or assumes a concept no fragment holds.
    """
    match definition:
        case Definition.SUBGROUP:
            return tuple(
                StatementGap(cell, statement)
                for cell, records in cells.each.items()
                for statement in ungrounded_from(
                    fragments, records.observed, {*records.observed.held, *records.assumed}
                )
            )
        case Definition.POPULATION:
            whole = ungrounded_from(fragments, cells.whole, cells.whole.held)
            return tuple(
                StatementGap(cell, statement) for cell in cells.each for statement in whole
            )
        case Definition.FREQUENCY:
            return _frequency(fragments, cells)
        case _:
            assert_never(definition)


def completeness_request(fragments: Sequence[BKB], cells: Cells) -> dict[Cell, int]:
    """The completeness request: `UNDERREPRESENTED_BELOW` rows for each cell holding a gap.

    ADR `research/bko-formal-gap-definition` (Decision, Completeness) defines it over the
    per-subgroup gaps: every cell holding one is asked for the threshold, not its shortfall. It
    reads the gaps itself, so no caller can pass another definition's.

    Args:
        fragments: The ontology fragments, as `gaps` takes them.
        cells: The training split's cells.

    Returns:
        Rows per cell, in grid order.

    Raises:
        GapError: If no cell holds a per-subgroup gap, so the request asks nothing.
        QueryError: As `gaps`.
    """
    holding = {gap.cell for gap in gaps(Definition.SUBGROUP, fragments, cells)}
    request = {cell: UNDERREPRESENTED_BELOW for cell in cells.each if cell in holding}
    if not request:
        msg = (
            f"{cells.whole.name} holds no per-subgroup gap, "
            "so the completeness request asks nothing"
        )
        raise GapError(msg)
    return request


def _frequency(fragments: Sequence[BKB], cells: Cells) -> tuple[StatementGap, ...]:
    flagged: dict[Cell, set[str]] = {}
    for flag in under_represented(cells):
        flagged.setdefault(flag.cell, set()).add(flag.concept)
    found: list[StatementGap] = []
    for cell, records in cells.each.items():
        under = flagged.get(cell)
        if not under:
            continue
        base = {*records.observed.held, *records.assumed}
        with_them = set(ungrounded_from(fragments, records.observed, base | under))
        without = ungrounded_from(fragments, records.observed, base - under)
        found.extend(
            StatementGap(cell, statement) for statement in without if statement not in with_them
        )
    return tuple(found)
