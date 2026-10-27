"""The first-flag scan: where each detector first flags a truth item as a whole ablation thins it.

A whole E2 ablation's cases are removed one at a time in its seed's draw order
(`bko.datasets.ablation.removal_order`), the order a fraction ablation takes its first cases from,
and every detector of `bko.gaps.experiment` reads the population after each removal. For each
truth item of the full removal, a (cell, statement) pair for a definition and an age-sex-label
cell for the count-based detector, the scan records the removed count from which the item is in
the truth, the removed count at which the detector first lists it (0 when it is listed before any
removal), and the cell's own removed cases then. Per-subgroup, population-level and count-based
listings only grow as cases go, so an item listed first at removed count `r` is a gap after the
fraction ablation that removes `ceil(f * n)` of the `n` cases exactly when `r` is at most that
count; the frequency-aware reference share moves as cases go, so that definition can unlist an
item, and each time it does is a reversal. Removed counts are counted, never turned into
fractions: `r / n` does not round back to `r` at every `n`. The rows and log lines are rendered
here, for `python -m bko.gaps first-flag`.
"""

from __future__ import annotations

import bisect
import itertools
import logging
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from bko.bkb.structure import BKB
from bko.datasets.ablation import AblationSpec, removal_order, without
from bko.datasets.cells import CELL_COLUMNS, cell_counts, cell_keys
from bko.datasets.synthea import Population, subjects
from bko.gaps.definitions import Definition, StatementGap, gaps
from bko.gaps.detect import count_gaps, grounded
from bko.gaps.errors import GapError
from bko.gaps.experiment import SYNTHEA, AblationRun, run_ablation
from bko.gaps.observe import Cell, cell_key, dataset_fragment, population_cells
from bko.gaps.report import STATEMENT_COLUMNS, Label, Row, cell_cells, statement_cells
from bko.ontology.schema import CodeMapping


logger = logging.getLogger(__name__)

type Item = StatementGap | tuple[str, str, int]
"""A truth item: a (cell, statement) pair for a definition, an age-sex-label cell for the count."""

COUNT: Final = "count"
"""The count-based detector's name in a row."""

FIRST_FLAG_COLUMNS: Final[Row] = (
    "ablation",
    "detector",
    *STATEMENT_COLUMNS,
    *CELL_COLUMNS,
    "cases",
    "truth_from",
    "first_flag",
    "cell_removed",
    "reversals",
    "flagged_at_all",
)


@dataclass(frozen=True, slots=True)
class FirstFlag:
    """Where one detector first flags one truth item of a whole ablation.

    Attributes:
        detector: A definition's value, or `count`.
        item: The truth item.
        cases: The ablation's cases in the item's cell.
        truth_from: The removed count from which the item is in the truth.
        first_flag: The removed count at which the detector first lists the item, 0 when it is
            listed before any removal; `None` when it never is.
        cell_removed: The cell's removed cases at `first_flag`; `None` when it never is listed.
        reversals: How many times the detector stops listing the item as cases go.
    """

    detector: str
    item: Item
    cases: int
    truth_from: int
    first_flag: int | None
    cell_removed: int | None
    reversals: int

    @property
    def flagged_at_all(self) -> bool:
        """Whether the detector lists the item at any removed count."""
        return self.first_flag is not None


def scan(
    name: str,
    population: Population,
    spec: AblationSpec,
    fragments: Sequence[BKB],
    rows: Sequence[CodeMapping],
) -> tuple[FirstFlag, ...]:
    """Remove `spec`'s cases one at a time and find where each detector first flags each item.

    Args:
        name: The ablation's name.
        population: The population before any removal.
        spec: A whole ablation: `fraction` 1, its `seed` ordering the removals.
        fragments: The ontology fragments over the concepts `rows` map.
        rows: The code mappings.

    Returns:
        One flag per detector and truth item: the definitions in report order, then `count`, each
        over the full removal's truth in its order.

    Raises:
        DatasetError: As `bko.datasets.ablation.ablate`.
        GapError: If `spec` removes a fraction below 1, or the full removal leaves no patient.
    """
    if spec.fraction != 1:
        msg = (
            f"the first-flag scan reads a whole ablation, at fraction 1; "
            f"{name} removes fraction {spec.fraction}"
        )
        raise GapError(msg)
    full = run_ablation(name, population, spec, fragments, rows)
    order = removal_order(full.ablation.removed, spec.seed)
    keys = cell_keys(subjects(population, codes=spec.codes), edges=spec.edges).loc[order]
    places: dict[Cell, list[int]] = {}
    for removed, (band, sex) in enumerate(keys[["age_band", "sex"]].itertuples(index=False), 1):
        places.setdefault((str(band), str(sex)), []).append(removed)
    listings = _listings(full, population, spec, fragments, rows, order)
    entered = _truth_from(full, population, spec, fragments, rows, order, places)
    detectors = [*(definition.value for definition in Definition), COUNT]
    flags: list[FirstFlag] = []
    for detector in detectors:
        items: Sequence[Item] = full.count_truth if detector == COUNT else full.truth
        for item in items:
            cell = _cell(item)
            listed = [item in listing[detector] for listing in listings]
            first = listed.index(True) if any(listed) else None
            held = places.get(cell, [])
            flags.append(
                FirstFlag(
                    detector=detector,
                    item=item,
                    cases=len(held),
                    truth_from=held[0] if detector == COUNT else entered[item],
                    first_flag=first,
                    cell_removed=None if first is None else bisect.bisect_right(held, first),
                    reversals=sum(was and not now for was, now in itertools.pairwise(listed)),
                )
            )
    return tuple(flags)


def first_flag_rows(name: str, flags: Sequence[FirstFlag], label: Label) -> Iterator[Row]:
    """The CSV rows of one ablation's flags, in `FIRST_FLAG_COLUMNS` order.

    A count row leaves the statement columns blank, a definition's row leaves `target` blank, and
    an item never listed leaves `first_flag` and `cell_removed` blank.
    """
    for flag in flags:
        if isinstance(flag.item, StatementGap):
            where = (*statement_cells(flag.item.statement, label), *cell_cells(flag.item.cell))
        else:
            where = (*("",) * len(STATEMENT_COLUMNS), *cell_cells(flag.item))
        yield (
            name,
            flag.detector,
            *where,
            str(flag.cases),
            str(flag.truth_from),
            _blank(flag.first_flag),
            _blank(flag.cell_removed),
            str(flag.reversals),
            str(flag.flagged_at_all),
        )


def log_first_flags(name: str, flags: Sequence[FirstFlag]) -> None:
    """Log one line per detector and truth cell: its cases, items, items flagged, first and last."""
    groups: dict[tuple[str, Cell], list[FirstFlag]] = {}
    for flag in flags:
        groups.setdefault((flag.detector, _where(flag.item)), []).append(flag)
    for (detector, cell), members in groups.items():
        firsts = [flag.first_flag for flag in members if flag.first_flag is not None]
        logger.info(
            "first-flag ablation=%s detector=%s cell=%s cases=%d items=%d flagged=%d "
            "first=%s last=%s",
            name,
            detector,
            cell_key(cell),
            members[0].cases,
            len(members),
            len(firsts),
            min(firsts) if firsts else "n/a",
            max(firsts) if firsts else "n/a",
        )


def _listings(
    full: AblationRun,
    population: Population,
    spec: AblationSpec,
    fragments: Sequence[BKB],
    rows: Sequence[CodeMapping],
    order: pd.Index,
) -> list[dict[str, Collection[Item]]]:
    """Each detector's listing after each removed count, 0 through every case."""
    first: dict[str, Collection[Item]] = {d.value: set(full.before[d]) for d in Definition}
    last: dict[str, Collection[Item]] = {d.value: set(full.after[d]) for d in Definition}
    first[COUNT] = {gap.cell for gap in full.count_before}
    last[COUNT] = {gap.cell for gap in full.count_after}
    listings = [first]
    for removed in range(1, len(order)):
        kept = without(population, order[:removed])
        cells = population_cells(kept, rows, name=SYNTHEA, edges=spec.edges)
        listing: dict[str, Collection[Item]] = {
            definition.value: set(gaps(definition, fragments, cells)) for definition in Definition
        }
        counts = cell_counts(subjects(kept, codes=spec.codes), edges=spec.edges)
        listing[COUNT] = {gap.cell for gap in count_gaps(counts)}
        listings.append(listing)
    listings.append(last)
    return listings


def _truth_from(
    full: AblationRun,
    population: Population,
    spec: AblationSpec,
    fragments: Sequence[BKB],
    rows: Sequence[CodeMapping],
    order: pd.Index,
    places: dict[Cell, list[int]],
) -> dict[Item, int]:
    """The removed count from which each truth pair's statement its cell's removed cases ground."""
    entered: dict[Item, int] = {}
    for cell in dict.fromkeys(gap.cell for gap in full.truth):
        wanted = {gap for gap in full.truth if gap.cell == cell}
        held = places[cell]
        for taken, removed in enumerate(held, 1):
            mine = order[[place - 1 for place in held[:taken]]]
            alone = without(population, population.patients.index.difference(mine))
            cells = population_cells(alone, rows, name=SYNTHEA, edges=spec.edges)
            for statement in grounded(fragments, dataset_fragment(cells.each[cell].observed)):
                gap = StatementGap(cell, statement)
                if gap in wanted:
                    wanted.discard(gap)
                    entered[gap] = removed
            if not wanted:
                break
    return entered


def _cell(item: Item) -> Cell:
    """The age-sex cell an item lies in; a count cell's label is always that of the cases."""
    return item.cell if isinstance(item, StatementGap) else item[:2]


def _where(item: Item) -> Cell:
    """The cell an item names: age-sex for a definition, age-sex-label for the count."""
    return item.cell if isinstance(item, StatementGap) else item


def _blank(value: int | None) -> str:
    return "" if value is None else str(value)
