"""E2's configuration and reading: what the Synthea ablation removes, and what each detector finds.

`docs/EVIDENCE.md` (E2) names these values and ADR `research/synthea-ablation-design` the ablation.
No command-line flag changes a value here.

Every detector reads the ablation's age bands: each gap definition of `bko.gaps.definitions` over
(cell, statement) pairs on the age-sex cells, and the count-based detector over the age-sex-label
cells. For each, the truth is the gaps the removed patients account for: for a definition, the
pairs whose statement the removed patients of the cell ground, whether or not the cell's other
patients ground it too; for the count-based detector, the cells whose count fell. A truth gap
present before the removal is not eligible; an eligible gap present after it is recovered; a gap
after it that lies outside the truth is outside, and new outside when it was absent before; a gap
present before it and absent after it vanished. Every ontology statement reads from one class, so
under a removal of whole patients the per-subgroup definition and the count-based detector put no
gap new outside the truth and lose none; population-level gaps hold in every cell, so they fall new
outside the truth in the cells the removal never touched; the frequency-aware reference share moves
with the removal, so its gaps can vanish.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.bkb.structure import BKB
from bko.datasets.ablation import DEFAULT_ABLATION, Ablation, AblationSpec, ablate
from bko.datasets.cells import CELL_COLUMNS
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, Population
from bko.datasets.synthea_run import DEFAULT_RUN
from bko.gaps.definitions import Definition, StatementGap, gaps
from bko.gaps.detect import CellGap, Statement, count_gaps, grounded
from bko.gaps.errors import GapError
from bko.gaps.observe import Cell, Cells, cell_key, dataset_fragment, population_cells
from bko.ontology.schema import CodeMapping


SYNTHEA: Final = "synthea"
"""The source name of a Synthea population's dataset fragment."""

E2_RUN: Final = DEFAULT_RUN
"""The population E2 reads."""

E2_ABLATIONS: Final[Mapping[str, AblationSpec]] = MappingProxyType(
    {
        "cell": DEFAULT_ABLATION,
        "condition": AblationSpec(
            codes=frozenset({ISCHEMIC_HEART_DISEASE}), age_band=None, sex=None
        ),
    }
)
"""E2's ablations, in report order: one cell, whose statements other patients' records still ground
for population-level grounding, and the whole condition, whose statements they do not."""


@dataclass(frozen=True, slots=True)
class Recovery:
    """One detector's reading of one removal.

    Attributes:
        truth: The gaps the removed patients account for.
        eligible: The truth gaps absent before the removal.
        recovered: The eligible gaps present after it.
        outside: The gaps after it that lie outside the truth.
        new_outside: Those of them absent before it.
        vanished: The gaps present before it and absent after it.
    """

    truth: int
    eligible: int
    recovered: int
    outside: int
    new_outside: int
    vanished: int


def recovery(
    before: Iterable[Hashable], after: Iterable[Hashable], truth: Iterable[Hashable]
) -> Recovery:
    """Read one detector's gaps before and after a removal against the removal's truth."""
    was, now, true = set(before), set(after), set(truth)
    eligible = true - was
    return Recovery(
        truth=len(true),
        eligible=len(eligible),
        recovered=len(eligible & now),
        outside=len(now - true),
        new_outside=len(now - was - true),
        vanished=len(was - now),
    )


@dataclass(frozen=True, slots=True, eq=False)
class AblationRun:
    """One ablation, read by every gap definition and the count-based detector.

    Attributes:
        name: The ablation's name in `E2_ABLATIONS`.
        ablation: The removal and its ground truth.
        before_cells: The population before the removal, observed per age-sex cell.
        after_cells: The population after it.
        truth: The (cell, statement) pairs whose statement the removed patients of the cell ground,
            in grid order.
        before: Each definition's gaps before the removal; a read-only copy.
        after: Each definition's gaps after it; a read-only copy.
        count_before: The underrepresented cells before the removal.
        count_after: Those after it.
        count_truth: The cells whose count fell, in grid order.
    """

    name: str
    ablation: Ablation
    before_cells: Cells
    after_cells: Cells
    truth: tuple[StatementGap, ...]
    before: Mapping[Definition, tuple[StatementGap, ...]]
    after: Mapping[Definition, tuple[StatementGap, ...]]
    count_before: tuple[CellGap, ...]
    count_after: tuple[CellGap, ...]
    count_truth: tuple[tuple[str, str, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "before", MappingProxyType(dict(self.before)))
        object.__setattr__(self, "after", MappingProxyType(dict(self.after)))

    def reading(self, definition: Definition) -> Recovery:
        """`definition`'s reading of the removal."""
        return recovery(self.before[definition], self.after[definition], self.truth)

    def listed(self, definition: Definition) -> tuple[StatementGap, ...]:
        """`definition`'s gaps before or after the removal and the truth, in grid order.

        Each cell's statements are sorted; the grid is walked, never sorted, since band labels
        sort out of order (`40-49` before `<40`).
        """
        by_cell: dict[Cell, list[Statement]] = {}
        for gap in {*self.before[definition], *self.after[definition], *self.truth}:
            by_cell.setdefault(gap.cell, []).append(gap.statement)
        off_grid = sorted(cell_key(cell) for cell in by_cell.keys() - self.before_cells.each.keys())
        if off_grid:
            msg = f"the {self.name} ablation lists gaps off its grid: {', '.join(off_grid)}"
            raise GapError(msg)
        return tuple(
            StatementGap(cell, statement)
            for cell in self.before_cells.each
            for statement in sorted(by_cell.get(cell, ()))
        )

    @property
    def count(self) -> Recovery:
        """The count-based detector's reading."""
        return recovery(
            (gap.cell for gap in self.count_before),
            (gap.cell for gap in self.count_after),
            self.count_truth,
        )


def run_ablation(
    name: str,
    population: Population,
    spec: AblationSpec,
    fragments: Sequence[BKB],
    rows: Sequence[CodeMapping],
) -> AblationRun:
    """Remove `spec`'s cases from `population` and read every detector before and after.

    Every reading uses `spec`'s age bands, so the definitions' cells and the count-based
    detector's share a grid.

    Args:
        name: The ablation's name.
        population: The population before the removal.
        spec: What to remove.
        fragments: The ontology fragments over the concepts `rows` map.
        rows: The code mappings.

    Raises:
        DatasetError: As `ablate`.
        GapError: If the removal leaves no patient.
    """
    ablation = ablate(population, spec)
    if ablation.kept.patients.empty:
        msg = f"the {name} ablation removes every patient, so no concept has a share after it"
        raise GapError(msg)
    before = population_cells(population, rows, name=SYNTHEA, edges=spec.edges)
    after = population_cells(ablation.kept, rows, name=SYNTHEA, edges=spec.edges)
    removed = population_cells(
        _removed(population, ablation.removed), rows, name=SYNTHEA, edges=spec.edges
    )
    return AblationRun(
        name=name,
        ablation=ablation,
        before_cells=before,
        after_cells=after,
        truth=_truth(fragments, removed),
        before={definition: gaps(definition, fragments, before) for definition in Definition},
        after={definition: gaps(definition, fragments, after) for definition in Definition},
        count_before=count_gaps(ablation.before),
        count_after=count_gaps(ablation.after),
        count_truth=_fallen(ablation.before, ablation.after),
    )


def _truth(fragments: Sequence[BKB], removed: Cells) -> tuple[StatementGap, ...]:
    return tuple(
        StatementGap(cell, statement)
        for cell, records in removed.each.items()
        for statement in grounded(fragments, dataset_fragment(records.observed))
    )


def _removed(population: Population, removed: pd.Index) -> Population:
    conditions = population.conditions
    return Population(
        patients=population.patients.loc[removed],
        conditions=conditions.loc[conditions["patient"].isin(removed)].reset_index(drop=True),
        snapshot_date=population.snapshot_date,
    )


def _fallen(before: pd.DataFrame, after: pd.DataFrame) -> tuple[tuple[str, str, int], ...]:
    was = before.set_index(list(CELL_COLUMNS))["count"]
    now = after.set_index(list(CELL_COLUMNS))["count"].loc[was.index]
    fell = was.index[(now < was).to_numpy()]
    return tuple((str(band), str(sex), int(target)) for band, sex, target in fell)
