"""E2's configuration and reading: what the Synthea ablation removes, and what each detector finds.

`docs/EVIDENCE.md` (E2) names these values and ADR `research/synthea-ablation-design` the ablation.
No command-line flag changes a value here.

A removal can only add gaps, and some cells hold fewer records than the threshold before any
removal, so recovery is read from what the removal changed. For each detector the truth is the gaps
the removed patients account for: the cells whose count fell, and the statements the removed
patients' records ground by themselves, whether or not other patients ground them too. A truth gap
present before the removal is not eligible; an eligible gap present after it is recovered; a gap
new after it and outside the truth is spurious, which neither detector can produce, so a spurious
gap is a defect.
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
from bko.gaps.detect import CellGap, Statement, count_gaps, detect, grounded
from bko.gaps.observe import dataset_fragment, observe_population
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
"""E2's ablations, in report order: one cell, which other patients' records still ground for the
BKO detector, and the whole condition, which they do not."""


@dataclass(frozen=True, slots=True)
class Recovery:
    """One detector's reading of one removal.

    Attributes:
        truth: The gaps the removed patients account for.
        eligible: The truth gaps absent before the removal.
        recovered: The eligible gaps present after it.
        spurious: The gaps new after it and outside the truth.
    """

    truth: int
    eligible: int
    recovered: int
    spurious: int


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
        spurious=len(now - was - true),
    )


@dataclass(frozen=True, slots=True, eq=False)
class AblationRun:
    """One ablation, read by both detectors.

    Attributes:
        name: The ablation's name in `E2_ABLATIONS`.
        ablation: The removal and its ground truth.
        bko_before: The ungrounded statements before the removal.
        bko_after: Those after it.
        bko_truth: The statements the removed patients' records ground by themselves.
        count_before: The underrepresented cells before the removal.
        count_after: Those after it.
        count_truth: The cells whose count fell, in grid order.
    """

    name: str
    ablation: Ablation
    bko_before: tuple[Statement, ...]
    bko_after: tuple[Statement, ...]
    bko_truth: tuple[Statement, ...]
    count_before: tuple[CellGap, ...]
    count_after: tuple[CellGap, ...]
    count_truth: tuple[tuple[str, str, int], ...]

    @property
    def bko(self) -> Recovery:
        """The BKO detector's reading."""
        return recovery(self.bko_before, self.bko_after, self.bko_truth)

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
    ontology: Sequence[BKB],
    rows: Sequence[CodeMapping],
) -> AblationRun:
    """Remove `spec`'s cases from `population` and read both detectors before and after.

    Args:
        name: The ablation's name.
        population: The population before the removal.
        spec: What to remove.
        ontology: The ontology fragments over the concepts `rows` map.
        rows: The code mappings.

    Raises:
        DatasetError: As `ablate`.
        GapError: If the population, or what is left of it, holds no patient.
    """
    ablation = ablate(population, spec)
    removed = _removed(population, ablation.removed)
    return AblationRun(
        name=name,
        ablation=ablation,
        bko_before=detect(ontology, _fragment(population, rows)),
        bko_after=detect(ontology, _fragment(ablation.kept, rows)),
        bko_truth=grounded(ontology, _fragment(removed, rows)),
        count_before=count_gaps(ablation.before),
        count_after=count_gaps(ablation.after),
        count_truth=_fallen(ablation.before, ablation.after),
    )


def _fragment(population: Population, rows: Sequence[CodeMapping]) -> BKB:
    return dataset_fragment(observe_population(population, rows, name=SYNTHEA))


def _removed(population: Population, removed: pd.Index) -> Population:
    conditions = population.conditions
    return Population(
        patients=population.patients.loc[removed],
        conditions=conditions.loc[conditions["patient"].isin(removed)].reset_index(drop=True),
        snapshot_date=population.snapshot_date,
    )


def _fallen(before: pd.DataFrame, after: pd.DataFrame) -> tuple[tuple[str, str, int], ...]:
    fell = after["count"].to_numpy() < before["count"].to_numpy()
    cells = before.loc[fell, list(CELL_COLUMNS)].itertuples(index=False, name=None)
    return tuple((str(band), str(sex), int(target)) for band, sex, target in cells)
