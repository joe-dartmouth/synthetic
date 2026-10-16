"""E1's configuration, fixed before any run, and the requests it gives every arm on a split.

ADR `research/e1-evaluation-protocol` records the protocol and `docs/EVIDENCE.md` (E1) states how
each criterion is read. No command-line flag changes a value here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from bko.baselines.request import SEED_LIMIT, Method, Request, Strata, Stratum
from bko.datasets.cells import UNDERREPRESENTED_BELOW, cell_counts
from bko.evaluate.errors import EvaluationError


@dataclass(frozen=True, slots=True)
class Experiment:
    """What a run of the harness fixes before it starts.

    Attributes:
        holdout_seeds: One cell-stratified split per seed (`bko.datasets.split`); every request on
            a split takes its seed.
        arms: The arms scored on every split, by name.
        candidate: The arm the criteria judge.
        comparators: The arms the candidate must beat.
        replica_arms: The arms also asked for a replica of the training split's cells.
        fill_to: Each cell holding fewer training records than this is filled up to it.
        margin: What the candidate must beat each comparator by, in the measure's own units.
        privacy_margin: The share of the test rows' 5th-percentile distance the added rows' must
            reach.
        confidence: The two-sided level of every interval.
    """

    holdout_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    candidate: str
    comparators: tuple[str, ...]
    replica_arms: tuple[str, ...]
    fill_to: int = UNDERREPRESENTED_BELOW
    margin: float = 0.02
    privacy_margin: float = 0.9
    confidence: float = 0.95

    def __post_init__(self) -> None:
        """Refuse a configuration the harness cannot run or judge.

        Raises:
            ValueError: If there are fewer than 2 holdout seeds, a seed repeats or leaves
                `[0, SEED_LIMIT)`, an arm is named twice, a comparator or replica arm is not an
                arm, or the candidate is among its comparators.
        """
        seeds = self.holdout_seeds
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            msg = f"holdout seeds {seeds} must be 2 or more distinct seeds"
            raise ValueError(msg)
        if not all(0 <= seed < SEED_LIMIT for seed in seeds):
            msg = f"every holdout seed must lie in [0, {SEED_LIMIT})"
            raise ValueError(msg)
        if len(set(self.arms)) != len(self.arms):
            msg = f"arms {self.arms} name an arm twice"
            raise ValueError(msg)
        stray = sorted((set(self.comparators) | set(self.replica_arms)) - set(self.arms))
        if stray:
            msg = f"{stray} are not arms of this experiment"
            raise ValueError(msg)
        if self.candidate in self.comparators:
            msg = f"the candidate {self.candidate} cannot be one of its comparators"
            raise ValueError(msg)


HOLDOUTS: Final = 100
"""E1's holdout count; its seeds are 0 to 99."""

METHOD_ARMS: Final = tuple(method.value for method in Method)
"""Every comparison method of `bko.baselines`, by name."""

E1: Final = Experiment(
    holdout_seeds=tuple(range(HOLDOUTS)),
    arms=METHOD_ARMS,
    candidate="bko",
    comparators=METHOD_ARMS,
    replica_arms=tuple(method.value for method in Method if method.adds_rows),
)
"""E1's configuration; `docs/EVIDENCE.md` (E1) names it."""


def flagged(train: pd.DataFrame, experiment: Experiment) -> tuple[Stratum, ...]:
    """The cells holding fewer than `experiment.fill_to` training records, in grid order."""
    return tuple(_short(train, experiment))


def gap_request(train: pd.DataFrame, experiment: Experiment, *, seed: int) -> Request:
    """Ask for every flagged cell to be filled up to `experiment.fill_to` training records.

    Raises:
        EvaluationError: If no cell is flagged, so the row-adding arms have nothing to fill.
    """
    quota = _short(train, experiment)
    if not quota:
        msg = f"no cell holds fewer than {experiment.fill_to} training records; nothing to fill"
        raise EvaluationError(msg)
    return Request(quota=quota, strata=Strata.CELL, seed=seed)


def replica_request(train: pd.DataFrame, *, seed: int) -> Request:
    """Ask for as many rows per cell as the training split holds: a synthetic copy of its cells."""
    quota = {cell: count for cell, count, _ in _cells(train, UNDERREPRESENTED_BELOW) if count}
    return Request(quota=quota, strata=Strata.CELL, seed=seed)


def _short(train: pd.DataFrame, experiment: Experiment) -> dict[Stratum, int]:
    return {
        cell: experiment.fill_to - count
        for cell, count, short in _cells(train, experiment.fill_to)
        if short
    }


def _cells(train: pd.DataFrame, below: int) -> list[tuple[Stratum, int, bool]]:
    counts = cell_counts(train, below=below)
    return [
        ((str(band), str(sex), int(label)), int(count), bool(short))
        for band, sex, label, count, short in counts.itertuples(index=False)
    ]
