"""Remove a spec's patients from a Synthea population, keeping them as the ground truth.

A case is a patient: every patient with one of the spec's codes by the snapshot date, narrowed to
the spec's age band and sex when it names them, is removed with all their condition rows, so
nothing is left in the loaded tables to infer the case from. A spec removes all of its cases, or a
fraction of them drawn by a seed. Removal of a case is complete, because a single remaining record
grounds a BKO statement (ADR `research/synthea-ablation-design`). `thinned` removes a fraction of
one cell's training rows of a frame the same way, by the same draw and count.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
import pandas as pd

from bko.datasets.cells import LABEL, SEXES, age_band_labels, cell_counts, cell_keys
from bko.datasets.errors import DatasetError
from bko.datasets.synthea import (
    ISCHEMIC_HEART_DISEASE,
    SYNTHEA_AGE_BAND_EDGES,
    Population,
    subjects,
)


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """The cases to remove.

    Attributes:
        codes: Condition codes that make a case.
        age_band: One of `age_band_labels(edges)`, or `None` for every band.
        sex: 0 female, 1 male, or `None` for both.
        edges: Age band starts.
        fraction: The share of the selected cases to remove, in (0, 1]; 1 removes them all. Fewer
            cases are chosen by a draw seeded with `seed`, and the count is `ceil(f * n)` of `n`
            cases, computed exactly from the fraction's decimal text (0.07 of 100 is 7, which
            floating point would make 8), so a positive fraction removes at least one case.
        seed: The seed of the draw that orders the cases when `fraction` is below 1.
    """

    codes: frozenset[str]
    age_band: str | None
    sex: int | None
    edges: tuple[int, ...] = SYNTHEA_AGE_BAND_EDGES
    fraction: float = 1.0
    seed: int = 0


DEFAULT_ABLATION: Final = AblationSpec(
    codes=frozenset({ISCHEMIC_HEART_DISEASE}), age_band="60-69", sex=1
)
"""The command's demonstration cell, chosen by the pilot ADR `research/synthea-ablation-design`
records. E2's configuration fixes its own targets."""


@dataclass(frozen=True, slots=True, eq=False)
class Ablation:
    """An ablated population and its ground truth.

    Attributes:
        spec: What was removed.
        kept: The population without the removed patients or any of their rows.
        removed: The removed patient ids, sorted.
        before: `cell_counts` of the population before removal.
        after: `cell_counts` after.
    """

    spec: AblationSpec
    kept: Population
    removed: pd.Index
    before: pd.DataFrame
    after: pd.DataFrame


def cases(population: Population, spec: AblationSpec) -> pd.Index:
    """Every case `spec` selects from `population`, whatever its fraction.

    Args:
        population: The population; it is not modified.
        spec: What to select.

    Returns:
        The selected patient ids, sorted.

    Raises:
        DatasetError: If `spec` names no code, bad band edges, an unknown band or sex, a fraction
            outside (0, 1], or selects no patient.
    """
    try:
        labels = age_band_labels(spec.edges)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    if not spec.codes:
        msg = "an ablation needs at least one condition code"
        raise DatasetError(msg)
    if spec.age_band is not None and spec.age_band not in labels:
        msg = f"age band {spec.age_band!r} is not one of {labels}"
        raise DatasetError(msg)
    if spec.sex is not None and (isinstance(spec.sex, bool) or spec.sex not in SEXES):
        msg = f"sex must be 0, 1 or None, not {spec.sex}"
        raise DatasetError(msg)
    if not 0 < spec.fraction <= 1:
        msg = f"the removed fraction must lie in (0, 1], not {spec.fraction}"
        raise DatasetError(msg)
    people = subjects(population, codes=spec.codes)
    keys = cell_keys(people, edges=spec.edges)
    selected = keys[LABEL] == 1
    if spec.age_band is not None:
        selected &= keys["age_band"] == spec.age_band
    if spec.sex is not None:
        selected &= keys["sex"] == SEXES[spec.sex]
    found = people.index[selected.to_numpy()].sort_values()
    if found.empty:
        where = [
            *([f"age band {spec.age_band}"] if spec.age_band is not None else []),
            *([f"sex {SEXES[spec.sex]}"] if spec.sex is not None else []),
        ]
        msg = (
            f"the spec selects no patient: {int(people[LABEL].sum())} patients have "
            f"{sorted(spec.codes)}, none in {' and '.join(where) or 'the population'}"
        )
        raise DatasetError(msg)
    return found


def removal_order(selected: pd.Index, seed: int) -> pd.Index:
    """`selected` in the order a draw seeded by `seed` removes them.

    A fraction `f` of `selected` removes the first `ceil(f * len(selected))` of this order, the
    product taken exactly, so each fraction at one seed removes the cases of every smaller one.
    The order is numpy's `default_rng` stream under the locked numpy.

    Args:
        selected: The cases, sorted, as `cases` returns them.
        seed: The draw's seed.

    Returns:
        `selected`, permuted.
    """
    return selected[np.random.default_rng(seed).permutation(len(selected))]


def without(population: Population, removed: pd.Index) -> Population:
    """`population` without the `removed` patients or any of their condition rows.

    Args:
        population: The population; it is not modified.
        removed: Patient ids of `population`.

    Returns:
        The rest of the population.
    """
    conditions = population.conditions
    return Population(
        patients=population.patients.drop(index=removed),
        conditions=conditions.loc[~conditions["patient"].isin(removed)].reset_index(drop=True),
        snapshot_date=population.snapshot_date,
    )


def only(population: Population, kept: pd.Index) -> Population:
    """The `kept` patients of `population` alone, with their condition rows and no one else's.

    Args:
        population: The population; it is not modified.
        kept: Patient ids of `population`, in the order the result holds them.

    Returns:
        Those patients.
    """
    conditions = population.conditions
    return Population(
        patients=population.patients.loc[kept],
        conditions=conditions.loc[conditions["patient"].isin(kept)].reset_index(drop=True),
        snapshot_date=population.snapshot_date,
    )


def ablate(population: Population, spec: AblationSpec) -> Ablation:
    """Remove every case `spec` selects from `population`, or its seeded fraction of them.

    Args:
        population: The population; it is not modified.
        spec: What to remove.

    Returns:
        The ablation.

    Raises:
        DatasetError: As `cases`.
    """
    removed = _drawn(cases(population, spec), spec)
    kept = without(population, removed)
    return Ablation(
        spec=spec,
        kept=kept,
        removed=removed,
        before=cell_counts(subjects(population, codes=spec.codes), edges=spec.edges),
        after=cell_counts(subjects(kept, codes=spec.codes), edges=spec.edges),
    )


def removed_count(fraction: float, selected: int) -> int:
    """`ceil(fraction * selected)`, the product taken exactly from the fraction's decimal text."""
    return math.ceil(Fraction(str(fraction)) * selected)


def thinned(
    train: pd.DataFrame,
    *,
    cell: tuple[str | int, ...],
    edges: Sequence[int],
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`train` without `fraction` of its rows in `cell`, drawn as an ablation draws its cases.

    The cell's rows, sorted by index, are ordered by `removal_order` at `seed`, and the first
    `removed_count` of them leave; every other row stays, in its order. E3 thins a holdout's
    training rows this way and never its test rows (ADR `research/synthea-downstream-design`).

    Args:
        train: The training rows, with `age`, `sex` and `LABEL`.
        cell: The age band, sex label and label of the rows to thin.
        edges: Age band starts.
        fraction: The share of the cell's rows to remove, in (0, 1).
        seed: The draw's seed.

    Returns:
        The rows kept, then the rows removed, each in `train`'s order.

    Raises:
        DatasetError: If the fraction is not in (0, 1) or `cell` holds no row of `train`.
    """
    if not 0 < fraction < 1:
        msg = f"the thinned fraction must lie in (0, 1), not {fraction}"
        raise DatasetError(msg)
    keys = cell_keys(train, edges=edges)
    band, sex, label = cell
    inside = (keys["age_band"] == band) & (keys["sex"] == sex) & (keys[LABEL] == label)
    selected = train.index[inside.to_numpy()].sort_values()
    if selected.empty:
        msg = f"cell {band}/{sex}/{label} holds no training row to thin"
        raise DatasetError(msg)
    removed = train.index.isin(
        removal_order(selected, seed)[: removed_count(fraction, len(selected))]
    )
    return train.loc[~removed], train.loc[removed]


def _drawn(selected: pd.Index, spec: AblationSpec) -> pd.Index:
    """The first `spec.fraction` of `selected` in a draw seeded by `spec.seed`, sorted."""
    if spec.fraction == 1:
        return selected
    count = removed_count(spec.fraction, len(selected))
    return removal_order(selected, spec.seed)[:count].sort_values()
