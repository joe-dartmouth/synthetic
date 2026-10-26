"""Remove a spec's patients from a Synthea population, keeping them as the ground truth.

A case is a patient: every patient with one of the spec's codes by the snapshot date, narrowed to
the spec's age band and sex when it names them, is removed with all their condition rows, so
nothing is left in the loaded tables to infer the case from. A spec removes all of its cases, or a
fraction of them drawn by a seed. Removal of a case is complete, because a single remaining record
grounds a BKO statement (ADR `research/synthea-ablation-design`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
            cases are chosen by a draw seeded with `seed`, and the count rounds up, so a positive
            fraction removes at least one case.
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


def ablate(population: Population, spec: AblationSpec) -> Ablation:
    """Remove every case `spec` selects from `population`.

    Args:
        population: The population; it is not modified.
        spec: What to remove.

    Returns:
        The ablation.

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
    removed = people.index[selected.to_numpy()].sort_values()
    if removed.empty:
        where = [
            *([f"age band {spec.age_band}"] if spec.age_band is not None else []),
            *([f"sex {SEXES[spec.sex]}"] if spec.sex is not None else []),
        ]
        msg = (
            f"the spec selects no patient: {int(people[LABEL].sum())} patients have "
            f"{sorted(spec.codes)}, none in {' and '.join(where) or 'the population'}"
        )
        raise DatasetError(msg)
    removed = _drawn(removed, spec)
    conditions = population.conditions
    kept = Population(
        patients=population.patients.drop(index=removed),
        conditions=conditions.loc[~conditions["patient"].isin(removed)].reset_index(drop=True),
        snapshot_date=population.snapshot_date,
    )
    return Ablation(
        spec=spec,
        kept=kept,
        removed=removed,
        before=cell_counts(people, edges=spec.edges),
        after=cell_counts(subjects(kept, codes=spec.codes), edges=spec.edges),
    )


def _drawn(selected: pd.Index, spec: AblationSpec) -> pd.Index:
    """The first `spec.fraction` of `selected` in a draw seeded by `spec.seed`, sorted."""
    if spec.fraction == 1:
        return selected
    count = math.ceil(spec.fraction * len(selected))
    order = np.random.default_rng(spec.seed).permutation(len(selected))
    return selected[np.sort(order[:count])]
