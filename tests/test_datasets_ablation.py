"""Removing a spec's patients from a population, keeping them as the ground truth."""

from __future__ import annotations

from datetime import date

import pytest

from bko.datasets.ablation import AblationSpec, ablate
from bko.datasets.errors import DatasetError
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, load_population
from tests._datasets_fixtures import SMALL_POPULATION, synthea_tables


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

IHD = frozenset({ISCHEMIC_HEART_DISEASE})
CELL = AblationSpec(codes=IHD, age_band="60-69", sex=1)
TARGET = ("60-69", "male", 1)


@pytest.fixture
def population(tmp_path):
    synthea_tables(tmp_path, *SMALL_POPULATION)
    return load_population(tmp_path, snapshot_date=date(2026, 1, 1))


def counts(frame):
    return frame.set_index(["age_band", "sex", "target"])["count"]


def test_the_cells_patients_are_removed(population):
    assert list(ablate(population, CELL).removed) == ["p1", "p2"]


def test_kept_and_removed_partition_the_population(population):
    result = ablate(population, CELL)
    kept = set(result.kept.patients.index)
    assert kept.isdisjoint(result.removed)
    assert kept | set(result.removed) == set(population.patients.index)


def test_no_row_of_a_removed_patient_survives(population):
    assert {"p1", "p2"} <= set(population.conditions["patient"])
    result = ablate(population, CELL)
    assert not set(result.kept.conditions["patient"]) & set(result.removed)
    assert len(result.kept.conditions) + 3 == len(population.conditions)


def test_the_cell_empties_and_no_other_cell_moves(population):
    result = ablate(population, CELL)
    before, after = counts(result.before), counts(result.after)
    assert (before[TARGET], after[TARGET]) == (2, 0)
    assert before.drop(TARGET).equals(after.drop(TARGET))


def test_a_spec_without_band_or_sex_removes_the_condition_everywhere(population):
    result = ablate(population, AblationSpec(codes=IHD, age_band=None, sex=None))
    assert list(result.removed) == ["p1", "p2", "p3"]
    assert counts(result.after).xs(1, level="target").sum() == 0


def test_the_population_is_not_mutated(population):
    patients, conditions = population.patients.copy(), population.conditions.copy()
    ablate(population, CELL)
    assert population.patients.equals(patients)
    assert population.conditions.equals(conditions)


def test_a_spec_matching_no_patient_names_the_funnel(population):
    with pytest.raises(DatasetError, match="3 patients have"):
        ablate(population, AblationSpec(codes=IHD, age_band="40-49", sex=0))


def test_an_unknown_band_is_refused(population):
    with pytest.raises(DatasetError, match="age band"):
        ablate(population, AblationSpec(codes=IHD, age_band="65-70", sex=1))
