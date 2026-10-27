"""Removing a spec's patients from a population, keeping them as the ground truth."""

from __future__ import annotations

import math
from datetime import date

import pytest

from bko.datasets.ablation import AblationSpec, ablate, cases, removal_order
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


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (AblationSpec(codes=frozenset(), age_band=None, sex=None), "at least one condition code"),
        (AblationSpec(codes=IHD, age_band=None, sex=2), "sex must be 0, 1 or None"),
        (AblationSpec(codes=IHD, age_band=None, sex=True), "sex must be 0, 1 or None"),
        (AblationSpec(codes=IHD, age_band=None, sex=None, edges=(60, 40)), "strictly increasing"),
    ],
    ids=["no-codes", "sex-2", "sex-bool", "bad-edges"],
)
def test_a_malformed_spec_is_refused(population, spec, message):
    with pytest.raises(DatasetError, match=message):
        ablate(population, spec)


def test_every_code_in_the_spec_makes_a_case(population):
    spec = AblationSpec(codes=IHD | {"22298006"}, age_band=None, sex=None)
    assert list(ablate(population, spec).removed) == ["p1", "p2", "p3", "p4"]


def test_a_whole_population_spec_that_matches_nothing_says_so(population):
    with pytest.raises(DatasetError, match=r"0 patients have \['999'\], none in the population$"):
        ablate(population, AblationSpec(codes=frozenset({"999"}), age_band=None, sex=None))


def test_a_fraction_removes_that_share_of_the_cases_rounded_up(population):
    spec = AblationSpec(codes=IHD, age_band=None, sex=None, fraction=0.5)
    assert len(ablate(population, spec).removed) == 2
    tiny = AblationSpec(codes=IHD, age_band=None, sex=None, fraction=0.01)
    assert len(ablate(population, tiny).removed) == 1


def test_a_fractions_draw_is_seeded_sorted_and_a_subset_of_the_cases(population):
    spec = AblationSpec(codes=IHD, age_band=None, sex=None, fraction=0.5, seed=3)
    first, again = ablate(population, spec), ablate(population, spec)
    assert list(first.removed) == list(again.removed)
    assert list(first.removed) == sorted(first.removed)
    assert set(first.removed) <= {"p1", "p2", "p3"}
    kept = set(first.kept.patients.index)
    assert kept.isdisjoint(first.removed)
    assert kept | set(first.removed) == set(population.patients.index)


def test_a_fraction_of_one_removes_every_case(population):
    spec = AblationSpec(codes=IHD, age_band=None, sex=None, fraction=1.0, seed=9)
    assert list(ablate(population, spec).removed) == ["p1", "p2", "p3"]


@pytest.mark.parametrize("fraction", [0, -0.5, 1.5])
def test_a_fraction_outside_zero_to_one_is_refused(population, fraction):
    with pytest.raises(DatasetError, match="fraction"):
        ablate(population, AblationSpec(codes=IHD, age_band=None, sex=None, fraction=fraction))


def test_cases_are_every_selected_patient_sorted(population):
    assert list(cases(population, AblationSpec(codes=IHD, age_band=None, sex=None))) == [
        "p1",
        "p2",
        "p3",
    ]
    with pytest.raises(DatasetError, match="age band"):
        cases(population, AblationSpec(codes=IHD, age_band="65-70", sex=1))


def test_a_removal_order_is_a_seeded_permutation_of_the_cases(population):
    selected = cases(population, AblationSpec(codes=IHD, age_band=None, sex=None))
    order = removal_order(selected, 3)
    assert list(order) == list(removal_order(selected, 3))
    assert sorted(order) == list(selected)


@pytest.mark.parametrize("seed", [0, 1, 3])
@pytest.mark.parametrize("fraction", [0.01, 0.34, 0.5, 0.67, 0.99, 1.0])
def test_a_fraction_removes_the_first_cases_of_its_seeds_order(population, seed, fraction):
    spec = AblationSpec(codes=IHD, age_band=None, sex=None, fraction=fraction, seed=seed)
    order = removal_order(cases(population, spec), seed)
    expected = order[: math.ceil(fraction * len(order))].sort_values()
    assert list(ablate(population, spec).removed) == list(expected)
