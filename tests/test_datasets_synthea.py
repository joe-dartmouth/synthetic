"""Loading a generated Synthea population as the patients alive on its snapshot date."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bko.datasets.cells import cell_counts
from bko.datasets.errors import ContractError, UnavailableError
from bko.datasets.synthea import (
    ISCHEMIC_HEART_DISEASE,
    SYNTHEA_AGE_BAND_EDGES,
    ages,
    load_population,
    subjects,
)
from tests._datasets_fixtures import SMALL_POPULATION, condition, patient, synthea_tables


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

SNAPSHOT = date(2026, 1, 1)
IHD = frozenset({ISCHEMIC_HEART_DISEASE})


def load(tmp_path, patients=None, conditions=None):
    synthea_tables(
        tmp_path,
        patients if patients is not None else SMALL_POPULATION[0],
        conditions if conditions is not None else SMALL_POPULATION[1],
    )
    return load_population(tmp_path, snapshot_date=SNAPSHOT)


def test_only_the_needed_columns_are_loaded(tmp_path):
    population = load(tmp_path)
    assert list(population.patients.columns) == ["birth_date", "sex"]
    assert list(population.conditions.columns) == ["patient", "code", "start"]


def test_patients_dead_or_unborn_on_the_snapshot_are_not_in_the_population(tmp_path):
    unborn = patient("p6", "2026-06-01", "F")
    population = load(tmp_path, [*SMALL_POPULATION[0], unborn])
    assert list(population.patients.index) == ["p1", "p2", "p3", "p4"]
    assert "p5" not in set(population.conditions["patient"])


def test_genders_map_to_the_uci_sex_codes(tmp_path):
    population = load(tmp_path)
    assert population.patients["sex"].to_dict() == {"p1": 1, "p2": 1, "p3": 0, "p4": 1}


@pytest.mark.parametrize(
    ("birth", "age"), [("1960-01-01", 66), ("1960-01-02", 65), ("1959-12-31", 66)]
)
def test_age_is_completed_years_on_the_snapshot(birth, age):
    assert ages(pd.Series(pd.to_datetime([birth])), SNAPSHOT).tolist() == [age]


def test_a_missing_column_is_named(tmp_path):
    (tmp_path / "csv").mkdir()
    (tmp_path / "csv" / "patients.csv").write_text("Id,BIRTHDATE,GENDER\n", encoding="utf-8")
    (tmp_path / "csv" / "conditions.csv").write_text("START,PATIENT,CODE\n", encoding="utf-8")
    with pytest.raises(ContractError, match="DEATHDATE"):
        load_population(tmp_path, snapshot_date=SNAPSHOT)


def test_a_missing_table_is_unavailable(tmp_path):
    with pytest.raises(UnavailableError, match=r"patients\.csv"):
        load_population(tmp_path, snapshot_date=SNAPSHOT)


def test_a_duplicate_id_is_refused(tmp_path):
    with pytest.raises(ContractError, match="duplicate"):
        load(tmp_path, [patient("p1", "1960-01-01", "M"), patient("p1", "1961-01-01", "F")], [])


def test_an_unknown_gender_is_refused(tmp_path):
    with pytest.raises(ContractError, match="GENDER"):
        load(tmp_path, [patient("p1", "1960-01-01", "X")], [])


def test_a_condition_naming_no_patient_is_refused(tmp_path):
    with pytest.raises(ContractError, match="name no patient"):
        load(
            tmp_path,
            [patient("p1", "1960-01-01", "M")],
            [condition("ghost", "414545008", "2020-01-01")],
        )


def test_a_malformed_date_is_refused(tmp_path):
    with pytest.raises(ContractError, match="BIRTHDATE"):
        load(tmp_path, [patient("p1", "01/01/1960", "M")], [])


def test_codes_are_text(tmp_path):
    population = load(
        tmp_path, [patient("p1", "1960-01-01", "M")], [condition("p1", "0123", "2020-01-01")]
    )
    assert population.conditions["code"].tolist() == ["0123"]


def test_subjects_mark_patients_with_the_code_by_the_snapshot(tmp_path):
    late = condition("p4", "414545008", "2026-06-01")
    population = load(tmp_path, conditions=[*SMALL_POPULATION[1], late])
    frame = subjects(population, codes=IHD)
    assert frame["target"].to_dict() == {"p1": 1, "p2": 1, "p3": 1, "p4": 0}
    assert frame["age"].to_dict() == {"p1": 65, "p2": 64, "p3": 64, "p4": 35}


def test_subjects_are_counted_by_the_cell_grid(tmp_path):
    counts = cell_counts(subjects(load(tmp_path), codes=IHD), edges=SYNTHEA_AGE_BAND_EDGES)
    cell = counts.set_index(["age_band", "sex", "target"])["count"]
    assert cell["60-69", "male", 1] == 2
    assert cell["60-69", "female", 1] == 1
    assert counts["count"].sum() == 4


def test_the_snapshot_day_counts_births_and_onsets_but_not_deaths(tmp_path):
    """Born on the snapshot: in. Died on it: out. An onset on it: a case."""
    patients = [
        patient("b", "2026-01-01", "F"),
        patient("d", "1960-01-01", "M", death="2026-01-01"),
        patient("o", "1960-01-01", "M"),
    ]
    conditions = [
        condition("o", "414545008", "2026-01-01"),
        condition("d", "414545008", "2020-01-01"),
    ]
    population = load(tmp_path, patients, conditions)
    assert list(population.patients.index) == ["b", "o"]
    assert subjects(population, codes=IHD)["target"].to_dict() == {"b": 0, "o": 1}


@pytest.mark.parametrize(
    ("patients", "conditions", "where"),
    [
        ([patient("p1", "", "M")], [], "BIRTHDATE"),
        ([patient("p1", "1960-01-01", "M")], [condition("p1", "414545008", "")], "START"),
        ([patient("p1", "1960-01-01", "M", death="2020/01/01")], [], "DEATHDATE"),
        ([patient("p1", "1960-1-5", "M")], [], "BIRTHDATE"),
    ],
    ids=["blank-birth", "blank-start", "malformed-death", "unpadded-birth"],
)
def test_a_blank_or_malformed_date_is_refused(tmp_path, patients, conditions, where):
    with pytest.raises(ContractError, match=f"{where}: .* is not a YYYY-MM-DD date"):
        load(tmp_path, patients, conditions)


@pytest.mark.parametrize(
    ("birth", "age"),
    [
        ("1960-06-14", 66),
        ("1960-06-15", 66),
        ("1960-06-16", 65),
        ("1960-07-01", 65),
        ("1960-05-31", 66),
    ],
)
def test_age_on_a_mid_year_snapshot(birth, age):
    assert ages(pd.Series(pd.to_datetime([birth])), date(2026, 6, 15)).tolist() == [age]


def test_a_leap_day_birth_ages_on_the_first_of_march():
    births = pd.Series(pd.to_datetime(["2000-02-29"]))
    assert ages(births, date(2026, 2, 28)).tolist() == [25]
    assert ages(births, date(2026, 3, 1)).tolist() == [26]
