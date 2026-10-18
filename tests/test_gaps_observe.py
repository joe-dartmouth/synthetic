"""The dataset fragment: one prior per mapped concept its records show, at its share of them."""

from __future__ import annotations

from datetime import date
from types import MappingProxyType

import pandas as pd
import pytest

from bko.bkb.structure import SNode
from bko.datasets.cells import LABEL
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, load_population
from bko.gaps.errors import GapError
from bko.gaps.observe import (
    Observed,
    dataset_fragment,
    observe_population,
    observe_records,
    population_cells,
    record_cells,
)
from bko.ontology.fragments import INDIVIDUAL
from bko.ontology.schema import AtLeast, CodeMapping, Codes, FieldMapping
from tests._datasets_fixtures import SMALL_POPULATION, condition, patient, synthea_tables
from tests._gaps_fixtures import ANGINA, ISCHEMIA, STENOSIS


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CHOL = FieldMapping(
    when=(AtLeast("chol", 240.0),),
    concept="HP:9000003",
    label="Angina pectoris",
    source="test",
    limits="test",
)
"""A second row on ANGINA's concept, so a record either row maps shows it once."""


def frame(*rows: tuple[int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["cp", "target", "chol"], dtype="int64")


def test_a_record_two_rows_map_to_one_concept_counts_once():
    observed = observe_records(
        frame((1, 0, 250), (1, 0, 100), (4, 0, 250), (4, 1, 100)),
        [ANGINA, CHOL, STENOSIS],
        name="uci",
    )
    assert (observed.name, observed.records) == ("uci", 4)
    assert observed.holders == {"HP:9000003": 3, "MONDO:9000004": 1}


def test_the_at_least_limit_is_inclusive():
    observed = observe_records(frame((4, 0, 239), (4, 0, 240)), [CHOL], name="uci")
    assert observed.holders == {"HP:9000003": 1}


def test_a_concept_no_record_shows_has_no_prior_and_one_record_is_enough():
    observed = observe_records(frame((1, 0, 100), (4, 0, 100)), [ANGINA, STENOSIS], name="uci")
    assert observed.held == ("HP:9000003",)
    assert dataset_fragment(observed).snodes == (
        SNode(head=("HP:9000003", INDIVIDUAL), prob=0.5, tail=frozenset()),
    )


def test_each_prior_is_its_concept_share_of_the_records():
    holders = MappingProxyType({"A:1": 1, "B:1": 3, "C:1": 0})
    fragment = dataset_fragment(Observed(name="uci", records=3, holders=holders))
    assert fragment.name == "uci"
    assert [(s.head[0], s.prob) for s in fragment.snodes] == [("A:1", 1 / 3), ("B:1", 1.0)]


def test_an_empty_dataset_is_refused():
    with pytest.raises(GapError, match="uci holds no record"):
        observe_records(frame(), [ANGINA], name="uci")


def test_a_row_on_a_field_the_records_lack_is_refused():
    thal = FieldMapping(
        when=(AtLeast("thal", 6.0),), concept="HP:1", label="x", source="t", limits="t"
    )
    with pytest.raises(GapError, match="the rows read fields uci lacks: thal"):
        observe_records(frame((1, 0, 100)), [thal], name="uci")


def test_a_patient_shows_a_concept_by_the_snapshot_date_while_alive(tmp_path):
    synthea_tables(tmp_path, *SMALL_POPULATION)
    people = load_population(tmp_path, snapshot_date=date(2026, 1, 1))
    observed = observe_population(people, [ISCHEMIA], name="synthea")
    assert (observed.records, observed.holders) == (4, {"MONDO:9000003": 3})


def test_a_condition_after_the_snapshot_does_not_count(tmp_path):
    synthea_tables(
        tmp_path,
        [patient("p1", "1960-03-01", "M")],
        [condition("p1", ISCHEMIC_HEART_DISEASE, "2027-01-01")],
    )
    people = load_population(tmp_path, snapshot_date=date(2026, 1, 1))
    assert observe_population(people, [ISCHEMIA], name="synthea").holders == {"MONDO:9000003": 0}


def test_a_concept_two_codes_map_counts_a_patient_with_either(tmp_path):
    synthea_tables(tmp_path, *SMALL_POPULATION)
    people = load_population(tmp_path, snapshot_date=date(2026, 1, 1))
    infarction = CodeMapping(
        snomed="22298006", concept=ISCHEMIA.concept, label=ISCHEMIA.label, source="mondo"
    )
    observed = observe_population(people, [ISCHEMIA, infarction], name="synthea")
    assert observed.holders == {"MONDO:9000003": 4}


def test_a_population_with_no_living_patient_is_refused(tmp_path):
    synthea_tables(tmp_path, [patient("p1", "1960-03-01", "M", death="2010-01-01")], [])
    people = load_population(tmp_path, snapshot_date=date(2026, 1, 1))
    with pytest.raises(GapError, match="synthea holds no record"):
        observe_population(people, [ISCHEMIA], name="synthea")


def test_a_missing_value_shows_no_concept():
    frame = pd.DataFrame({"cp": [1.0, float("nan")], "target": [0, 0], "chol": [250.0, None]})
    observed = observe_records(frame, [ANGINA, CHOL], name="uci")
    assert observed.holders == {"HP:9000003": 1}


def test_holders_are_a_read_only_copy_and_observed_hashes():
    holders = {"A:1": 1}
    observed = Observed(name="uci", records=1, holders=holders)
    holders["A:1"] = 5
    assert observed.holders == {"A:1": 1}
    with pytest.raises(TypeError):
        observed.holders["A:1"] = 2  # ty: ignore[invalid-assignment]
    assert hash(observed) == hash(observed)


def test_a_concept_left_out_of_held_gets_no_prior():
    observed = Observed(name="uci", records=4, holders={"A:1": 2, "B:1": 1, "C:1": 0})
    fragment = dataset_fragment(observed, held={"B:1", "C:1"})
    assert [(s.head[0], s.prob) for s in fragment.snodes] == [("B:1", 0.25)]


UCI_GRID = [
    (band, sex, label) for band in ("<55", ">=55") for sex in ("female", "male") for label in (0, 1)
]


def uci_frame(*rows: tuple[int, int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["age", "sex", "target", "cp"], dtype="int64")


def test_record_cells_observe_every_cell_of_the_grid_and_the_whole():
    cells = record_cells(
        uci_frame((40, 0, 0, 1), (40, 0, 0, 4), (60, 1, 1, 1), (60, 1, 1, 4), (60, 1, 1, 4)),
        [ANGINA, STENOSIS],
        name="uci",
    )
    assert (cells.whole.records, cells.whole.holders) == (
        5,
        {"HP:9000003": 2, "MONDO:9000004": 3},
    )
    assert list(cells.each) == UCI_GRID
    seen = {
        cell: (records.observed.records, dict(records.observed.holders))
        for cell, records in cells.each.items()
    }
    assert seen.pop(("<55", "female", 0)) == (2, {"HP:9000003": 1, "MONDO:9000004": 0})
    assert seen.pop((">=55", "male", 1)) == (3, {"HP:9000003": 1, "MONDO:9000004": 3})
    assert all(value == (0, {"HP:9000003": 0, "MONDO:9000004": 0}) for value in seen.values())
    assert cells.each["<55", "female", 0].observed.name == "uci/<55/female/0"
    assert [records.label for records in cells.each.values()] == [0, 1] * 4
    assumed = {cell: records.assumed for cell, records in cells.each.items()}
    assert assumed == {
        cell: frozenset({"MONDO:9000004"}) if cell[2] == 0 else frozenset() for cell in UCI_GRID
    }


@pytest.mark.parametrize(
    ("codes", "assumed_at"),
    [({1}, {0: True, 1: False}), ({0}, {0: False, 1: True}), ({0, 1}, {0: False, 1: False})],
    ids=["disease only", "no disease only", "either label"],
)
def test_a_cell_assumes_a_concept_only_rows_of_another_label_name(codes, assumed_at):
    row = FieldMapping(
        when=(Codes(LABEL, frozenset(codes)),),
        concept="MONDO:9000004",
        label="x",
        source="t",
        limits="t",
    )
    cells = record_cells(uci_frame((40, 0, 0, 4)), [row], name="uci")
    for cell, records in cells.each.items():
        assert ("MONDO:9000004" in records.assumed) is assumed_at[cell[2]], cell


def test_a_concept_one_row_maps_without_the_label_is_never_assumed():
    plain = FieldMapping(
        when=(Codes("cp", frozenset({1})),),
        concept="MONDO:9000004",
        label="x",
        source="t",
        limits="t",
    )
    cells = record_cells(uci_frame((40, 0, 0, 4)), [STENOSIS, plain], name="uci")
    assert all(records.assumed == frozenset() for records in cells.each.values())


def test_a_label_row_with_another_clause_still_assumes_its_concept():
    both = FieldMapping(
        when=(Codes(LABEL, frozenset({1})), Codes("cp", frozenset({1}))),
        concept="MONDO:9000004",
        label="x",
        source="t",
        limits="t",
    )
    cells = record_cells(uci_frame((40, 0, 0, 4)), [both], name="uci")
    assert cells.each["<55", "female", 0].assumed == frozenset({"MONDO:9000004"})


def test_record_cells_pair_records_by_position_whatever_the_index():
    records = uci_frame((40, 0, 0, 1), (60, 1, 1, 4))
    records.index = pd.Index([7, 7])
    cells = record_cells(records, [ANGINA], name="uci")
    assert cells.each["<55", "female", 0].observed.holders == {"HP:9000003": 1}
    assert cells.each[">=55", "male", 1].observed.holders == {"HP:9000003": 0}


def test_record_cells_refuse_an_empty_frame_and_a_frame_without_its_cell_columns():
    with pytest.raises(GapError, match="uci holds no record"):
        record_cells(uci_frame(), [ANGINA], name="uci")
    with pytest.raises(GapError, match="the cells read fields uci lacks: age, sex"):
        record_cells(frame((1, 0, 100)), [ANGINA], name="uci")


def test_population_cells_observe_each_age_sex_cell_of_the_living(tmp_path):
    synthea_tables(tmp_path, *SMALL_POPULATION)
    people = load_population(tmp_path, snapshot_date=date(2026, 1, 1))
    cells = population_cells(people, [ISCHEMIA], name="synthea", edges=(60,))
    assert (cells.whole.records, cells.whole.holders) == (4, {"MONDO:9000003": 3})
    assert list(cells.each) == [
        ("<60", "female"),
        ("<60", "male"),
        (">=60", "female"),
        (">=60", "male"),
    ]
    assert [
        (records.observed.records, records.observed.holders["MONDO:9000003"])
        for records in cells.each.values()
    ] == [(0, 0), (1, 0), (1, 1), (2, 2)]
    assert all(r.label is None and r.assumed == frozenset() for r in cells.each.values())


def test_population_cells_follow_the_grid_order_of_their_age_bands(tmp_path):
    synthea_tables(tmp_path, *SMALL_POPULATION)
    people = load_population(tmp_path, snapshot_date=date(2026, 1, 1))
    cells = population_cells(people, [ISCHEMIA], name="synthea", edges=(40, 50))
    assert [band for band, _ in cells.each][::2] == ["<40", "40-49", ">=50"]
