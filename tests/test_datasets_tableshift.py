"""TableShift's readmission partition and frame over an invented Diabetes 130 archive.

ADR `research/tableshift-comparison` records the task, the partition and the frame. The archive is
`tests._tableshift_fixtures`'; `tests/test_datasets_tableshift_pinned.py` checks the pinned one.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Final

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from bko.datasets.cells import LABEL, SEXES
from bko.datasets.diabetes import COLUMNS, parse_diabetic
from bko.datasets.errors import ContractError
from bko.datasets.frames import check_records
from bko.datasets.tableshift import (
    OOD_SOURCE,
    SEED,
    kept_rows,
    load_tableshift,
    partition,
    shuffle_split,
)
from bko.datasets.tableshift_frame import FEATURES, encode, tableshift_frame
from bko.ontology.frames import LAB_FIELD_ROWS
from bko.ontology.schema import Codes
from tests._datasets_fixtures import pin, plant
from tests._diabetes_fixtures import archive, text
from tests._tableshift_fixtures import (
    IN_DOMAIN,
    KEPT,
    MISSING_RACE,
    line,
    tableshift_archive,
    tableshift_lines,
)


_FEATURES: Final = [
    "race",
    "sex",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]
"""TableShift's 46 features less the domain column, in the archive's order, `gender` as `sex`."""


def _raw() -> pd.DataFrame:
    return parse_diabetic(text(*tableshift_lines()))


def test_rows_with_missing_race_are_dropped():
    raw = _raw()
    kept = kept_rows(raw)
    assert len(raw) == KEPT + MISSING_RACE
    assert len(kept) == KEPT
    assert "?" not in set(kept["race"])
    positions = [i + 1 for i, race in enumerate(raw["race"]) if race != "?"]
    assert kept.index.tolist() == positions
    assert kept.index.name == "record_id"


_FIRST: Final = [1, 2, 3, 4, 5, 7]
"""The first kept record ids: rows 0 to 4 and 6 of the fixture, row 5's race being `?`."""


def test_the_label_is_any_readmission():
    kept = kept_rows(_raw())
    records = encode(kept)
    # readmitted NO, >30, <30, NO, >30, NO
    assert records.loc[_FIRST, LABEL].tolist() == [0, 1, 1, 0, 1, 0]
    assert set(kept.loc[records[LABEL] == 1, "readmitted"]) == {">30", "<30"}


def test_the_domains_split_on_admission_source_7():
    raw = _raw()
    kept = kept_rows(raw)
    parts = partition(raw)
    ood = set(kept.index[kept["admission_source_id"] == str(OOD_SOURCE)])
    assert set(parts.ood_validation) | set(parts.ood_test) == ood
    inside = set(parts.train) | set(parts.validation) | set(parts.test)
    assert inside == set(kept.index) - ood
    assert len(inside) == IN_DOMAIN


def test_the_partition_sizes_follow_the_ceil_rule():
    parts = partition(_raw())
    sizes = tuple(len(getattr(parts, field.name)) for field in dataclasses.fields(parts))
    assert sizes == (160, 20, 20, 5, 45)


def test_the_partition_membership_equals_scikit_learns():
    raw = _raw()
    kept = kept_rows(raw)
    ood = (kept["admission_source_id"] == str(OOD_SOURCE)).to_numpy()
    inside, outside = kept.index.to_numpy()[~ood], kept.index.to_numpy()[ood]
    train, held = train_test_split(inside, test_size=0.1 + 0.1, random_state=SEED)
    validation, test = train_test_split(held, test_size=0.1 / (0.1 + 0.1), random_state=SEED)
    ood_test, ood_validation = train_test_split(outside, test_size=0.1, random_state=SEED)
    parts = partition(raw)
    assert parts.train.tolist() == list(train)
    assert parts.validation.tolist() == list(validation)
    assert parts.test.tolist() == list(test)
    assert parts.ood_validation.tolist() == list(ood_validation)
    assert parts.ood_test.tolist() == list(ood_test)


def test_shuffle_split_holds_the_first_ceil_of_the_permutation():
    rest, held = shuffle_split(np.arange(10, 17), 0.3, 5)
    assert held.tolist() == [16, 12, 14]
    assert rest.tolist() == [11, 10, 15, 13]


def test_sex_is_male_against_every_other_value():
    kept = kept_rows(_raw())
    records = encode(kept)
    assert "Unknown/Invalid" in set(kept["gender"])
    # Unknown/Invalid, Male, Female, Male, Unknown/Invalid, Male
    assert records.loc[_FIRST, "sex"].tolist() == [0, 1, 0, 1, 0, 1]
    assert tableshift_frame(kept).codebook["sex"] == SEXES


def test_race_is_caucasian_against_every_other_value():
    kept = kept_rows(_raw())
    records = encode(kept)
    # Caucasian, AfricanAmerican, Hispanic, Other, Asian, AfricanAmerican
    assert records.loc[_FIRST, "race"].tolist() == [1, 0, 0, 0, 0, 0]


def test_the_frame_holds_46_features_and_no_domain_column():
    kept = kept_rows(_raw())
    spec = tableshift_frame(kept)
    records = encode(kept)
    assert list(FEATURES) == _FEATURES
    assert spec.features == FEATURES
    assert "admission_source_id" not in spec.columns
    assert {"encounter_id", "patient_nbr", "readmitted"}.isdisjoint(spec.columns)
    expected = {"sex" if name == "gender" else name for name in COLUMNS}
    expected -= {"encounter_id", "patient_nbr", "admission_source_id", "readmitted"}
    assert set(FEATURES) == expected
    assert spec.name == "tableshift"
    assert spec.edges == (1, 2, 4, 6, 8)
    assert tuple(records.columns) == spec.columns
    assert set(records.dtypes.astype(str)) == {"int64"}
    check_records(records, spec)


def test_the_text_fields_are_coded_by_sorted_position():
    kept = kept_rows(_raw())
    spec = tableshift_frame(kept)
    assert spec.codebook["weight"] == {0: "?", 1: "[50-75)", 2: "[75-100)"}
    assert spec.codebook["payer_code"] == {0: "?", 1: "HM", 2: "MC"}
    assert spec.codebook["medical_specialty"] == {
        0: "?",
        1: "Cardiology",
        2: "InternalMedicine",
        3: "Surgery-General",
    }
    # weight ?, [75-100), [50-75), ?, [75-100), ?
    assert encode(kept).loc[_FIRST, "weight"].tolist() == [0, 2, 1, 0, 2, 0]


def test_the_count_ranges_are_the_kept_rows_least_and_greatest():
    spec = tableshift_frame(kept_rows(_raw()))
    found = {limit.field: (limit.low, limit.high) for limit in spec.ranges}
    assert found == {
        "age": (0, 9),
        "time_in_hospital": (1, 14),
        "num_lab_procedures": (1, 80),
        "num_procedures": (0, 6),
        "num_medications": (1, 40),
        "number_outpatient": (0, 3),
        "number_emergency": (0, 2),
        "number_inpatient": (0, 4),
        "number_diagnoses": (1, 9),
    }


def test_every_lab_field_rows_field_and_codes_are_in_the_frames_codebook():
    spec = tableshift_frame(kept_rows(_raw()))
    for row in LAB_FIELD_ROWS:
        for clause in row.when:
            assert isinstance(clause, Codes)
            assert clause.field in spec.categorical
            assert clause.codes <= set(spec.codebook[clause.field])


def test_a_drug_state_outside_the_codes_is_refused():
    raw = parse_diabetic(text(line(1), line(2, nateglinide="Hi")))
    with pytest.raises(ContractError, match="encounter 9022: nateglinide is 'Hi', not one of"):
        kept_rows(raw)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"gender": "X"}, "encounter 9022: gender is 'X', not one of"),
        ({"metformin": "Hi"}, "encounter 9022: metformin is 'Hi', not one of"),
        ({"admission_type_id": "1.5"}, "encounter 9022: admission_type_id is '1.5', not a whole"),
        ({"admission_source_id": "x"}, "encounter 9022: admission_source_id is 'x', not a whole"),
    ],
)
def test_kept_rows_refuses_a_value_outside_its_schema_naming_the_encounter(change, message):
    raw = parse_diabetic(text(line(1), line(2, **change), line(3, **change)))
    with pytest.raises(ContractError, match=message):
        kept_rows(raw)


@pytest.mark.parametrize("domain", ["id", "ood"])
def test_load_refuses_an_empty_test_split(tmp_path, domain):
    data = archive(text(*(line(n, ood=domain == "id") for n in range(1, 5))))
    source = pin(data, "uci-diabetes-130")
    plant(tmp_path, source, data)
    part = "test" if domain == "id" else "ood_test"
    with pytest.raises(ContractError, match=f"TableShift's {part} split holds no record"):
        load_tableshift(tmp_path, source=source, domain=domain)


@pytest.mark.parametrize("domain", ["id", "ood"])
def test_load_reads_the_training_split_and_the_domains_test_held(tmp_path, caplog, domain):
    data = tableshift_archive()
    source = pin(data, "uci-diabetes-130")
    plant(tmp_path, source, data)
    caplog.set_level(logging.INFO, logger="bko.datasets.tableshift")
    records, held, spec = load_tableshift(tmp_path, source=source, domain=domain)
    parts = partition(_raw())
    test = parts.test if domain == "id" else parts.ood_test
    assert held == frozenset(test.tolist())
    assert set(records.index) == set(parts.train) | held
    assert records.index.is_monotonic_increasing
    assert tuple(records.columns) == spec.columns
    check_records(records, spec)
    assert (
        "tableshift partition commit=fca9429814703a07e3902d005d46563a207b7f0a seed=264738 "
        "train=160 validation=20 test=20 ood_validation=5 ood_test=45 dropped_race=12"
    ) in caplog.messages


def test_an_archive_without_a_race_is_refused(tmp_path):
    data = archive(text(line(1, race="?")))
    source = pin(data, "uci-diabetes-130")
    plant(tmp_path, source, data)
    with pytest.raises(ContractError, match="no row"):
        load_tableshift(tmp_path, source=source, domain="id")
