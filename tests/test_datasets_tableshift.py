"""TableShift's readmission partition and frame over an invented Diabetes 130 archive.

ADR `research/tableshift-comparison` records the task, the partition and the frame. The archive is
`tests._tableshift_fixtures`'; `tests/test_datasets_tableshift_pinned.py` checks the pinned one.
"""

from __future__ import annotations

import dataclasses
import logging
import math

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


def test_the_label_is_any_readmission():
    kept = kept_rows(_raw())
    records = encode(kept)
    expected = (kept["readmitted"] != "NO").astype("int64")
    assert records[LABEL].tolist() == expected.tolist()
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
    positions = np.arange(10, 17)
    rest, held = shuffle_split(positions, 0.3, 5)
    order = np.random.RandomState(5).permutation(7)
    count = math.ceil(0.3 * 7)
    assert held.tolist() == positions[order[:count]].tolist()
    assert rest.tolist() == positions[order[count:]].tolist()


def test_sex_is_male_against_every_other_value():
    kept = kept_rows(_raw())
    records = encode(kept)
    assert "Unknown/Invalid" in set(kept["gender"])
    assert records["sex"].tolist() == (kept["gender"] == "Male").astype("int64").tolist()
    assert tableshift_frame(kept).codebook["sex"] == SEXES


def test_race_is_caucasian_against_every_other_value():
    kept = kept_rows(_raw())
    records = encode(kept)
    assert records["race"].tolist() == (kept["race"] == "Caucasian").astype("int64").tolist()


def test_the_frame_holds_46_features_and_no_domain_column():
    kept = kept_rows(_raw())
    spec = tableshift_frame(kept)
    records = encode(kept)
    assert len(FEATURES) == 46
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


def test_every_lab_field_rows_field_and_codes_are_in_the_frames_codebook():
    spec = tableshift_frame(kept_rows(_raw()))
    for row in LAB_FIELD_ROWS:
        for clause in row.when:
            assert isinstance(clause, Codes)
            assert clause.field in spec.categorical
            assert clause.codes <= set(spec.codebook[clause.field])


def test_a_drug_state_outside_the_codes_is_refused():
    raw = parse_diabetic(text(line(1), line(2, nateglinide="Hi")))
    with pytest.raises(ContractError, match="not one of"):
        kept_rows(raw)


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
