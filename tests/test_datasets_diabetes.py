"""Reading Diabetes 130 into frame records: header, codes, cohort, label, indicators and loading."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.cells import LABEL
from bko.datasets.diabetes import (
    A1C_CODES,
    AGE_BRACKETS,
    CHANGE_CODES,
    COLUMNS,
    COUNTS,
    DIABETES_MED_CODES,
    DRUG_CODES,
    DRUGS,
    EXCLUDED_DISCHARGES,
    GLU_CODES,
    RACE_CODES,
    excluded_discharges,
    frame_records,
    load_diabetes,
    parse_diabetic,
)
from bko.datasets.errors import ContractError
from tests._datasets_fixtures import plant
from tests._diabetes_fixtures import (
    IDS_EXCERPT,
    REAL_HEADER,
    REAL_LAYOUT,
    archive,
    diabetic_line,
    pin,
    text,
)


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

REPO = Path(__file__).resolve().parents[1]

INDICATORS: Mapping[str, tuple[str, ...]] = {"dx_250_4": ("250.4",), "dx_401": ("401",)}


def _records(*lines: str, indicators: Mapping[str, tuple[str, ...]] = INDICATORS) -> pd.DataFrame:
    return frame_records(parse_diabetic(text(*lines)), indicators=indicators)


def _patient(number: int, **fields: str) -> str:
    """Return patient `number`'s encounter `number`: each such line is its own first encounter."""
    return diabetic_line(encounter_id=str(number), patient_nbr=str(number), **fields)


# Header and parsing


def test_the_header_must_be_the_archives_fifty_columns_in_order():
    assert tuple(REAL_HEADER.split(",")) == COLUMNS
    assert len(COLUMNS) == 50
    raw = parse_diabetic(text(REAL_LAYOUT))
    assert list(raw.columns) == list(COLUMNS)
    assert raw.loc[0, "diag_1"] == "8"
    assert raw.loc[0, "diag_2"] == "250.43"
    columns = list(COLUMNS)
    columns[2], columns[3] = columns[3], columns[2]
    with pytest.raises(ContractError, match="header"):
        parse_diabetic(text(REAL_LAYOUT, header=",".join(columns)))
    with pytest.raises(ContractError, match="header"):
        parse_diabetic(text(header=REAL_HEADER + ",extra"))


def test_a_row_with_the_wrong_field_count_names_its_line():
    with pytest.raises(ContractError, match="line 3: 51 fields, expected 50"):
        parse_diabetic(text(diabetic_line(), diabetic_line() + ",x"))
    with pytest.raises(ContractError, match="line 2: 49 fields, expected 50"):
        parse_diabetic(text(diabetic_line().rpartition(",")[0]))


def test_none_in_the_lab_columns_is_a_state_not_missing():
    raw = parse_diabetic(text(_patient(1, A1Cresult="None", max_glu_serum="None")))
    assert raw.loc[0, "A1Cresult"] == "None"
    frame = frame_records(raw, indicators=INDICATORS)
    assert frame.loc[1, "A1Cresult"] == A1C_CODES["None"] == 0
    assert frame.loc[1, "max_glu_serum"] == GLU_CODES["None"] == 0
    assert not frame.isna().any().any()


_STATES = [
    *(("A1Cresult", state, code) for state, code in A1C_CODES.items()),
    *(("max_glu_serum", state, code) for state, code in GLU_CODES.items()),
    *((drug, state, code) for drug in DRUGS for state, code in DRUG_CODES.items()),
    *(("change", state, code) for state, code in CHANGE_CODES.items()),
    *(("diabetesMed", state, code) for state, code in DIABETES_MED_CODES.items()),
]


@pytest.mark.parametrize(("column", "state", "code"), _STATES)
def test_every_lab_and_drug_state_maps_and_an_unknown_state_is_refused(column, state, code):
    assert _records(_patient(1, **{column: state})).loc[1, column] == code
    with pytest.raises(ContractError, match=f"encounter 7: {column} is 'Sometimes'"):
        _records(_patient(7, **{column: "Sometimes"}))


# Demographics


def test_race_question_mark_is_code_0_not_recorded():
    assert RACE_CODES["?"] == 0
    races = list(RACE_CODES)
    frame = _records(*(_patient(n, race=race) for n, race in enumerate(races, start=1)))
    assert frame["race"].tolist() == [RACE_CODES[race] for race in races]
    with pytest.raises(ContractError, match="encounter 9: race is 'Martian'"):
        _records(_patient(9, race="Martian"))


def test_gender_unknown_invalid_rows_are_dropped_and_counted(caplog):
    caplog.set_level(logging.INFO, logger="bko.datasets.diabetes")
    frame = _records(
        _patient(1, gender="Female"),
        _patient(2, gender="Unknown/Invalid"),
        _patient(3, gender="Male"),
    )
    assert frame["sex"].tolist() == [0, 1]
    assert "unknown_gender=1 " in caplog.text
    with pytest.raises(ContractError, match="encounter 4: gender is 'Other'"):
        _records(_patient(4, gender="Other"))


def test_each_decade_bracket_maps_to_its_index():
    brackets = [f"[{10 * d}-{10 * (d + 1)})" for d in range(2, 10)]
    frame = _records(*(_patient(n, age=b) for n, b in enumerate(brackets, start=1)))
    assert frame["age"].tolist() == list(range(2, 10))
    assert {f"[{10 * d}-{10 * (d + 1)})": d for d in range(10)} == AGE_BRACKETS


@pytest.mark.parametrize("bracket", ["[100-110)", "[25-35)", "[20-30]", "25"])
def test_an_unlisted_bracket_is_refused(bracket):
    with pytest.raises(ContractError, match=re.escape(f"encounter 5: age is '{bracket}'")):
        _records(_patient(1), _patient(5, age=bracket))


def test_brackets_under_20_are_dropped(caplog):
    caplog.set_level(logging.INFO, logger="bko.datasets.diabetes")
    frame = _records(
        _patient(1, age="[0-10)"),
        _patient(2, age="[10-20)"),
        _patient(3, age="[20-30)"),
    )
    assert frame["age"].tolist() == [2]
    assert "under_20=2 " in caplog.text


# Cohort selection


def test_the_excluded_discharges_are_the_mappings_expired_and_hospice_rows(tmp_path):
    assert excluded_discharges(IDS_EXCERPT) == EXCLUDED_DISCHARGES
    assert frozenset({11, 13, 14, 19, 20, 21}) == EXCLUDED_DISCHARGES
    ids = (1, 7, 11, 12, 13, 14, 18, 19, 20, 21, 25)
    frame = _records(
        *(_patient(n, discharge_disposition_id=str(n), time_in_hospital=str(n)) for n in ids)
    )
    assert frame["time_in_hospital"].tolist() == [1, 7, 12, 18, 25]
    widened = IDS_EXCERPT.replace("12,Still patient", "12,Expired or still patient")
    data = archive(text(REAL_LAYOUT), ids=widened)
    source = pin(data, "uci-diabetes-130")
    plant(tmp_path, source, data)
    with pytest.raises(ContractError, match="12"):
        load_diabetes(tmp_path, indicators=INDICATORS, source=source)
    with pytest.raises(ContractError, match="discharge_disposition_id"):
        excluded_discharges("admission_type_id,description\r\n1,Emergency\r\n")


def test_a_patients_first_encounter_is_the_lowest_encounter_id_not_the_first_line():
    frame = _records(
        diabetic_line(encounter_id="900", patient_nbr="5", time_in_hospital="9"),
        diabetic_line(encounter_id="100", patient_nbr="5", time_in_hospital="1"),
        diabetic_line(encounter_id="500", patient_nbr="5", time_in_hospital="5"),
    )
    assert frame["time_in_hospital"].tolist() == [1]


def test_first_encounter_then_exclusion():
    frame = _records(
        diabetic_line(encounter_id="100", patient_nbr="5", discharge_disposition_id="11"),
        diabetic_line(encounter_id="200", patient_nbr="5", discharge_disposition_id="1"),
        diabetic_line(encounter_id="300", patient_nbr="6", time_in_hospital="6"),
        diabetic_line(encounter_id="400", patient_nbr="7", gender="Unknown/Invalid"),
        diabetic_line(encounter_id="500", patient_nbr="7", time_in_hospital="7"),
    )
    assert frame["time_in_hospital"].tolist() == [6]


# Label


def test_the_label_is_one_for_readmission_under_30_days():
    frame = _records(
        _patient(1, readmitted="<30"), _patient(2, readmitted=">30"), _patient(3, readmitted="NO")
    )
    assert frame[LABEL].tolist() == [1, 0, 0]


def test_an_unknown_readmitted_value_is_refused():
    with pytest.raises(ContractError, match="encounter 3: readmitted is 'YES'"):
        _records(_patient(1), _patient(3, readmitted="YES"))


# Diagnosis indicators


def test_an_indicator_is_set_when_any_diagnosis_falls_in_its_codes():
    indicators = {"heart_failure": ("428", "402.01"), "dx_401": ("401",)}
    frame = _records(
        _patient(1, diag_1="428.0", diag_2="?", diag_3="?"),
        _patient(2, diag_1="250", diag_2="402.01", diag_3="?"),
        _patient(3, diag_1="250", diag_2="402.1", diag_3="401.9"),
        _patient(4, diag_1="250", diag_2="276", diag_3="428"),
        indicators=indicators,
    )
    assert frame["heart_failure"].tolist() == [1, 1, 0, 1]
    assert frame["dx_401"].tolist() == [0, 0, 1, 0]


def test_a_missing_diagnosis_sets_nothing():
    frame = _records(_patient(1, diag_1="?", diag_2="?", diag_3="?"))
    assert frame[list(INDICATORS)].eq(0).all().all()


def test_v_and_e_codes_never_match_a_numeric_code():
    indicators = {"v45": ("45",), "e878": ("878",)}
    frame = _records(
        _patient(1, diag_1="V45", diag_2="E878", diag_3="V45.8"), indicators=indicators
    )
    assert frame[["v45", "e878"]].eq(0).all().all()
    with pytest.raises(ValueError, match="V45"):
        _records(_patient(1), indicators={"v45": ("V45",)})


@pytest.mark.parametrize(
    ("code", "token", "hit"),
    [
        ("250.4", "250.43", 1),
        ("250.4", "250.4", 1),
        ("250.4", "250.04", 0),
        ("250.4", "250", 0),
        ("250", "250.43", 1),
        ("25", "250", 0),
        ("40", "401", 0),
        ("401", "40", 0),
        ("8", "8", 1),
        ("008", "8", 1),
        ("8", "8.5", 1),
        ("8", "80", 0),
    ],
)
def test_a_code_matches_by_category_not_string_prefix(code, token, hit):
    frame = _records(_patient(1, diag_1=token, diag_2="?", diag_3="?"), indicators={"dx": (code,)})
    assert frame.loc[1, "dx"] == hit


@pytest.mark.parametrize("token", ["250.x", "", "2500", "250.123", "V4", "E87", "250."])
def test_a_malformed_token_is_refused_naming_its_encounter(token):
    with pytest.raises(ContractError, match=re.escape(f"encounter 777: diag_3 is '{token}'")):
        _records(_patient(1), _patient(777, diag_3=token))


# Output shape and loading


def test_identifier_columns_never_enter_the_frame():
    frame = _records(REAL_LAYOUT)
    assert list(frame.columns) == [
        "age",
        "sex",
        "race",
        *COUNTS,
        "A1Cresult",
        "max_glu_serum",
        *DRUGS,
        "change",
        "diabetesMed",
        *INDICATORS,
        LABEL,
    ]
    assert "encounter_id" not in frame.columns
    assert "patient_nbr" not in frame.columns
    assert frame.index.name == "record_id"
    assert set(frame.dtypes.astype(str)) == {"int64"}
    record = frame.loc[1]
    assert (record["age"], record["sex"], record["dx_250_4"], record["insulin"]) == (3, 1, 1, 2)


def test_record_ids_are_positions_in_encounter_order():
    frame = _records(
        _patient(30, time_in_hospital="3"),
        _patient(10, time_in_hospital="1"),
        _patient(20, time_in_hospital="2"),
    )
    assert frame.index.tolist() == [1, 2, 3]
    assert frame["time_in_hospital"].tolist() == [1, 2, 3]


def test_the_loader_accounts_for_every_row(caplog):
    caplog.set_level(logging.INFO, logger="bko.datasets.diabetes")
    frame = _records(
        diabetic_line(encounter_id="1", patient_nbr="1", readmitted="<30"),
        diabetic_line(encounter_id="2", patient_nbr="1"),
        diabetic_line(encounter_id="3", patient_nbr="2", discharge_disposition_id="13"),
        diabetic_line(encounter_id="4", patient_nbr="3", gender="Unknown/Invalid"),
        diabetic_line(encounter_id="5", patient_nbr="4", age="[10-20)"),
        diabetic_line(encounter_id="6", patient_nbr="5"),
    )
    assert len(frame) == 2
    assert (
        "cohort encounters=6 repeat_encounters=1 excluded_discharges=1 unknown_gender=1 "
        "under_20=1 records=2 readmitted_under_30=1"
    ) in caplog.text


def _mixed_lines() -> list[str]:
    return [
        diabetic_line(
            encounter_id=str(100 + n),
            patient_nbr=str(n % 4),
            time_in_hospital=str(n),
            diag_3="401" if n % 3 else "250.41",
            readmitted="<30" if n % 2 else "NO",
        )
        for n in range(12)
    ]


def test_frame_bytes_do_not_depend_on_file_order_or_hash_seed(tmp_path):
    lines = _mixed_lines()
    expected = _records(*lines).to_csv()
    shuffled = [*lines[7::-1], *lines[:7:-1]]
    assert shuffled != lines
    assert _records(*shuffled).to_csv() == expected
    path = tmp_path / "diabetic_data.csv"
    path.write_text(text(*shuffled), encoding="utf-8")
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from bko.datasets.diabetes import frame_records, parse_diabetic\n"
        "raw = parse_diabetic(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "indicators = {'dx_250_4': ('250.4',), 'dx_401': ('401',)}\n"
        "sys.stdout.write(frame_records(raw, indicators=indicators).to_csv())\n"
    )
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO / "src")}
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-c", script, str(path)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        outputs.append(done.stdout)
    assert outputs == [expected, expected]


def test_load_reads_the_member_of_a_verified_snapshot(tmp_path):
    body = text(*_mixed_lines())
    data = archive(body)
    source = pin(data, "uci-diabetes-130")
    plant(tmp_path, source, data)
    frame = load_diabetes(tmp_path, indicators=INDICATORS, source=source)
    pd.testing.assert_frame_equal(frame, _records(*_mixed_lines()))
    tampered = tmp_path / "tampered"
    plant(tampered, source, archive(text(REAL_LAYOUT)))
    with pytest.raises(ContractError, match=r"sha256|size"):
        load_diabetes(tampered, indicators=INDICATORS, source=source)
    stray = tmp_path / "stray"
    lone = archive(body).replace(b"IDS_mapping.csv", b"IDS_mapping.txt")
    lone_source = pin(lone, "uci-diabetes-130")
    plant(stray, lone_source, lone)
    with pytest.raises(ContractError, match=re.escape("IDS_mapping.csv")):
        load_diabetes(stray, indicators=INDICATORS, source=lone_source)
