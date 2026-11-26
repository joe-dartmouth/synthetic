"""`bko.datasets.physionet2012`: parsing stays and outcomes, the reduction rule and the cohort.

Every input is built in memory or in `tmp_path`; the one real stay is `REAL_STAY`.
"""

from __future__ import annotations

import logging
import operator
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.cells import LABEL
from bko.datasets.errors import ContractError
from bko.datasets.frames import check_records, digest
from bko.datasets.physionet2012 import (
    FEATURES,
    MEASURED,
    frame_records,
    load_physionet2012,
    parse_outcomes,
    parse_stay,
)
from bko.datasets.physionet2012_frame import physionet2012_frame
from bko.ontology.physionet2012_rows import PHYSIONET2012_FIELD_ROWS
from bko.ontology.schema import Codes, shown
from tests._physionet2012_fixtures import (
    REAL_ID,
    REAL_OUTCOME,
    REAL_STAY,
    archived,
    mixed,
    outcomes,
    planted,
    real_sets,
    stay,
)


def _record_id(text: str) -> str:
    """The `RecordID` a stay's text gives on its first line after the header."""
    return text.split("\n")[1].split(",")[2]


def records(*stays: tuple[str, int]) -> pd.DataFrame:
    """The frame records of each `(text, death)`, every stay a member of set A."""
    parsed = [parse_stay(f"set-a/{_record_id(text)}.txt", text) for text, _ in stays]
    deaths = {one.record_id: death for one, (_, death) in zip(parsed, stays, strict=True)}
    return frame_records(parsed, deaths)


def one(*observations: tuple[str, str, str], **descriptors: str) -> pd.Series:
    """The one record of a stay holding `observations`."""
    found = records((stay(1, *observations, **descriptors), 0))
    return found.iloc[0]


def test_the_real_stay_reads_as_its_literal_fields() -> None:
    record = records((REAL_STAY, 0)).loc[REAL_ID]
    assert record.to_dict() == {
        "age": 76,
        "sex": 1,
        "ICUType": 3,
        "hr_max": 2,
        "resprate_max": 0,
        "temp_max": 1,
        "temp_min": 1,
        "wbc_max": 1,
        "wbc_min": 2,
        "map_min": 1,
        "creatinine_max": 1,
        "bilirubin_max": 2,
        "platelets_min": 1,
        "gcs_min": 1,
        "mechvent": 1,
        LABEL: 0,
    }


def test_the_real_outcome_line_reads_as_its_death() -> None:
    assert parse_outcomes(outcomes() + REAL_OUTCOME + "\n", "o") == {REAL_ID: 0}


def test_each_variable_is_reduced_to_its_minimum_and_maximum() -> None:
    record = one(
        ("00:10", "HR", "88"),
        ("12:00", "HR", "91"),
        ("47:59", "HR", "90"),
        ("01:00", "Temp", "35.9"),
        ("02:00", "Temp", "37"),
        ("40:00", "Temp", "38.1"),
        ("03:00", "WBC", "12"),
        ("04:00", "WBC", "4"),
        ("05:00", "Platelets", "149"),
        ("06:00", "Platelets", "300"),
        ("07:00", "GCS", "15"),
        ("48:00", "GCS", "14"),
        ("08:00", "Creatinine", "1.1"),
        ("09:00", "Creatinine", "1.2"),
    )
    assert (record["hr_max"], record["temp_max"], record["temp_min"]) == (2, 2, 1)
    assert (record["wbc_max"], record["wbc_min"]) == (1, 2)
    assert (record["platelets_min"], record["gcs_min"], record["creatinine_max"]) == (1, 1, 2)


def test_each_coded_field_is_at_its_cutoffs() -> None:
    assert {name: (cut.at_or_above, cut.above) for name, cut in MEASURED.items()} == {
        "hr_max": ((), (90.0,)),
        "resprate_max": ((), (20.0,)),
        "temp_max": ((), (38.0,)),
        "temp_min": ((36.0,), ()),
        "wbc_max": ((), (12.0,)),
        "wbc_min": ((4.0,), ()),
        "map_min": ((70.0,), ()),
        "creatinine_max": ((1.2,), ()),
        "bilirubin_max": ((1.2,), ()),
        "platelets_min": ((150.0,), ()),
        "gcs_min": ((15.0,), ()),
    }
    assert {cut.source for cut in MEASURED.values()} == {"singer-2016"}
    at = one(("01:00", "HR", "90"), ("01:00", "RespRate", "20"), ("01:00", "MAP", "70"))
    assert (at["hr_max"], at["resprate_max"], at["map_min"]) == (1, 1, 2)
    past = one(("01:00", "HR", "90.5"), ("01:00", "RespRate", "21"), ("01:00", "MAP", "69.9"))
    assert (past["hr_max"], past["resprate_max"], past["map_min"]) == (2, 2, 1)


def test_minus_1_is_missing() -> None:
    record = one(("01:00", "HR", "-1"), ("02:00", "Temp", "-1"), ("03:00", "Temp", "37"))
    assert (record["hr_max"], record["temp_max"], record["temp_min"]) == (0, 1, 2)


def test_a_value_below_0_is_no_reading_and_is_counted(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.datasets.physionet2012"):
        record = one(("01:00", "Temp", "-17.8"), ("02:00", "Temp", "36.5"), ("03:00", "HR", "-1"))
    assert (record["temp_min"], record["hr_max"]) == (2, 0)
    assert " below_zero=2 " in caplog.messages[-1]


def test_a_recorded_0_is_a_reading() -> None:
    record = one(("01:00", "Temp", "0"), ("02:00", "MAP", "0"))
    assert (record["temp_min"], record["temp_max"], record["map_min"]) == (1, 1, 1)


def test_a_stay_without_a_variable_shows_it_not_measured() -> None:
    record = one(("01:00", "HR", "70"))
    assert record["hr_max"] == 1
    assert all(record[name] == 0 for name in MEASURED if name != "hr_max")
    assert record["mechvent"] == 0


def test_map_falls_back_to_the_non_invasive_reading() -> None:
    both = one(("01:00", "MAP", "75"), ("02:00", "NIMAP", "60"))
    assert both["map_min"] == 2
    cuff = one(("02:00", "NIMAP", "60"), ("03:00", "MAP", "-1"))
    assert cuff["map_min"] == 1
    neither = one(("02:00", "NISysABP", "60"))
    assert neither["map_min"] == 0


@pytest.mark.parametrize(
    ("values", "expected"),
    [((), 0), (("-1",), 0), (("0",), 0), (("0", "1"), 1), (("1",), 1)],
)
def test_mechvent_is_any_1(values: tuple[str, ...], expected: int) -> None:
    record = one(*((f"0{n}:00", "MechVent", value) for n, value in enumerate(values)))
    assert record["mechvent"] == expected


def test_age_is_as_published_and_gender_is_sex() -> None:
    record = one(age="90", gender="0", icu="4")
    assert (record["age"], record["sex"], record["ICUType"]) == (90, 0, 4)


@pytest.mark.parametrize("death", [0, 1])
def test_in_hospital_death_is_the_label(death: int) -> None:
    assert records((stay(1), death)).loc[1, LABEL] == death


@pytest.mark.parametrize(
    "unknown", [{"age": "-1"}, {"gender": "-1"}, {"age": "-1", "gender": "-1"}]
)
def test_a_stay_with_unknown_age_or_gender_is_left_out_and_counted(unknown, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.datasets.physionet2012"):
        found = records((stay(1, **unknown), 1), (stay(2), 0))
    assert list(found.index) == [2]
    ages, sexes = int("age" in unknown), int("gender" in unknown)
    assert caplog.messages[-1].startswith(
        f"cohort dataset=physionet2012 stays=2 unknown_age={ages} unknown_sex={sexes} "
        "left_out=1 records=1 died=0 unnamed=0 below_zero=0 "
    )


def test_an_unnamed_line_is_skipped_and_counted(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.datasets.physionet2012"):
        record = one(("01:00", "", "2.9"), ("02:00", "", "3.3"), ("03:00", "HR", "95"))
    assert record["hr_max"] == 2
    assert " unnamed=2 " in caplog.messages[-1]


def test_the_cohort_line_counts_measured_fields_and_ventilation(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.datasets.physionet2012"):
        records((stay(1, ("01:00", "HR", "70"), ("02:00", "MechVent", "1")), 0), (stay(2), 1))
    message = caplog.messages[-1]
    assert " measured_hr_max=1 measured_resprate_max=0 " in message
    assert message.endswith(" measured_gcs_min=0 mechvent=1")


def test_the_header_must_be_the_published_one() -> None:
    text = stay(1).replace("Time,Parameter,Value", "Time,Value,Parameter", 1)
    with pytest.raises(
        ContractError, match=r"set-a/1\.txt: the header is not Time,Parameter,Value"
    ):
        parse_stay("set-a/1.txt", text)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("01:00,HR", r"line 8: 2 fields, not 3"),
        ("01:00,HR,70,1", r"line 8: 4 fields, not 3"),
        ("48:01,HR,70", r"line 8: time '48:01' is not hh:mm from 00:00 to 48:00"),
        ("1:00,HR,70", r"line 8: time '1:00' is not hh:mm"),
        ("01:60,HR,70", r"line 8: time '01:60' is not hh:mm"),
        ("01:00,HR,", r"line 8: value '' is not a number"),
        ("01:00,HR,nan", r"line 8: value 'nan' is not finite"),
        ("01:00,Heart,70", r"line 8: 'Heart' is no descriptor or time series"),
        ("01:00,Age,70", r"line 8: Age must appear once, at 00:00"),
        ("00:00,Gender,0", r"line 8: Gender must appear once, at 00:00"),
        ("01:00,MechVent,2", r"line 8: MechVent is 2, not 0, 1 or -1"),
    ],
)
def test_a_malformed_line_names_its_line(line: str, match: str) -> None:
    with pytest.raises(ContractError, match=rf"set-a/1\.txt {match}"):
        parse_stay("set-a/1.txt", stay(1) + line + "\n")


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (operator.itemgetter(slice(-1)), "does not end in a newline"),
        (lambda text: text.replace("00:00,ICUType,3\n", ""), "no ICUType at 00:00"),
        (
            lambda text: text.replace("RecordID,1\n", "RecordID,2\n"),
            "RecordID is 2, not the name's 1",
        ),
        (lambda text: text.replace("Age,60\n", "Age,60.5\n"), "Age is 60.5, not a whole age or -1"),
        (lambda text: text.replace("Age,60\n", "Age,-2\n"), "Age is -2, not a whole age or -1"),
        (lambda text: text.replace("Gender,1\n", "Gender,2\n"), "Gender is 2, not 0, 1 or -1"),
        (lambda text: text.replace("ICUType,3\n", "ICUType,-1\n"), "ICUType is -1, not 1 to 4"),
    ],
)
def test_a_malformed_stay_is_refused(edit, match: str) -> None:
    with pytest.raises(ContractError, match=rf"set-a/1\.txt: {match}"):
        parse_stay("set-a/1.txt", edit(stay(1)))


def test_a_member_that_is_not_a_stays_name_is_refused() -> None:
    with pytest.raises(ContractError, match=r"notes\.txt: not a stay's name"):
        parse_stay("notes.txt", stay(1))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (outcomes((1, 0))[:-1], "o: does not end in a newline"),
        (outcomes((1, 0)).replace("SAPS-I", "SAPS"), "o: the header is not RecordID,SAPS-I"),
        (outcomes((1, 0)).replace(",0\n", ",2\n"), "o line 2: not a RecordID"),
        (outcomes((1, 0)).replace("1,12", "x,12"), "o line 2: not a RecordID"),
        (outcomes((1, 0)).replace(",-1,", ","), "o line 2: not a RecordID"),
        (outcomes((1, 0), (1, 1)), "o line 3: RecordID 1 is listed more than once"),
    ],
)
def test_a_malformed_outcomes_file_is_refused(text: str, match: str) -> None:
    with pytest.raises(ContractError, match=match):
        parse_outcomes(text, "o")


def test_the_severity_scores_never_enter_the_frame() -> None:
    severe = outcomes((1, 0)).replace("1,12,4,9,-1,0", "1,80,24,90,30,0")
    assert parse_outcomes(severe, "o") == parse_outcomes(outcomes((1, 0)), "o") == {1: 0}
    found = records((stay(1), 0))
    for name in ("SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"):
        assert name not in found.columns


def test_the_excluded_columns_never_enter_the_frame() -> None:
    found = records((REAL_STAY, 0))
    assert tuple(found.columns) == (*FEATURES, LABEL)
    for name in ("Height", "Weight", "Lactate", "PaCO2", "FiO2", "NIMAP", "MAP", "HR", "GCS"):
        assert name not in found.columns


def test_load_reads_every_set_and_its_outcomes(tmp_path: Path) -> None:
    found = load_physionet2012(tmp_path, sources=planted(tmp_path, real_sets()))
    assert len(found) == 13
    assert found.index.is_monotonic_increasing
    assert int(found[LABEL].sum()) == 3
    assert found.loc[REAL_ID, "bilirubin_max"] == 2
    assert found.loc[140011, "ICUType"] == 4
    assert found.dtypes.unique().tolist() == ["int64"]


def test_the_frame_does_not_depend_on_member_order(tmp_path: Path) -> None:
    sets = mixed()
    first = load_physionet2012(tmp_path / "1", sources=planted(tmp_path / "1", sets))
    reversed_sets = {label: dict(reversed(held.items())) for label, held in sets.items()}
    again = load_physionet2012(tmp_path / "2", sources=planted(tmp_path / "2", reversed_sets))
    assert digest(first) == digest(again)


def test_a_stay_with_no_outcome_row_is_refused(tmp_path: Path) -> None:
    sets = mixed()
    texts = {"b": outcomes(*((rid, d) for rid, (_, d) in list(sets["b"].items())[1:]))}
    sources = planted(tmp_path, sets, outcome_texts=texts)
    with pytest.raises(ContractError, match="set b: stay 140004 has no outcome line"):
        load_physionet2012(tmp_path, sources=sources)


def test_an_outcome_with_no_stay_is_refused(tmp_path: Path) -> None:
    sets = mixed()
    rows = [(rid, d) for rid, (_, d) in sets["c"].items()] + [(150000, 1)]
    sources = planted(tmp_path, sets, outcome_texts={"c": outcomes(*rows)})
    with pytest.raises(ContractError, match="set c: the outcome line of 150000 names no stay"):
        load_physionet2012(tmp_path, sources=sources)


def test_a_record_id_repeated_across_sets_is_refused(tmp_path: Path) -> None:
    sets = mixed()
    sets["b"] = {**sets["b"], 140000: sets["a"][140000]}
    with pytest.raises(ContractError, match="set b: RecordID 140000 is a stay of an earlier set"):
        load_physionet2012(tmp_path, sources=planted(tmp_path, sets))


def test_load_takes_six_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reads 6 files, not 5"):
        load_physionet2012(tmp_path, sources=planted(tmp_path)[:5])


def _truncated(label: str) -> bytes:
    data = archived(label, {rid: text for rid, (text, _) in mixed()[label].items()})
    return data[: len(data) // 2]


@pytest.mark.parametrize(
    ("label", "data", "match"),
    [
        ("a", b"not a zip archive", "physionet2012-set-a: cannot read the archive: BadZipFile"),
        ("b", None, "physionet2012-set-b: cannot read the archive: BadZipFile"),
        ("c", None, r"physionet2012-set-c: cannot read the archive: \w+"),
        ("c", b"\x1f\x8b not gzip", r"physionet2012-set-c: cannot read the archive: \w+"),
        ("a", archived("c", {1: stay(1)}), "physionet2012-set-a: cannot read the archive"),
        ("c", archived("a", {1: stay(1)}), "physionet2012-set-c: cannot read the archive"),
    ],
    ids=["not-a-zip", "truncated-zip", "truncated-tar", "bad-gzip", "tar-as-a", "zip-as-c"],
)
def test_a_damaged_archive_is_a_contract_error(tmp_path, label, data, match) -> None:
    sources = planted(tmp_path, archives={label: _truncated(label) if data is None else data})
    with pytest.raises(ContractError, match=match):
        load_physionet2012(tmp_path, sources=sources)


def test_a_member_of_another_set_is_refused(tmp_path: Path) -> None:
    sources = planted(tmp_path, archives={"b": archived("a", {1: stay(1)})})
    with pytest.raises(ContractError, match=r"set-a/1\.txt: outside the pattern set-b/"):
        load_physionet2012(tmp_path, sources=sources)


def test_an_outcomes_file_that_is_not_ascii_is_refused(tmp_path: Path) -> None:
    sources = planted(tmp_path, outcome_texts={"a": "RecordIDé\n"})
    with pytest.raises(ContractError, match="physionet2012-outcomes-a: not ASCII text"):
        load_physionet2012(tmp_path, sources=sources)


@pytest.mark.parametrize("label", ["a", "c"], ids=["zip", "tar"])
def test_a_member_over_the_cap_is_refused_before_it_is_read(tmp_path, monkeypatch, label) -> None:
    held = mixed()[label]
    sources = planted(tmp_path, {label: held})
    monkeypatch.setattr("bko.datasets.archive.MEMBER_LIMIT", 100)
    match = rf"physionet2012-set-{label}: set-{label}/14000\d\.txt: \d+ bytes, over the 100-byte"
    with pytest.raises(ContractError, match=match):
        load_physionet2012(tmp_path, sources=sources)


@pytest.mark.parametrize(
    ("sets", "match"),
    [
        ({label: {} for label in "abc"}, "the cohort holds no records"),
        (
            {"a": {1: (stay(1, age="-1"), 1)}, "b": {}, "c": {}},
            "the cohort holds no records",
        ),
        (
            {"a": {1: (stay(1), 0)}, "b": {2: (stay(2), 0)}, "c": {}},
            "the cohort holds no in-hospital death",
        ),
    ],
    ids=["no-stay", "all-left-out", "no-death"],
)
def test_load_refuses_a_cohort_it_cannot_score(tmp_path: Path, sets, match: str) -> None:
    with pytest.raises(ContractError, match=f"physionet2012: {match}"):
        load_physionet2012(tmp_path, sources=planted(tmp_path, sets))


def _shown(record: pd.Series) -> set[str]:
    """The labels of the concepts `record`'s field rows show."""
    labels = {row.concept: row.label for row in PHYSIONET2012_FIELD_ROWS}
    values = {str(name): float(value) for name, value in record.items()}
    return {labels[concept] for concept in shown(values, PHYSIONET2012_FIELD_ROWS)}


def test_a_missing_value_shows_no_concept() -> None:
    assert _shown(one()) == set()
    assert _shown(one(*(("01:00", name, "-1") for name in ("HR", "Temp", "WBC", "GCS")))) == set()


def test_each_abnormal_state_shows_its_concept() -> None:
    low = one(
        ("01:00", "Temp", "35"),
        ("01:00", "WBC", "3"),
        ("01:00", "NIMAP", "65"),
        ("01:00", "Platelets", "149"),
        ("01:00", "GCS", "14"),
    )
    assert _shown(low) == {
        "Hypothermia",
        "Decreased total leukocyte count",
        "Hypotension",
        "Thrombocytopenia",
        "Reduced consciousness",
    }
    high = one(
        ("01:00", "HR", "91"),
        ("01:00", "RespRate", "21"),
        ("01:00", "Temp", "38.5"),
        ("01:00", "WBC", "12.5"),
        ("01:00", "Creatinine", "1.2"),
        ("01:00", "Bilirubin", "1.2"),
    )
    assert _shown(high) == {
        "Tachycardia",
        "Tachypnea",
        "Fever",
        "Increased total leukocyte count",
        "Elevated circulating creatinine concentration",
        "Hyperbilirubinemia",
    }
    normal = one(
        ("01:00", "HR", "90"),
        ("01:00", "RespRate", "20"),
        ("01:00", "Temp", "36"),
        ("02:00", "Temp", "38"),
        ("01:00", "WBC", "4"),
        ("02:00", "WBC", "12"),
        ("01:00", "MAP", "70"),
        ("01:00", "Creatinine", "1.19"),
        ("01:00", "Bilirubin", "1.1"),
        ("01:00", "Platelets", "150"),
        ("01:00", "GCS", "15"),
        ("01:00", "MechVent", "1"),
    )
    assert _shown(normal) == set()


def test_the_frame_holds_exactly_the_named_fields() -> None:
    found = records((REAL_STAY, 0), (stay(2, age="15", gender="0"), 1))
    frame = physionet2012_frame(found)
    assert frame.columns == (
        *("age", "sex", "ICUType", "hr_max", "resprate_max", "temp_max", "temp_min"),
        *("wbc_max", "wbc_min", "map_min", "creatinine_max", "bilirubin_max"),
        *("platelets_min", "gcs_min", "mechvent", "target"),
    )
    assert frame.numeric == ("age",)
    assert frame.edges == (50, 65, 80)
    assert (frame.ranges[0].low, frame.ranges[0].high) == (15, 76)
    assert tuple(found.columns) == frame.columns
    check_records(found, frame)
    assert dict(frame.codebook["ICUType"]) == {
        1: "Coronary Care Unit",
        2: "Cardiac Surgery Recovery Unit",
        3: "Medical ICU",
        4: "Surgical ICU",
    }
    assert dict(frame.codebook["hr_max"]) == {0: "not measured", 1: "at most 90", 2: "above 90"}
    assert dict(frame.codebook["temp_min"]) == {0: "not measured", 1: "below 36", 2: "at least 36"}
    assert dict(frame.codebook["mechvent"]) == {0: "no ventilation recorded", 1: "ventilated"}
    assert dict(frame.codebook["sex"]) == {0: "female", 1: "male"}


def test_the_label_codes_are_0_and_1() -> None:
    found = records((stay(1), 1), (stay(2), 0))
    assert found[LABEL].to_dict() == {1: 1, 2: 0}
    frame = physionet2012_frame(found)
    assert (frame.label_codes, dict(frame.label_meanings)) == (
        (0, 1),
        {0: "survived the hospital stay", 1: "died in hospital"},
    )


def test_every_field_row_reads_a_frame_column_within_its_codebook() -> None:
    frame = physionet2012_frame(records((REAL_STAY, 0)))
    fields = []
    for row in PHYSIONET2012_FIELD_ROWS:
        [clause] = row.when
        assert isinstance(clause, Codes)
        assert clause.field in frame.columns
        assert clause.codes <= frame.codebook[clause.field].keys() - {0}, row
        fields.append(clause.field)
    assert sorted(fields) == sorted(MEASURED)
