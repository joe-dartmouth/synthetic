"""NHANES's loader and frame over XPORT files built in memory; no test reads `data/`."""

from __future__ import annotations

import io
import logging
import operator
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.errors import ContractError
from bko.datasets.frames import check_records, digest
from bko.datasets.nhanes import (
    COLUMNS,
    COMPONENTS,
    READINGS,
    check_readings,
    frame_records,
    load_nhanes,
    read_component,
)
from bko.datasets.nhanes_frame import nhanes_frame
from bko.datasets.sources import Source
from bko.ontology.nhanes_rows import NHANES_FIELD_ROWS
from bko.ontology.schema import shown
from tests._datasets_fixtures import pin, plant
from tests._nhanes_fixtures import (
    REAL_COLUMNS,
    REAL_ROWS,
    Respondent,
    components,
    files,
    mixed,
    xpt_bytes,
)


MIXED_DIGEST = "30723c2794bed7d62c105ebbefdf760484153f99973bf141ed24d2e8dc5488b0"
"""The sha256 of `mixed()`'s records as `bko.datasets.frames.digest` renders them."""

EXAM = ("BPXOSY", "BPXODI", "LBXGH", "LBXTC", "LBDHDD", "BMXBMI")
"""The coded exam and laboratory fields, in record order."""


def _records(people: list[Respondent]) -> pd.DataFrame:
    read = {name: read_component(data, name) for name, data in files(people).items()}
    return frame_records(read)


def _values(records: pd.DataFrame, record: int, fields: tuple[str, ...]) -> list[int]:
    return records.loc[[record], list(fields)].to_numpy().tolist()[0]


def _planted(tmp_path: Path, data: dict[str, bytes]) -> tuple[Source, ...]:
    sources = []
    for name in COMPONENTS:
        source = pin(data[name], f"nhanes-{name.lower()}")
        plant(tmp_path, source, data[name])
        sources.append(source)
    return tuple(sources)


def test_the_fixture_writer_round_trips_through_read_sas() -> None:
    for name, frame in components(mixed()).items():
        back = pd.read_sas(io.BytesIO(xpt_bytes(frame, name)), format="xport")
        assert isinstance(back, pd.DataFrame)
        assert tuple(back.columns) == REAL_COLUMNS[name]
        expected = frame.map(lambda v: float("nan") if v is None else v)
        pd.testing.assert_frame_equal(back, expected, check_dtype=False)
    real = pd.read_sas(
        io.BytesIO(xpt_bytes(components([Respondent(130378)])["GHB_L"])), format="xport"
    )
    assert isinstance(real, pd.DataFrame)
    assert tuple(real.iloc[0]) == (130378.0, 56042.129409583235, 5.6)


def test_the_published_columns_are_the_fixtures_real_ones() -> None:
    assert dict(COLUMNS) == dict(REAL_COLUMNS)
    assert tuple(COLUMNS) == COMPONENTS
    assert REAL_COLUMNS["MCQ_L"][10:15] == ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F")


def test_components_join_on_the_respondent_sequence_number() -> None:
    people = [
        Respondent(200002, age=30.0, hba1c=6.5, systolic=(140.0, 140.0, 140.0)),
        Respondent(200001, age=50.0, hba1c=5.0, systolic=(110.0, 110.0, 110.0)),
    ]
    data = components(people)
    data["GHB_L"] = data["GHB_L"].iloc[::-1]
    read = {name: read_component(xpt_bytes(frame, name), name) for name, frame in data.items()}
    records = frame_records(read)
    assert records.index.tolist() == [200001, 200002]
    assert records.index.name == "record_id"
    assert _values(records, 200001, ("age", "LBXGH", "BPXOSY")) == [50, 1, 1]
    assert _values(records, 200002, ("age", "LBXGH", "BPXOSY")) == [30, 3, 2]


def test_a_repeated_respondent_number_is_refused() -> None:
    data = components([Respondent(1), Respondent(2)])
    twice = pd.concat([data["MCQ_L"], data["MCQ_L"].iloc[[0]]])
    with pytest.raises(ContractError, match=r"^MCQ_L: respondent 1 is listed more than once$"):
        read_component(xpt_bytes(twice), "MCQ_L")


def test_a_respondent_number_that_is_not_whole_names_its_row() -> None:
    data = components([Respondent(1), Respondent(2)])["DEMO_L"]
    data.loc[1, "SEQN"] = 2.5
    with pytest.raises(ContractError, match=r"^DEMO_L row 2: SEQN 2\.5 is not whole$"):
        read_component(xpt_bytes(data), "DEMO_L")


def test_a_missing_respondent_number_names_its_row() -> None:
    data = components([Respondent(1), Respondent(2)])["GHB_L"]
    data.loc[1, "SEQN"] = None
    with pytest.raises(ContractError, match=r"^GHB_L row 2: SEQN nan is not whole$"):
        read_component(xpt_bytes(data), "GHB_L")


@pytest.mark.parametrize("component", ["MCQ_L", "BPXO_L", "GHB_L"])
def test_a_respondent_absent_from_the_demographics_is_refused(component: str) -> None:
    data = components([Respondent(1), Respondent(2), Respondent(3)])
    read = {name: read_component(xpt_bytes(frame, name), name) for name, frame in data.items()}
    read["DEMO_L"] = read["DEMO_L"].loc[[1]]
    match = rf"^{component}: 2 respondents are absent from DEMO_L, first 2$"
    with pytest.raises(ContractError, match=match):
        frame_records({**read, **{n: read[n].loc[[1]] for n in COMPONENTS if n != component}})


def test_every_reading_is_a_column_of_its_component() -> None:
    check_readings(READINGS, COLUMNS)
    with pytest.raises(ValueError, match=r"^LBXGH: GHB_L holds no column LBXGHX$"):
        check_readings({**READINGS, "LBXGH": ("GHB_L", "value", ("LBXGHX",))}, COLUMNS)
    with pytest.raises(ValueError, match=r"^LBXGH: GHB_M is not a component file$"):
        check_readings({**READINGS, "LBXGH": ("GHB_M", "value", ("LBXGH",))}, COLUMNS)


def test_only_adults_of_20_and_over_enter() -> None:
    records = _records([Respondent(1, age=19.0), Respondent(2, age=20.0), Respondent(3, age=0.0)])
    assert records.index.tolist() == [2]
    assert records.loc[2, "age"] == 20


def test_age_80_stays_80() -> None:
    records = _records([Respondent(1, age=80.0), Respondent(2, age=79.0)])
    assert records["age"].tolist() == [80, 79]
    assert nhanes_frame(records).ranges[0].high == 80


@pytest.mark.parametrize("item", range(4))
def test_any_yes_is_label_1_and_four_noes_are_label_0(item: int) -> None:
    answers = [9.0, 7.0, None, 2.0]
    answers[item] = 1.0
    records = _records([Respondent(1, answers=tuple(answers)), Respondent(2)])
    assert records["target"].to_dict() == {1: 1, 2: 0}


@pytest.mark.parametrize("unknown", [7.0, 9.0, None], ids=["refused", "dont-know", "missing"])
def test_a_record_with_no_yes_and_an_unknown_answer_is_left_out(unknown, caplog) -> None:
    people = [Respondent(1, answers=(2.0, 2.0, unknown, 2.0)), Respondent(2), Respondent(3)]
    with caplog.at_level(logging.INFO, logger="bko.datasets.nhanes"):
        records = _records(people)
    assert records.index.tolist() == [2, 3]
    assert caplog.messages == [
        (
            "cohort dataset=nhanes demographics=3 with_conditions=3 adults=3 any_yes=0 "
            "all_no=2 left_out=1 records=2 composite=0 measured_BPXOSY=2 measured_BPXODI=2 "
            "measured_LBXGH=2 measured_LBXTC=2 measured_LBDHDD=2 measured_BMXBMI=2"
        )
    ]


def test_the_cohort_line_accounts_for_every_filter(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.datasets.nhanes"):
        _records([*mixed(), Respondent(130390, interviewed=False)])
    assert caplog.messages == [
        (
            "cohort dataset=nhanes demographics=13 with_conditions=12 adults=11 any_yes=4 "
            "all_no=5 left_out=2 records=9 composite=4 measured_BPXOSY=7 measured_BPXODI=8 "
            "measured_LBXGH=8 measured_LBXTC=7 measured_LBDHDD=8 measured_BMXBMI=7"
        )
    ]


def test_the_angina_item_does_not_make_label_1() -> None:
    angina = {"MCQ_L": {"MCQ160D": 1.0}}
    records = _records([Respondent(1, extra=angina), Respondent(2, answers=(1.0, 2.0, 2.0, 2.0))])
    assert records["target"].to_dict() == {1: 0, 2: 1}


@pytest.mark.parametrize("unknown", [7.0, 9.0, None], ids=["refused", "dont-know", "missing"])
def test_an_unknown_angina_answer_with_four_noes_is_not_left_out(unknown) -> None:
    unsure = {"MCQ_L": {"MCQ160D": unknown}}
    records = _records([Respondent(1, extra=unsure), Respondent(2, answers=(1.0,) * 4)])
    assert records["target"].to_dict() == {1: 0, 2: 1}


def test_the_four_questions_never_enter_the_frame() -> None:
    records = _records(mixed())
    frame = nhanes_frame(records)
    excluded = {
        *("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"),
        *("WTINT2YR", "WTMEC2YR", "WTPH2YR", "SEQN"),
    }
    assert excluded.isdisjoint(records.columns)
    assert excluded.isdisjoint(frame.columns)


def test_the_excluded_columns_never_enter_the_frame() -> None:
    records = _records(mixed())
    every = {name for columns in COLUMNS.values() for name in columns}
    assert set(records.columns) & every == {"LBXGH", "LBXTC", "LBDHDD", "BMXBMI"}


def test_an_interview_only_adult_has_every_lab_not_measured() -> None:
    records = _records([Respondent(1, examined=False), Respondent(2)])
    assert _values(records, 1, EXAM) == [0, 0, 0, 0, 0, 0]
    assert _values(records, 2, EXAM) == [2, 2, 1, 2, 2, 2]


def test_a_component_with_the_wrong_columns_is_refused() -> None:
    data = components([Respondent(1)])["HDL_L"]
    with pytest.raises(
        ContractError, match=r"^HDL_L: the columns are not the published file's 4 names in order$"
    ):
        read_component(xpt_bytes(data.drop(columns="LBDHDDSI")), "HDL_L")
    swapped = data[["SEQN", "LBDHDD", "WTPH2YR", "LBDHDDSI"]]
    with pytest.raises(ContractError, match="names in order"):
        read_component(xpt_bytes(swapped), "HDL_L")


def test_each_cited_field_is_coded_at_its_cutoffs() -> None:
    people = [
        Respondent(1, systolic=(129.0,) * 3, diastolic=(79.0,) * 3, hba1c=5.6, cholesterol=239.0),
        Respondent(2, systolic=(130.0,) * 3, diastolic=(80.0,) * 3, hba1c=5.7, cholesterol=240.0),
        Respondent(3, hba1c=6.4, hdl=39.0),
        Respondent(4, hba1c=6.5, hdl=40.0),
    ]
    records = _records(people)
    assert records[["BPXOSY", "BPXODI", "LBXTC"]].loc[[1, 2]].values.tolist() == [
        [1, 1, 1],
        [2, 2, 2],
    ]
    assert records["LBXGH"].tolist() == [1, 2, 2, 3]
    assert records.loc[[3, 4], "LBDHDD"].tolist() == [1, 2]


def test_a_blood_pressure_is_the_mean_of_the_readings_present() -> None:
    people = [
        Respondent(1, systolic=(128.0, 130.0, 132.0)),
        Respondent(2, systolic=(129.0, 130.0, 130.0)),
        Respondent(3, systolic=(None, 131.0, None)),
        Respondent(4, systolic=(None, None, None)),
    ]
    assert _records(people)["BPXOSY"].tolist() == [2, 1, 2, 0]


def test_a_value_is_read_at_two_decimals_before_it_is_coded() -> None:
    people = [Respondent(1, hba1c=5.699999), Respondent(2, hba1c=5.694)]
    assert _records(people)["LBXGH"].tolist() == [2, 1]


def test_a_glycohemoglobin_of_0_is_no_lab_result() -> None:
    assert _records([Respondent(1, hba1c=0.0), Respondent(2, cholesterol=0.0)])[
        ["LBXGH", "LBXTC"]
    ].values.tolist() == [[0, 2], [1, 1]]


def test_each_bmi_tertile_takes_the_upper_state_at_its_cutoff() -> None:
    people = [Respondent(n, bmi=b) for n, b in enumerate((23.39, 23.4, 29.59, 29.6), start=1)]
    assert _records(people)["BMXBMI"].tolist() == [1, 2, 2, 3]


@pytest.mark.parametrize(
    ("component", "column", "value", "match"),
    [
        ("DEMO_L", "RIAGENDR", 3.0, "^respondent 1: RIAGENDR is 3.0, not 1 or 2$"),
        ("DEMO_L", "RIDAGEYR", 40.5, "^respondent 1: RIDAGEYR is 40.5, not a whole age 0 to 80$"),
        ("DEMO_L", "RIDAGEYR", 81.0, "^respondent 1: RIDAGEYR is 81.0, not a whole age 0 to 80$"),
        ("DEMO_L", "RIDAGEYR", -1.0, "^respondent 1: RIDAGEYR is -1.0, not a whole age 0 to 80$"),
        ("DEMO_L", "RIAGENDR", None, "^respondent 1: RIAGENDR is nan, not 1 or 2$"),
        ("MCQ_L", "MCQ160E", 3.0, r"^respondent 1: MCQ160E is 3.0, not 1, 2, 7, 9 or \.$"),
        ("TCHOL_L", "LBXTC", -1.0, "^respondent 1: LBXTC is -1.0, negative$"),
        ("BPXO_L", "BPXOSY2", -1.0, "^respondent 1: BPXOSY2 is -1.0, negative$"),
        ("BPXO_L", "BPXODI3", -1.0, "^respondent 1: BPXODI3 is -1.0, negative$"),
        ("BMX_L", "BMXBMI", -1.0, "^respondent 1: BMXBMI is -1.0, negative$"),
        ("HDL_L", "LBDHDD", -1.0, "^respondent 1: LBDHDD is -1.0, negative$"),
        ("GHB_L", "LBXGH", -1.0, "^respondent 1: LBXGH is -1.0, negative$"),
    ],
    ids=[
        *("sex", "age", "age-81", "age-negative", "sex-missing", "answer", "cholesterol"),
        *("systolic", "diastolic", "bmi", "hdl", "hba1c"),
    ],
)
def test_a_value_outside_its_codebook_names_its_respondent(
    component: str, column: str, value: float, match: str
) -> None:
    person = Respondent(1, extra={component: {column: value}})
    with pytest.raises(ContractError, match=match):
        _records([person, Respondent(2)])


def test_the_first_respondent_outside_the_codebook_is_named() -> None:
    bad = {"DEMO_L": {"RIAGENDR": 3.0}}
    people = [Respondent(1), Respondent(2, extra=bad), Respondent(3, extra=bad)]
    with pytest.raises(ContractError, match=r"^respondent 2: RIAGENDR is 3\.0, not 1 or 2$"):
        _records(people)


def test_a_missing_value_shows_no_concept() -> None:
    people = [
        Respondent(1, examined=False),
        Respondent(2, systolic=(140.0,) * 3, diastolic=(90.0,) * 3, hba1c=6.0, hdl=35.0),
    ]
    first, second = (
        {str(name): float(value) for name, value in record.items()}
        for record in _records(people).to_dict("records")
    )
    labels = {row.concept: row.label for row in NHANES_FIELD_ROWS}
    assert shown(first, NHANES_FIELD_ROWS) == set()
    assert {labels[c] for c in shown(second, NHANES_FIELD_ROWS)} == {
        "Elevated systolic blood pressure",
        "Elevated diastolic blood pressure",
        "Elevated hemoglobin A1c",
        "Hypercholesterolemia",
        "Decreased circulating HDL-C concentration",
    }


def test_the_frame_holds_exactly_the_named_fields() -> None:
    records = _records(mixed())
    frame = nhanes_frame(records)
    assert frame.columns == (
        *("age", "sex", "BPXOSY", "BPXODI", "LBXGH", "LBXTC", "LBDHDD", "BMXBMI", "target"),
    )
    assert frame.numeric == ("age",)
    assert frame.edges == (40, 60)
    assert tuple(records.columns) == frame.columns
    assert {str(dtype) for dtype in records.dtypes} == {"int64"}
    check_records(records, frame)
    assert (frame.ranges[0].low, frame.ranges[0].high) == (20, 80)
    assert dict(frame.codebook["LBXGH"]) == {
        0: "not measured",
        1: "below 5.7",
        2: "at least 5.7 and below 6.5",
        3: "at least 6.5",
    }
    assert dict(frame.codebook["sex"]) == {0: "female", 1: "male"}
    assert records["sex"].to_dict()[130380] == 0


def test_the_label_codes_are_0_and_1() -> None:
    records = _records(mixed())
    assert records["target"].to_dict() == {
        130378: 0,
        130380: 1,
        130381: 0,
        130383: 1,
        130384: 1,
        130385: 0,
        130386: 1,
        130388: 0,
        130389: 0,
    }
    assert nhanes_frame(records).label_codes == (0, 1)


def test_frame_bytes_do_not_depend_on_file_order() -> None:
    people = mixed()
    shuffled = [*people[7::-1], *people[:7:-1]]
    assert shuffled != people
    assert digest(_records(shuffled)) == MIXED_DIGEST
    assert digest(_records(people)) == MIXED_DIGEST


def test_load_reads_each_component_of_a_verified_snapshot(tmp_path: Path) -> None:
    sources = _planted(tmp_path, files(mixed()))
    loaded = load_nhanes(tmp_path, sources=sources)
    pd.testing.assert_frame_equal(loaded, _records(mixed()))
    with pytest.raises(ValueError, match="7 component files, not 1"):
        load_nhanes(tmp_path, sources=sources[:1])


def _observations(data: bytes) -> int:
    """The offset of `data`'s first observation: the byte after its OBS header record."""
    return data.index(b"HEADER RECORD*******OBS     HEADER RECORD") + 80


WIDTH = 8 * len(COLUMNS["DEMO_L"])
"""One DEMO_L observation's bytes: every column numeric, 8 bytes each."""


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda data: b"not an xport file" * 10, "^DEMO_L: not a readable XPORT file: "),
        (operator.itemgetter(slice(200)), "^DEMO_L: not a readable XPORT file: ValueError: "),
        (
            lambda data: data[: _observations(data)],
            "^DEMO_L: not a readable XPORT file: StopIteration: ",
        ),
        (
            lambda data: data[: _observations(data) + 1],
            "^DEMO_L: not a readable XPORT file: UserWarning: xport file may be corrupted",
        ),
        (
            lambda data: data[: _observations(data) + WIDTH],
            "^DEMO_L: not a readable XPORT file: UserWarning: xport file may be corrupted",
        ),
    ],
    ids=["not-xport", "cut-in-headers", "cut-at-observations", "cut-in-a-row", "cut-after-a-row"],
)
def test_a_damaged_component_is_a_contract_error(tmp_path: Path, make, match: str) -> None:
    data = files(mixed())
    data["DEMO_L"] = make(data["DEMO_L"])
    sources = _planted(tmp_path, data)
    with pytest.raises(ContractError, match=match):
        load_nhanes(tmp_path, sources=sources)


@pytest.mark.parametrize(
    ("people", "match"),
    [
        ([Respondent(1, age=19.0)], "the cohort holds no records"),
        ([Respondent(1), Respondent(2)], "no adult with a component of the composite"),
    ],
    ids=["empty", "no-positive"],
)
def test_load_refuses_a_cohort_it_cannot_score(tmp_path: Path, people, match: str) -> None:
    sources = _planted(tmp_path, files(people))
    with pytest.raises(ContractError, match=match):
        load_nhanes(tmp_path, sources=sources)


def test_the_fixtures_real_rows_are_one_respondent() -> None:
    assert {row[0] for row in REAL_ROWS.values()} == {130378.0}
    assert all(len(REAL_ROWS[name]) == len(REAL_COLUMNS[name]) for name in COMPONENTS)
