"""The serializer writes a record as one fixed prompt for training and scoring."""

from __future__ import annotations

import dataclasses
from typing import Final

import pandas as pd
import pytest

from bko.datasets.errors import ContractError
from bko.ontology.frames import DIABETES_FRAME
from bko.slm.serialize import ANSWERS, examples, prompt, prompts
from tests._frame_fixtures import TOY
from tests._slm_fixtures import records


_DOSE: Final = (
    "(No = not prescribed; Steady, Up or Down = prescribed with the dose unchanged, raised or "
    "lowered)"
)

RECORD: Final = {
    "age": 6,
    "sex": 1,
    "race": 2,
    "time_in_hospital": 3,
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 12,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 9,
    "A1Cresult": 3,
    "max_glu_serum": 0,
    "insulin": 2,
    "metformin": 0,
    "glipizide": 0,
    "glyburide": 1,
    "pioglitazone": 0,
    "rosiglitazone": 0,
    "glimepiride": 0,
    "repaglinide": 0,
    "change": 1,
    "diabetesMed": 1,
    "diabetic_kidney_disease": 0,
    "essential_hypertension": 1,
    "acute_myocardial_infarction": 0,
    "heart_failure": 1,
    "cerebral_artery_occlusion": 0,
    "pneumonia": 0,
    "chronic_bronchitis": 0,
    "chronic_obstructive_pulmonary_disease": 0,
    "chronic_kidney_disease": 1,
    "chronic_ulcer_of_skin": 0,
}

GOLDEN: Final = (
    "age bracket as its decade index, 2 for 20 to 29 years up to 9 for 90 to 99 years: 6\n"
    "sex: male\n"
    "race as the encounter records it (? = not recorded): AfricanAmerican\n"
    "days from admission to discharge: 3\n"
    "lab tests performed during the encounter: 41\n"
    "procedures other than lab tests performed during the encounter: 0\n"
    "distinct generic medications given during the encounter: 12\n"
    "outpatient visits in the year before the encounter: 0\n"
    "emergency visits in the year before the encounter: 1\n"
    "inpatient visits in the year before the encounter: 2\n"
    "diagnoses entered for the encounter: 9\n"
    "HbA1c result in percent during the encounter (None = not measured): >8\n"
    "serum glucose result in mg/dL during the encounter (None = not measured): None\n"
    f"insulin during the encounter {_DOSE}: Up\n"
    f"metformin during the encounter {_DOSE}: No\n"
    f"glipizide during the encounter {_DOSE}: No\n"
    f"glyburide during the encounter {_DOSE}: Steady\n"
    f"pioglitazone during the encounter {_DOSE}: No\n"
    f"rosiglitazone during the encounter {_DOSE}: No\n"
    f"glimepiride during the encounter {_DOSE}: No\n"
    f"repaglinide during the encounter {_DOSE}: No\n"
    "a diabetes medication's dose changed during the encounter (Ch = changed): Ch\n"
    "any diabetes medication prescribed during the encounter: Yes\n"
    "diabetic kidney disease (ICD-9 250.4) among the three diagnoses: absent\n"
    "essential hypertension (ICD-9 401) among the three diagnoses: present\n"
    "acute myocardial infarction (ICD-9 410) among the three diagnoses: absent\n"
    "heart failure (ICD-9 428) among the three diagnoses: present\n"
    "cerebral artery occlusion (ICD-9 434) among the three diagnoses: absent\n"
    "pneumonia (ICD-9 486) among the three diagnoses: absent\n"
    "chronic bronchitis (ICD-9 491) among the three diagnoses: absent\n"
    "chronic obstructive pulmonary disease (ICD-9 496) among the three diagnoses: absent\n"
    "chronic kidney disease (ICD-9 585) among the three diagnoses: present\n"
    "chronic ulcer of skin (ICD-9 707) among the three diagnoses: absent\n"
    "Question: was this patient readmitted within 30 days of discharge? Answer:"
)


def test_a_record_serializes_to_the_golden_string() -> None:
    assert prompt(RECORD, DIABETES_FRAME) == GOLDEN
    frame = pd.DataFrame([RECORD]).astype("float64")
    assert prompts(frame, DIABETES_FRAME) == (GOLDEN,)


def test_fields_follow_the_frames_feature_order() -> None:
    shuffled = dict(reversed(list(RECORD.items())))
    lines = prompt(shuffled, DIABETES_FRAME).split("\n")
    described = [line.rsplit(": ", 1)[0] for line in lines[:-1]]
    assert described == [DIABETES_FRAME.descriptions[name] for name in DIABETES_FRAME.features]


@pytest.mark.parametrize("label", [0, 1])
def test_the_training_text_minus_its_answer_is_the_scoring_prompt(label: int) -> None:
    frame = pd.DataFrame([{**RECORD, DIABETES_FRAME.label: label}])
    (example,) = examples(frame, DIABETES_FRAME)
    assert example.answer == ANSWERS[label]
    assert example.text == GOLDEN + ANSWERS[label]
    assert example.text.removesuffix(example.answer) == prompt(RECORD, DIABETES_FRAME)


def test_the_prompt_holds_no_label() -> None:
    fixture = records(6)
    with_label = prompts(fixture, DIABETES_FRAME)
    without = prompts(fixture.drop(columns=DIABETES_FRAME.label), DIABETES_FRAME)
    assert with_label == without
    for text in with_label:
        assert not text.endswith(tuple(ANSWERS.values()))
        assert DIABETES_FRAME.descriptions[DIABETES_FRAME.label] + ":" not in text


def test_an_integer_column_never_prints_a_decimal_point() -> None:
    fixture = records(8).astype("float64")
    for text in prompts(fixture, DIABETES_FRAME):
        for line, name in zip(text.split("\n"), DIABETES_FRAME.features, strict=False):
            if name in DIABETES_FRAME.numeric:
                assert "." not in line.rsplit(": ", 1)[1], line
    fractional = dataclasses.replace(
        DIABETES_FRAME, fractional=("time_in_hospital",), fraction_digits=2
    )
    assert "days from admission to discharge: 3.00\n" in prompt(RECORD, fractional)
    with pytest.raises(ContractError, match="whole"):
        prompt({**RECORD, "time_in_hospital": 3.5}, DIABETES_FRAME)


def test_a_value_outside_the_record_is_refused() -> None:
    with pytest.raises(ContractError, match="codebook lacks"):
        prompt({**RECORD, "race": 9}, DIABETES_FRAME)
    with pytest.raises(ContractError, match="no value"):
        prompt({**RECORD, "age": float("nan")}, DIABETES_FRAME)
    frame = pd.DataFrame([{**RECORD, DIABETES_FRAME.label: 2}])
    with pytest.raises(ContractError, match="no answer"):
        examples(frame, DIABETES_FRAME)


def test_a_frame_without_a_question_is_refused() -> None:
    with pytest.raises(ContractError, match="no question"):
        prompt(RECORD, TOY)
    with pytest.raises(ContractError, match="no question"):
        prompts(pd.DataFrame([RECORD]), TOY)
