"""The Diabetes 130 frame, its indicators and its field rows, read by `bko.evaluate.data.load`.

The frame's coded columns carry the archive's own state strings as their meanings, inverted from
the codes `bko.datasets.diabetes` gives them, so a prompt names each state as the archive writes
it; `check_records` refuses records whose columns or codes the frame does not hold.
`IDS_mapping.csv` names admission and discharge ids, none of which the frame keeps, so no codebook
reads it. The
indicators and the field rows come from `bko.ontology.icd9`: one indicator and one row per
`ICD9_MAPPINGS` row, and one row per `LAB_MAPPINGS` row over its field's codes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.diabetes import ADULTS_FROM, COUNTS, DRUGS, STATE_CODES
from bko.datasets.errors import ContractError
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.ontology.icd9 import ICD9_MAPPINGS, LAB_MAPPINGS, Icd9Mapping, LabMapping
from bko.ontology.schema import Codes, FieldMapping


DIABETES_NAME: Final = "diabetes"
"""The Diabetes frame's name and its `Dataset` value."""


def indicator(row: Icd9Mapping) -> str:
    """The indicator column of `row`: its label, lowercased, each run of other characters `_`."""
    return re.sub(r"[^a-z0-9]+", "_", row.label.lower()).strip("_")


DIABETES_INDICATORS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {indicator(row): row.codes for row in ICD9_MAPPINGS}
)
"""Each indicator column's ICD-9 categories, in `ICD9_MAPPINGS` order, as `load_diabetes` takes
them."""

_PRESENCE: Final[Mapping[int, str]] = MappingProxyType({0: "absent", 1: "present"})

_STATES: Final[dict[str, Mapping[int, str]]] = {
    name: {code: state for state, code in codes.items()} for name, codes in STATE_CODES.items()
}

_DOSE: Final = (
    "No = not prescribed; Steady, Up or Down = prescribed with the dose unchanged, raised or "
    "lowered"
)

_DESCRIPTIONS: Final[dict[str, str]] = {
    "age": "age bracket as its decade index, 2 for 20 to 29 years up to 9 for 90 to 99 years",
    "sex": "sex",
    "race": "race as the encounter records it (? = not recorded)",
    "time_in_hospital": "days from admission to discharge",
    "num_lab_procedures": "lab tests performed during the encounter",
    "num_procedures": "procedures other than lab tests performed during the encounter",
    "num_medications": "distinct generic medications given during the encounter",
    "number_outpatient": "outpatient visits in the year before the encounter",
    "number_emergency": "emergency visits in the year before the encounter",
    "number_inpatient": "inpatient visits in the year before the encounter",
    "number_diagnoses": "diagnoses entered for the encounter",
    "A1Cresult": "HbA1c result in percent during the encounter (None = not measured)",
    "max_glu_serum": "serum glucose result in mg/dL during the encounter (None = not measured)",
    **{name: f"{name} during the encounter ({_DOSE})" for name in DRUGS},
    "change": "a diabetes medication's dose changed during the encounter (Ch = changed)",
    "diabetesMed": "any diabetes medication prescribed during the encounter",
    **{
        indicator(row): f"{row.label} (ICD-9 {', '.join(row.codes)}) among the three diagnoses"
        for row in ICD9_MAPPINGS
    },
    LABEL: "readmitted within 30 days of discharge",
}

_COUNT_UNITS: Final[dict[str, str]] = {
    "num_lab_procedures": "{tests}",
    "num_procedures": "{procedures}",
    "num_medications": "{medications}",
    "number_outpatient": "{visits}",
    "number_emergency": "{visits}",
    "number_inpatient": "{visits}",
    "number_diagnoses": "{diagnoses}",
}

_COUNT_BOUNDS: Final[dict[str, tuple[int, int]]] = {
    "num_lab_procedures": (1, 132),
    "num_procedures": (0, 6),
    "num_medications": (1, 81),
    "number_outpatient": (0, 42),
    "number_emergency": (0, 76),
    "number_inpatient": (0, 21),
    "number_diagnoses": (1, 16),
}

_ARCHIVE_LIMITS: Final = (
    "the least and greatest value over the archive's 101,766 encounters; a value outside them is "
    "unseen, not impossible"
)

DIABETES_FRAME: Final = FrameSpec(
    name=DIABETES_NAME,
    features=(
        "age",
        "sex",
        "race",
        *COUNTS,
        *(name for name in STATE_CODES if name != "race"),
        *DIABETES_INDICATORS,
    ),
    numeric=("age", *COUNTS),
    fractional=(),
    fraction_digits=0,
    categorical=("sex", *STATE_CODES, *DIABETES_INDICATORS),
    codebook={
        "sex": SEXES,
        **_STATES,
        **dict.fromkeys(DIABETES_INDICATORS, _PRESENCE),
    },
    descriptions=_DESCRIPTIONS,
    ranges=(
        RangeConstraint(
            field="age",
            unit="10.a",
            low=ADULTS_FROM,
            high=9,
            source="clore-2014",
            limits="the archive's brackets [20-30) to [90-100); the frame drops those under 20",
        ),
        RangeConstraint(
            field="time_in_hospital",
            unit="d",
            low=1,
            high=14,
            source="strack-2014",
            limits="the archive's inclusion criterion: a stay of 1 to 14 days",
        ),
        *(
            RangeConstraint(
                field=name,
                unit=_COUNT_UNITS[name],
                low=low,
                high=high,
                source="clore-2014",
                limits=_ARCHIVE_LIMITS,
            )
            for name, (low, high) in _COUNT_BOUNDS.items()
        ),
    ),
    label_codes=tuple(sorted(LABEL_CODES)),
    label_meanings={0: "not readmitted within 30 days", 1: "readmitted within 30 days"},
    edges=(4, 6, 8),
    study="hospital readmission within 30 days of patients with diabetes",
    cohort="adult patient's first inpatient encounter with diabetes at a US hospital",
)
"""The frame over Diabetes 130's first encounters (`bko.datasets.diabetes.frame_records`)."""


def _icd9_row(row: Icd9Mapping) -> FieldMapping:
    return FieldMapping(
        when=(Codes(indicator(row), frozenset({1})),),
        concept=row.concept,
        label=row.label,
        source=row.source,
        limits=row.limits,
    )


def _lab_row(row: LabMapping) -> FieldMapping:
    codes = STATE_CODES[row.field]
    return FieldMapping(
        when=(Codes(row.field, frozenset(codes[state] for state in row.states)),),
        concept=row.concept,
        label=row.label,
        source=row.source,
        limits=row.limits,
    )


DIABETES_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = (
    *(_icd9_row(row) for row in ICD9_MAPPINGS),
    *(_lab_row(row) for row in LAB_MAPPINGS),
)
"""The Diabetes frame's field rows: each ICD-9 row's `Codes(indicator, {1})`, then each lab row's
`Codes(field, codes of its states)`."""


def check_records(records: pd.DataFrame) -> None:
    """Refuse records whose columns are not the frame's or whose codes its codebook lacks.

    Raises:
        ContractError: If the columns differ from `DIABETES_FRAME.columns`, in order, or a coded
            column or the label holds a code its codebook does not.
    """
    frame = DIABETES_FRAME
    if tuple(records.columns) != frame.columns:
        msg = f"the records' columns are not frame {frame.name}'s columns in order"
        raise ContractError(msg)
    coded = {**frame.codebook, frame.label: frame.label_meanings}
    for name, codes in coded.items():
        stray = sorted(set(records[name].unique()) - set(codes))
        if stray:
            msg = f"{name} holds codes {stray} that frame {frame.name}'s codebook lacks"
            raise ContractError(msg)
