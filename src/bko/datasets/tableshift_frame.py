"""The `tableshift` frame: TableShift's readmission features over Diabetes 130's kept rows.

The frame holds every column of `diabetic_data.csv` but the two identifiers, the domain column
`admission_source_id` and the label source `readmitted`: 46 features, in file order, with
`gender` read as `sex`. `sex` is 1 for `Male` against every other value and `race` 1 for
`Caucasian` against every other value, TableShift's grouper. The lab, drug, `change` and
`diabetesMed` columns take `bko.datasets.diabetes`' codes, the diagnoses their
`bko.datasets.diagnoses.code_number`, and `weight`, `payer_code` and `medical_specialty` the
position of each value among the kept rows' sorted values, so the frame is built from the rows.
`LABEL` is 1 for any readmission. ADR `research/tableshift-comparison` records the choices.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.diabetes import (
    A1C_CODES,
    AGE_BRACKETS,
    CHANGE_CODES,
    COLUMNS,
    COUNTS,
    DIABETES_MED_CODES,
    DIAGNOSES,
    DRUG_CODES,
    GLU_CODES,
)
from bko.datasets.diagnoses import UNKNOWN, code_number, code_text
from bko.datasets.frames import FrameSpec, RangeConstraint


NAME: Final = "tableshift"
"""The frame's name."""

DOMAIN: Final = "admission_source_id"
"""The column TableShift splits the domains on and drops after the split."""

FEATURES: Final[tuple[str, ...]] = tuple(
    "sex" if name == "gender" else name
    for name in COLUMNS
    if name not in {"encounter_id", "patient_nbr", DOMAIN, "readmitted"}
)
"""The 46 features, in `diabetic_data.csv`'s order, `gender` as `sex`."""

TEXT_FIELDS: Final = ("weight", "payer_code", "medical_specialty")
"""The free-text columns, coded by each value's position among the kept rows' sorted values."""

ID_FIELDS: Final = ("admission_type_id", "discharge_disposition_id")
"""The id columns, which keep their ids as codes."""

DRUG_COLUMNS: Final = COLUMNS[COLUMNS.index("metformin") : COLUMNS.index("change")]
"""The archive's 23 drug columns, in file order."""

STATES: Final[Mapping[str, Mapping[str, int]]] = MappingProxyType(
    {
        "max_glu_serum": GLU_CODES,
        "A1Cresult": A1C_CODES,
        **dict.fromkeys(DRUG_COLUMNS, DRUG_CODES),
        "change": CHANGE_CODES,
        "diabetesMed": DIABETES_MED_CODES,
    }
)
"""Each state column and its codes, those of the `diabetes` frame."""

EDGES: Final = (1, 2, 4, 6, 8)
"""The age band starts, in decade indices: under 10, 10 to 19, 20 to 39, 40 to 59, 60 to 79 and 80
on."""

_DOSE: Final = (
    "No = not prescribed; Steady, Up or Down = prescribed with the dose unchanged, raised or "
    "lowered"
)

_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "race": "race as the encounter records it, Caucasian against every other recorded race",
    "sex": "sex, male against every other recorded gender (Unknown/Invalid counts as female)",
    "age": "age bracket as its decade index, 0 for under 10 years up to 9 for 90 to 99 years",
    "weight": "weight bracket in pounds as the encounter records it (? = not recorded)",
    "admission_type_id": "admission type, as its IDS_mapping.csv id",
    "discharge_disposition_id": "discharge disposition, as its IDS_mapping.csv id",
    "time_in_hospital": "days from admission to discharge",
    "payer_code": "payer code as the encounter records it (? = not recorded)",
    "medical_specialty": "admitting physician's specialty (? = not recorded)",
    "num_lab_procedures": "lab tests performed during the encounter",
    "num_procedures": "procedures other than lab tests performed during the encounter",
    "num_medications": "distinct generic medications given during the encounter",
    "number_outpatient": "outpatient visits in the year before the encounter",
    "number_emergency": "emergency visits in the year before the encounter",
    "number_inpatient": "inpatient visits in the year before the encounter",
    "diag_1": "ICD-9 code of the encounter's primary diagnosis (? = none recorded)",
    "diag_2": "ICD-9 code of the encounter's secondary diagnosis (? = none recorded)",
    "diag_3": "ICD-9 code of the encounter's additional secondary diagnosis (? = none recorded)",
    "number_diagnoses": "diagnoses entered for the encounter",
    "max_glu_serum": "serum glucose result in mg/dL during the encounter (None = not measured)",
    "A1Cresult": "HbA1c result in percent during the encounter (None = not measured)",
    **{name: f"{name} during the encounter ({_DOSE})" for name in DRUG_COLUMNS},
    "change": "a diabetes medication's dose changed during the encounter (Ch = changed)",
    "diabetesMed": "any diabetes medication prescribed during the encounter",
    LABEL: "readmitted after discharge, within or after 30 days",
}

_RANGE_LIMITS: Final = (
    "the least and greatest value over the frame's kept rows; a value outside them is unseen, not "
    "impossible"
)


def encode(kept: pd.DataFrame) -> pd.DataFrame:
    """The records of `kept`, the output of `bko.datasets.tableshift.kept_rows`.

    Returns:
        `FEATURES`, then `LABEL`, all int64, on `kept`'s index.
    """
    texts = _text_codes(kept)
    columns: dict[str, pd.Series] = {}
    for name in FEATURES:
        if name == "sex":
            columns[name] = (kept["gender"] == "Male").astype("int64")
        elif name == "race":
            columns[name] = (kept["race"] == "Caucasian").astype("int64")
        elif name == "age":
            columns[name] = kept["age"].map(AGE_BRACKETS)
        elif name in TEXT_FIELDS:
            columns[name] = kept[name].map(texts[name])
        elif name in DIAGNOSES:
            tokens = kept[name].unique()
            columns[name] = kept[name].map({token: code_number(token) for token in tokens})
        elif name in STATES:
            columns[name] = kept[name].map(STATES[name])
        else:
            columns[name] = kept[name].astype("int64")
    columns[LABEL] = (kept["readmitted"] != "NO").astype("int64")
    return pd.DataFrame(columns, index=kept.index).astype("int64")


def tableshift_frame(kept: pd.DataFrame) -> FrameSpec:
    """The frame of `kept`, the output of `bko.datasets.tableshift.kept_rows`.

    The codebooks of `TEXT_FIELDS`, `ID_FIELDS` and `DIAGNOSES` and the counts' ranges are read
    from `kept`.
    """
    texts = _text_codes(kept)
    tokens = sorted({token for name in DIAGNOSES for token in kept[name].unique()})
    numbers = sorted(code_number(token) for token in tokens)
    diagnoses = {n: "? (none recorded)" if n == UNKNOWN else code_text(n) for n in numbers}
    codebook: dict[str, Mapping[int, str]] = {
        "race": {0: "not Caucasian", 1: "Caucasian"},
        "sex": SEXES,
        **{name: {code: value for value, code in texts[name].items()} for name in TEXT_FIELDS},
        **{
            name: {int(i): f"{name} {int(i)}" for i in sorted(kept[name].astype("int64").unique())}
            for name in ID_FIELDS
        },
        **dict.fromkeys(DIAGNOSES, diagnoses),
        **{name: {code: state for state, code in codes.items()} for name, codes in STATES.items()},
    }
    numeric = ("age", *COUNTS)
    return FrameSpec(
        name=NAME,
        features=FEATURES,
        numeric=numeric,
        fractional=(),
        fraction_digits=0,
        categorical=tuple(name for name in FEATURES if name not in numeric),
        codebook=codebook,
        descriptions=_DESCRIPTIONS,
        ranges=(
            RangeConstraint(
                field="age",
                unit="10.a",
                low=0,
                high=9,
                source="clore-2014",
                limits="the archive's brackets [0-10) to [90-100), every one kept",
            ),
            *(
                RangeConstraint(
                    field=name,
                    unit="d" if name == "time_in_hospital" else "1",
                    low=int(kept[name].astype("int64").min()),
                    high=int(kept[name].astype("int64").max()),
                    source="clore-2014",
                    limits=_RANGE_LIMITS,
                )
                for name in COUNTS
            ),
        ),
        label_codes=tuple(sorted(LABEL_CODES)),
        label_meanings={0: "not readmitted", 1: "readmitted, within or after 30 days"},
        edges=EDGES,
        study="hospital readmission of patients with diabetes",
        cohort="inpatient encounter with diabetes at a US hospital",
    )


def _text_codes(kept: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Each of `TEXT_FIELDS`' values and its code: its position among `kept`'s sorted values."""
    return {
        name: {value: code for code, value in enumerate(sorted(kept[name].unique()))}
        for name in TEXT_FIELDS
    }
