"""The `diabetes-codes` frame: Diabetes 130's cohort with each diagnosis code kept.

The frame is `bko.ontology.frames.DIABETES_FRAME` with the category indicators replaced by the
three diagnosis columns, each holding `bko.datasets.diagnoses.code_number` of the archive's token.
Their codebook is the codes the loaded records hold, so the frame is built from the records. The
field rows are one per (code of `bko.ontology.icd9_codes.ICD9_CODES`, diagnosis column), then the
lab rows (ADR `research/diabetes-diagnosis-codes`).
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.diabetes import DIAGNOSES
from bko.datasets.diagnoses import UNKNOWN, code_number, code_text
from bko.datasets.frames import FrameSpec
from bko.ontology.frames import DIABETES_FRAME, DIABETES_INDICATORS, LAB_FIELD_ROWS
from bko.ontology.icd9 import Icd9Code
from bko.ontology.icd9_codes import ICD9_CODES
from bko.ontology.schema import Codes, FieldMapping


DIABETES_CODES_NAME: Final = "diabetes-codes"
"""The frame's name and its `Dataset` value."""

_POSITIONS: Final = {"diag_1": "primary", "diag_2": "secondary", "diag_3": "additional secondary"}


def _code_row(row: Icd9Code, column: str) -> FieldMapping:
    return FieldMapping(
        when=(Codes(column, frozenset({code_number(row.code)})),),
        concept=row.concept,
        label=row.label,
        source="mondo",
        limits=(
            f"ICD-9 {row.code} as the encounter's {_POSITIONS[column]} diagnosis; the release "
            f"carries the code's xref on {row.concept}"
        ),
    )


DIABETES_CODES_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = (
    *(_code_row(row, column) for column in DIAGNOSES for row in ICD9_CODES),
    *LAB_FIELD_ROWS,
)
"""One `Codes(column, {code number})` row per diagnosis column and mapped code, then
`LAB_FIELD_ROWS`; each column's rows partition its mapped codes."""


def diabetes_codes_frame(records: pd.DataFrame) -> FrameSpec:
    """The frame of `records`, the output of `load_diabetes(..., diagnoses=True)`.

    Args:
        records: The frame records; their diagnosis columns' codes make the codebook.

    Returns:
        `DIABETES_FRAME` without its indicators, with `DIAGNOSES` as coded columns sharing one
        codebook: each code any of them holds, meaning its token.
    """
    present = sorted({int(code) for name in DIAGNOSES for code in records[name].unique()})
    book = MappingProxyType(
        {code: "? (none recorded)" if code == UNKNOWN else code_text(code) for code in present}
    )
    keep = [name for name in DIABETES_FRAME.features if name not in DIABETES_INDICATORS]
    descriptions = {
        name: text
        for name, text in DIABETES_FRAME.descriptions.items()
        if name not in DIABETES_INDICATORS
    }
    label = descriptions.pop(DIABETES_FRAME.label)
    return dataclasses.replace(
        DIABETES_FRAME,
        name=DIABETES_CODES_NAME,
        features=(*keep, *DIAGNOSES),
        categorical=(
            *(name for name in DIABETES_FRAME.categorical if name not in DIABETES_INDICATORS),
            *DIAGNOSES,
        ),
        codebook={
            **{
                name: codes
                for name, codes in DIABETES_FRAME.codebook.items()
                if name not in DIABETES_INDICATORS
            },
            **dict.fromkeys(DIAGNOSES, book),
        },
        descriptions={
            **descriptions,
            **{
                name: f"ICD-9 code of the encounter's {position} diagnosis (? = none recorded)"
                for name, position in _POSITIONS.items()
            },
            DIABETES_FRAME.label: label,
        },
    )
