"""Offline inputs for the `bko.datasets.diabetes` tests: `diabetic_data.csv` text and archives.

The real data here is `REAL_HEADER`, the archive's header line, `REAL_LAYOUT`, one real data row
of `diabetic_data.csv` (its fourth), kept so the parser is proven against the real formatting, and
`IDS_EXCERPT`, rows of the real `IDS_mapping.csv`. Every other row is built from `REAL_LAYOUT` with
invented values.
"""

from __future__ import annotations

import io
import zipfile

from bko.datasets.diabetes import IDS_MEMBER, MEMBER
from tests._datasets_fixtures import pin


__all__ = ["IDS_EXCERPT", "REAL_HEADER", "REAL_LAYOUT", "archive", "diabetic_line", "pin", "text"]

REAL_HEADER = (
    "encounter_id,patient_nbr,race,gender,age,weight,admission_type_id,discharge_disposition_id,"
    "admission_source_id,time_in_hospital,payer_code,medical_specialty,num_lab_procedures,"
    "num_procedures,num_medications,number_outpatient,number_emergency,number_inpatient,diag_1,"
    "diag_2,diag_3,number_diagnoses,max_glu_serum,A1Cresult,metformin,repaglinide,nateglinide,"
    "chlorpropamide,glimepiride,acetohexamide,glipizide,glyburide,tolbutamide,pioglitazone,"
    "rosiglitazone,acarbose,miglitol,troglitazone,tolazamide,examide,citoglipton,insulin,"
    "glyburide-metformin,glipizide-metformin,glimepiride-pioglitazone,metformin-rosiglitazone,"
    "metformin-pioglitazone,change,diabetesMed,readmitted"
)
"""The first line of the real `diabetic_data.csv`, without its CRLF."""

REAL_LAYOUT = (
    "500364,82442376,Caucasian,Male,[30-40),?,1,1,7,2,?,?,44,1,16,0,0,0,8,250.43,403,7,None,None,"
    "No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,Up,No,No,No,No,No,Ch,Yes,NO"
)
"""The fourth line of the real `diabetic_data.csv`: an adult kept by the cohort, with a diagnosis
whose leading zeros are stripped (`8`) and one with two decimals (`250.43`)."""

IDS_EXCERPT = (
    "admission_type_id,description\r\n"
    "1,Emergency\r\n"
    "6,NULL\r\n"
    ",\r\n"
    "discharge_disposition_id,description\r\n"
    "1,Discharged to home\r\n"
    "7,Left AMA\r\n"
    "11,Expired\r\n"
    "12,Still patient or expected to return for outpatient services\r\n"
    "13,Hospice / home\r\n"
    "14,Hospice / medical facility\r\n"
    "18,NULL\r\n"
    '19,"Expired at home. Medicaid only, hospice."\r\n'
    '20,"Expired in a medical facility. Medicaid only, hospice."\r\n'
    '21,"Expired, place unknown. Medicaid only, hospice."\r\n'
    "25,Not Mapped\r\n"
    ",\r\n"
    "admission_source_id,description\r\n"
    "7, Emergency Room\r\n"
    "26,Transfer from Hospice\r\n"
)
"""Rows of the real `IDS_mapping.csv`, every Expired or Hospice discharge among them, and an
admission source naming Hospice that is not a discharge."""


def diabetic_line(**fields: str) -> str:
    """Return a `diabetic_data.csv` line in `REAL_LAYOUT`'s format with the given fields replaced.

    Hyphenated drug columns go through a dict: `diabetic_line(**{"glyburide-metformin": "Up"})`.
    """
    columns = REAL_HEADER.split(",")
    values: dict[str, str] = dict(zip(columns, REAL_LAYOUT.split(","), strict=True))
    unknown = set(fields) - set(columns)
    if unknown:
        msg = f"not diabetic_data.csv columns: {sorted(unknown)}"
        raise KeyError(msg)
    values |= fields
    return ",".join(values[name] for name in columns)


def text(*lines: str, header: str = REAL_HEADER) -> str:
    """Return `diabetic_data.csv` text: `header`, then `lines`, each ended by CRLF as UCI's is."""
    return "".join(f"{line}\r\n" for line in (header, *lines))


def archive(diabetic: str, ids: str = IDS_EXCERPT) -> bytes:
    """Return the two-member zip holding `diabetic` and `ids`, with a fixed date so bytes repeat."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for member, body in ((MEMBER, diabetic), (IDS_MEMBER, ids)):
            bundle.writestr(zipfile.ZipInfo(member, date_time=(2023, 5, 22, 0, 0, 0)), body)
    return buffer.getvalue()
