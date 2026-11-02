"""Offline inputs for the `bko.datasets.diabetes` tests: `diabetic_data.csv` text and archives.

The real data here is `REAL_HEADER`, the archive's header line, `REAL_LAYOUT`, one real data row
of `diabetic_data.csv` (its fourth), kept so the parser is proven against the real formatting, and
`IDS_EXCERPT`, rows of the real `IDS_mapping.csv`. Every other row is built from `REAL_LAYOUT` with
invented values, `toy_archive`'s cohort among them. `diabetes_releases` names each concept the
Diabetes field rows map under an invented root; its labels are the rows' labels, and no stanza
copies release content.
"""

from __future__ import annotations

import io
import zipfile
from typing import Final

from bko.datasets.diabetes import DRUGS, IDS_MEMBER, MEMBER
from bko.datasets.sources import Source
from bko.ontology.icd9 import ICD9_MAPPINGS, LAB_MAPPINGS
from tests._datasets_fixtures import pin


__all__ = [
    "IDS_EXCERPT",
    "REAL_HEADER",
    "REAL_LAYOUT",
    "TOY_CELLS",
    "archive",
    "diabetes_releases",
    "diabetic_line",
    "pin",
    "text",
    "toy_archive",
]

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


_BANDS: Final = ("<4", "4-5", "6-7", ">=8")

TOY_CELLS: Final = {
    **{(band, sex, 0): 40 for band in _BANDS for sex in ("Female", "Male")},
    **{(band, sex, 1): 28 for band in _BANDS for sex in ("Female", "Male")},
    ("6-7", "Female", 1): 56,
}
"""The toy cohort's patients per (age band under the edges `(4, 6, 8)`, gender, label): the cell
`(6-7, female, 1)` holds twice the others' cases, so a 0.9 removal leaves it flagged."""

_DECADES: Final = {"<4": (2, 3), "4-5": (4, 5), "6-7": (6, 7), ">=8": (8, 9)}

_TOKENS: Final = (
    "250.41",
    "401",
    "410",
    "428.0",
    "434.91",
    "486",
    "491.21",
    "496",
    "585",
    "707.1",
    "8",
    "V45",
    "?",
    "276",
    "E888",
)
"""Diagnosis tokens: one in each mapped category, then tokens in none."""


def _toy_line(number: int, band: str, gender: str, label: int) -> str:
    decade = _DECADES[band][number % len(_DECADES[band])]
    states = ("No", "Steady", "Up", "Down")
    drugs = {name: states[(number // (position + 1)) % 4] for position, name in enumerate(DRUGS)}
    return diabetic_line(
        encounter_id=str(1000 + 7 * number),
        patient_nbr=str(500_000 + number),
        race=("Caucasian", "AfricanAmerican", "?", "Hispanic", "Other", "Asian")[number % 6],
        gender=gender,
        age=f"[{10 * decade}-{10 * (decade + 1)})",
        discharge_disposition_id="1",
        time_in_hospital=str(1 + number % 14),
        num_lab_procedures=str(1 + 7 * number % 80),
        num_procedures=str(number % 7),
        num_medications=str(1 + 3 * number % 40),
        number_outpatient=str(number % 4),
        number_emergency=str(number // 3 % 3),
        number_inpatient=str(number // 2 % 5),
        diag_1=_TOKENS[number % len(_TOKENS)],
        diag_2=_TOKENS[(3 * number + 1) % len(_TOKENS)],
        diag_3=_TOKENS[(5 * number + 2) % len(_TOKENS)],
        number_diagnoses=str(1 + number % 9),
        max_glu_serum=("None", "Norm", ">200", ">300")[number // 4 % 4],
        A1Cresult=("None", "Norm", ">7", ">8")[number % 4],
        **drugs,
        change="Ch" if drugs["insulin"] in {"Up", "Down"} else "No",
        diabetesMed="Yes" if set(drugs.values()) != {"No"} else "No",
        readmitted="<30" if label else ("NO", ">30")[number % 2],
    )


def toy_archive() -> bytes:
    """The toy cohort's archive: one adult first encounter per patient, `TOY_CELLS` patients."""
    lines: list[str] = []
    for (band, gender, label), count in TOY_CELLS.items():
        lines += [_toy_line(len(lines) + 1 + index, band, gender, label) for index in range(count)]
    return archive(text(*lines))


def _stanzas(concepts: dict[str, str], parent: str) -> list[str]:
    return [
        line
        for concept, label in concepts.items()
        for line in ("[Term]", f"id: {concept}", f"name: {label}", f"is_a: {parent}", "")
    ]


_MONDO: Final = "\n".join(
    [
        "format-version: 1.2",
        "data-version: releases/2026-01-01",
        "ontology: mondo",
        "",
        "[Term]",
        "id: MONDO:9000001",
        "name: disease",
        "",
        *_stanzas({row.concept: row.label for row in ICD9_MAPPINGS}, "MONDO:9000001"),
    ]
)

_HPO: Final = "\n".join(
    [
        "format-version: 1.2",
        "data-version: hp/releases/2026-01-01",
        "ontology: hp.obo",
        "",
        "[Term]",
        "id: HP:9000001",
        "name: Phenotypic abnormality",
        "",
        *_stanzas({row.concept: row.label for row in LAB_MAPPINGS}, "HP:9000001"),
    ]
)


def diabetes_releases() -> tuple[dict[str, Source], dict[str, bytes]]:
    """The MONDO- and HPO-shaped releases' pins by prefix, and their bodies by URL.

    Each names the concepts its Diabetes field rows map, under an invented root.
    """
    bodies = {"HP": _HPO.encode(), "MONDO": _MONDO.encode()}
    pins = {"HP": pin(bodies["HP"], name="hpo"), "MONDO": pin(bodies["MONDO"], name="mondo")}
    return pins, {pins[prefix].url: body for prefix, body in bodies.items()}
