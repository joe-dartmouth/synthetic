"""NHAMCS's field rows: the chronic-condition flags, then each mapped diagnosis token per column.

A flag maps, on code 1 (yes), to the live MONDO concept whose name or exact synonym is the
condition its item names, else HPO's; an item naming nested conditions maps the broader. Code 2,
the whole chronic item blank, shows nothing. Each ICD-10-CM id of the generated
`bko.ontology.icd10cm_codes` maps, in each diagnosis column, the number of the token that reads as
it. Labels are the pinned releases' names. ADR `research/nhamcs-frame` records the rows.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.icd10_tokens import token_number
from bko.datasets.nhamcs import DIAGNOSES
from bko.ontology.icd10 import Icd10Code
from bko.ontology.icd10cm_codes import ICD10CM_CODES
from bko.ontology.schema import Codes, FieldMapping


def _flag(field: str, concept: str, label: str, read: str) -> FieldMapping:
    return FieldMapping(
        when=(Codes(field, frozenset({1})),),
        concept=concept,
        label=label,
        source="nhamcs-nd",
        limits=(
            f"{field} 1, the provider ticked the item; {read}; 0 asserts no absence, and 2, the "
            "whole chronic item blank, shows nothing"
        ),
    )


FLAG_ROWS: Final[tuple[FieldMapping, ...]] = (
    _flag(
        "ALZHD", "MONDO:0001627", "dementia", "Alzheimer disease descends from dementia in MONDO"
    ),
    _flag("ASTHMA", "MONDO:0004979", "asthma", "the concept's name"),
    _flag("CANCER", "MONDO:0004992", "cancer", "the concept's name"),
    _flag(
        "CEBVD",
        "MONDO:0011057",
        "cerebrovascular disorder",
        "exact synonym cerebrovascular disease; stroke and TIA descend from it in MONDO",
    ),
    _flag("CKD", "MONDO:0005300", "chronic kidney disease", "the concept's name"),
    _flag("COPD", "MONDO:0005002", "chronic obstructive pulmonary disease", "the concept's name"),
    _flag("CHF", "MONDO:0005009", "congestive heart failure", "the concept's name"),
    _flag("DEPRN", "MONDO:0002050", "depressive disorder", "exact synonym depression"),
    _flag(
        "DIABTYP1",
        "MONDO:0005147",
        "type 1 diabetes mellitus",
        "exact synonym type I diabetes mellitus",
    ),
    _flag(
        "DIABTYP2",
        "MONDO:0005148",
        "type 2 diabetes mellitus",
        "exact synonym type II diabetes mellitus",
    ),
    _flag("DIABTYP0", "MONDO:0005015", "diabetes mellitus", "the concept's name; type unspecified"),
    _flag(
        "ESRD", "MONDO:0004375", "end stage renal failure", "exact synonym end stage renal disease"
    ),
    _flag(
        "EDHIV",
        "MONDO:0005109",
        "HIV infectious disease",
        "exact synonym HIV infection; AIDS descends from it in MONDO",
    ),
    _flag("HYPLIPID", "MONDO:0021187", "hyperlipidemia", "the concept's name"),
    _flag("HTN", "HP:0000822", "Hypertension", "HPO's name; MONDO holds no exact term"),
    _flag("OBESITY", "MONDO:0011122", "obesity disorder", "exact synonym obesity"),
    _flag(
        "OSA",
        "MONDO:0007147",
        "obstructive sleep apnea syndrome",
        "exact synonym obstructive sleep apnea",
    ),
    _flag("OSTPRSIS", "MONDO:0005298", "osteoporosis", "the concept's name"),
)
"""The chronic items with a concept, in file order; `ETOHAB`, `CAD`, `HPE` and `SUBSTAB` map
nothing."""

_POSITIONS: Final = dict(
    zip(DIAGNOSES, ("first", "second", "third", "fourth", "fifth"), strict=True)
)


def _diagnosis(row: Icd10Code, column: str) -> FieldMapping:
    return FieldMapping(
        when=(Codes(column, frozenset({token_number(row.token)})),),
        concept=row.concept,
        label=row.label,
        source="mondo",
        limits=(
            f"token {row.token}, ICD-10-CM {row.code}, as the visit's {_POSITIONS[column]} "
            f"diagnosis; the release carries {row.xref} on {row.concept} as equivalent; a token "
            "cut to four characters stands for every code below it"
        ),
    )


DIAGNOSIS_ROWS: Final[tuple[FieldMapping, ...]] = tuple(
    _diagnosis(row, column) for column in DIAGNOSES for row in ICD10CM_CODES
)
"""One `Codes(column, {token number})` row per diagnosis column and mapped id; each column's rows
partition its mapped tokens."""

NHAMCS_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = (*FLAG_ROWS, *DIAGNOSIS_ROWS)
"""The NHAMCS frame's field rows: the chronic flags, then the diagnoses."""
