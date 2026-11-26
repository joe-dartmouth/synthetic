"""ACS's field rows: four disability items on HPO, and each mapped occupation code on OccO.

Each row is a `Codes` clause on a code `bko.datasets.acs` keeps. A disability item maps, on code 1
(yes), to the live HPO term whose name or exact synonym names its difficulty; `DDRS` and `DOUT`
map nothing. Each `OCCP` code of the generated `bko.ontology.occupation_codes_low` and
`bko.ontology.occupation_codes_high` maps to its OccO class. Labels are the pinned releases' names.
ADR `research/acs-frame` records the rows.
"""

from __future__ import annotations

from typing import Final

from bko.ontology.occupation_codes_high import OCCUPATION_CODES as HIGH
from bko.ontology.occupation_codes_low import OCCUPATION_CODES as LOW
from bko.ontology.schema import Codes, FieldMapping, OccupationCode


OCCUPATION_CODES: Final[tuple[OccupationCode, ...]] = (*LOW, *HIGH)
"""Every mapped `OCCP` code, in code order across both generated parts."""


def _disability(field: str, name: str, concept: str, label: str) -> FieldMapping:
    return FieldMapping(
        when=(Codes(field, frozenset({1})),),
        concept=concept,
        label=label,
        source="acs-nd",
        limits=f"{field} 1, yes to {name} in the 2024 data dictionary; 2, no, asserts no absence",
    )


def _occupation(row: OccupationCode) -> FieldMapping:
    return FieldMapping(
        when=(Codes("OCCP", frozenset({int(row.code)})),),
        concept=row.concept,
        label=row.label,
        source="occo",
        limits=(
            f"OCCP {row.code} reaches SOC id {row.soc} through the record's own SOCP code; an "
            "aggregate SOCP code reaches the class that holds every code it may stand for"
        ),
    )


DISABILITY_ROWS: Final[tuple[FieldMapping, ...]] = (
    _disability("DEAR", "Hearing difficulty", "HP:0000365", "Hearing impairment"),
    _disability("DEYE", "Vision difficulty", "HP:0000505", "Visual impairment"),
    _disability("DREM", "Cognitive difficulty", "HP:0100543", "Cognitive impairment"),
    _disability("DPHY", "Ambulatory difficulty", "HP:0001288", "Gait disturbance"),
)
"""The four disability items with an HPO term, each on its yes code."""

ACS_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = (
    *DISABILITY_ROWS,
    *(_occupation(row) for row in OCCUPATION_CODES),
)
"""The ACS frame's field rows: the disability items, then one row per mapped `OCCP` code, each
code in exactly one row."""
