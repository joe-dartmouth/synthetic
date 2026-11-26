"""SUPPORT2's field rows: the disease groups, comorbidities and abnormal physiology it maps.

Each row is a `Codes` clause on a state `bko.datasets.support2` codes. The `dzgroup` levels that
name one disease map to MONDO, coma to HPO; the two multi-organ levels map nothing. `diabetes`,
`dementia` and `ca` map to MONDO. Each lab or vital coded at a cited cutoff maps its abnormal
states to HPO, so a missing value (state 0) shows no concept. The cutoffs are the SIRS criteria
(`bone-1992`) and SOFA's first abnormal score (`vincent-1996`), read in Singer et al.'s
reproduction (`singer-2016`, Box 1 and Table 1). `scoma` and the fields coded at tertiles map
nothing. Labels are the pinned releases' `name:` values. ADR `research/support2-frame` records
the rows.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.states import band_state
from bko.datasets.support2 import CA_CODES, DZGROUP_CODES, MEASURED
from bko.ontology.schema import Codes, FieldMapping


def _group(level: str, concept: str, label: str) -> FieldMapping:
    return FieldMapping(
        when=(Codes("dzgroup", frozenset({DZGROUP_CODES[level]})),),
        concept=concept,
        label=label,
        source="vanderbilt-2026",
        limits=(
            f"dzgroup {level}, the patient's disease group at study entry; a patient of another "
            "group may also hold it"
        ),
    )


def _comorbidity(field: str, codes: frozenset[int], concept: str, label: str) -> FieldMapping:
    return FieldMapping(
        when=(Codes(field, codes),),
        concept=concept,
        label=label,
        source="vanderbilt-2026",
        limits=f"{field} as SUPPORT recorded it at study entry; 0 or no asserts no absence",
    )


_READ_IN: Final = "read in singer-2016's reproduction"


def state_row(field: str, band: str, concept: str, label: str, criterion: str) -> FieldMapping:
    """The row of the state of `field` whose band is `band`, its source the reproduction read.

    The state is found by its band's name (`bko.datasets.states.band_state`), so a changed
    cutoff refuses the row instead of moving its concept to another band.

    Raises:
        ValueError: If `field`'s cutoffs cite no source, as tertiles do, or have no such band.
    """
    cut = MEASURED[field]
    if cut.source is None:
        msg = f"{field}'s cutoffs cite no source, so no row maps its states"
        raise ValueError(msg)
    return FieldMapping(
        when=(Codes(field, frozenset({band_state(cut, band)})),),
        concept=concept,
        label=label,
        source=cut.source,
        limits=(
            f"{field} {band} on study day 3, {criterion} ({_READ_IN}); one "
            "reading, not a diagnosis; a missing value is state 0 and shows nothing"
        ),
    )


_SIRS: Final = "a SIRS criterion of bone-1992"
_SOFA: Final = "SOFA's first abnormal score of vincent-1996"

SUPPORT2_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = (
    _group("COPD", "MONDO:0005002", "chronic obstructive pulmonary disease"),
    _group("CHF", "MONDO:0005009", "congestive heart failure"),
    _group("Cirrhosis", "MONDO:0005155", "cirrhosis of liver"),
    _group("Colon Cancer", "MONDO:0021063", "malignant colon neoplasm"),
    _group("Lung Cancer", "MONDO:0008903", "lung cancer"),
    _group("Coma", "HP:0001259", "Coma"),
    _comorbidity("diabetes", frozenset({1}), "MONDO:0005015", "diabetes mellitus"),
    _comorbidity("dementia", frozenset({1}), "MONDO:0001627", "dementia"),
    _comorbidity(
        "ca", frozenset({CA_CODES["yes"], CA_CODES["metastatic"]}), "MONDO:0004992", "cancer"
    ),
    state_row("hrt", "above 90", "HP:0001649", "Tachycardia", _SIRS),
    state_row("resp", "above 20", "HP:0002789", "Tachypnea", _SIRS),
    state_row("temp", "above 38", "HP:0001945", "Fever", _SIRS),
    state_row("temp", "below 36", "HP:0002045", "Hypothermia", _SIRS),
    state_row("wblc", "above 12", "HP:0001974", "Increased total leukocyte count", _SIRS),
    state_row("wblc", "below 4", "HP:0001882", "Decreased total leukocyte count", _SIRS),
    state_row("meanbp", "below 70", "HP:0002615", "Hypotension", _SOFA),
    state_row(
        "crea", "at least 1.2", "HP:0003259", "Elevated circulating creatinine concentration", _SOFA
    ),
    state_row("bili", "at least 1.2", "HP:0002904", "Hyperbilirubinemia", _SOFA),
    state_row("pafi", "below 400", "HP:0012418", "Hypoxemia", _SOFA),
)
"""The SUPPORT2 frame's field rows: the disease groups, the comorbidities, then the abnormal
states of the SIRS fields and the SOFA fields."""
