"""The Synthea and Diabetes 130 frames, their indicators and their field rows.

Every stage may read them: they are built from `bko.ontology.mapping` and `bko.ontology.icd9`
over `bko.datasets`, and import no later stage, so `bko.gaps` may import them as
`bko.evaluate.data.load` does (ADR `research/data-contradiction-handling`).

The Synthea frame's indicators and field rows come from `bko.ontology.mapping.CODE_MAPPINGS`
(ADR `research/synthea-downstream-design`). The Diabetes 130 frame's coded columns carry the
archive's own state strings as their meanings, inverted from the codes `bko.datasets.diabetes`
gives them, so a prompt names each state as the archive writes it;
`bko.datasets.frames.check_records` refuses records whose columns or codes the frame does not
hold. `IDS_mapping.csv` names admission and discharge ids, none of which the frame keeps, so no
codebook reads it. Its indicators and field rows come from `bko.ontology.icd9`: one indicator and
one row per `ICD9_MAPPINGS` row, and one row per `LAB_MAPPINGS` row over its field's codes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.diabetes import ADULTS_FROM, COUNTS, DRUGS, STATE_CODES
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE
from bko.ontology.icd9 import ICD9_MAPPINGS, LAB_MAPPINGS, Icd9Mapping, LabMapping
from bko.ontology.mapping import CODE_MAPPINGS
from bko.ontology.schema import CodeMapping, Codes, FieldMapping


_PRESENCE: Final[Mapping[int, str]] = MappingProxyType({0: "absent", 1: "present"})


SYNTHEA_NAME: Final = "synthea"
"""The Synthea frame's name and its dataset's source name, as E2's fragments take it."""

SYNTHEA_ADULTS_FROM: Final = 40
"""The youngest age, in whole years on the snapshot date, a Synthea record has."""

INDICATOR_ROWS: Final[tuple[CodeMapping, ...]] = tuple(
    row for row in CODE_MAPPINGS if row.snomed != ISCHEMIC_HEART_DISEASE
)
"""The code rows that become the Synthea frame's indicators: every one but the label's."""


def indicator(row: CodeMapping | Icd9Mapping) -> str:
    """The indicator column of `row`: its label, lowercased, each run of other characters `_`."""
    return re.sub(r"[^a-z0-9]+", "_", row.label.lower()).strip("_")


def indicators(rows: Sequence[CodeMapping]) -> Mapping[str, str]:
    """Each row's indicator column and its condition code, in `rows`' order, frozen.

    Raises:
        ValueError: If two rows name one column, or a row names `age`, `sex` or the label.
    """
    found: dict[str, str] = {}
    for row in rows:
        name = indicator(row)
        if name in found:
            msg = f"indicator {name!r} is named twice"
            raise ValueError(msg)
        if name in {"age", "sex", LABEL}:
            msg = f"indicator {name!r} is one of the frame's own columns"
            raise ValueError(msg)
        found[name] = row.snomed
    return MappingProxyType(found)


SYNTHEA_INDICATORS: Final[Mapping[str, str]] = indicators(INDICATOR_ROWS)
"""Each indicator column's condition code, in `CODE_MAPPINGS` order; checked when the module is
imported."""

SYNTHEA_FRAME: Final = FrameSpec(
    name=SYNTHEA_NAME,
    features=("age", "sex", *SYNTHEA_INDICATORS),
    numeric=("age",),
    fractional=(),
    fraction_digits=0,
    categorical=("sex", *SYNTHEA_INDICATORS),
    codebook={"sex": SEXES, **dict.fromkeys(SYNTHEA_INDICATORS, _PRESENCE)},
    descriptions={
        "age": "age in years on the snapshot date",
        "sex": "sex",
        **{indicator(row): f"{row.label} recorded by the snapshot date" for row in INDICATOR_ROWS},
        LABEL: "ischemic heart disease recorded by the snapshot date",
    },
    ranges=(
        RangeConstraint(
            field="age",
            unit="a",
            low=SYNTHEA_ADULTS_FROM,
            high=120,
            source="research/synthea-downstream-design",
            limits="the frame keeps patients aged 40 and over; 120 is UCI's team-specified top",
        ),
    ),
    label_codes=tuple(sorted(LABEL_CODES)),
    label_meanings={0: "no ischemic heart disease", 1: "ischemic heart disease"},
    edges=(65,),
    study="ischemic heart disease prediction",
    cohort="patient aged 40 or over in a generated population",
)
"""E3's frame over the pinned Synthea population (ADR `research/synthea-downstream-design`)."""


def _field_row(row: CodeMapping) -> FieldMapping:
    column = LABEL if row.snomed == ISCHEMIC_HEART_DISEASE else indicator(row)
    return FieldMapping(
        when=(Codes(column, frozenset({1})),),
        concept=row.concept,
        label=row.label,
        source=row.source,
        limits="the code recorded by the snapshot date; a patient without it may still hold it",
    )


SYNTHEA_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = tuple(_field_row(r) for r in CODE_MAPPINGS)
"""The Synthea frame's field rows, one per `CODE_MAPPINGS` row: each indicator's
`Codes(column, {1})`, and the label's `Codes(target, {1})` for ischemic heart disease (ADR
`research/ontology-mapping-design`)."""


DIABETES_NAME: Final = "diabetes"
"""The Diabetes frame's name and its `Dataset` value."""


def indicator_codes(rows: Sequence[Icd9Mapping]) -> Mapping[str, tuple[str, ...]]:
    """Each row's indicator column and its ICD-9 categories, in row order.

    Raises:
        ValueError: If two rows' labels give one indicator name.
    """
    codes: dict[str, tuple[str, ...]] = {}
    for row in rows:
        name = indicator(row)
        if name in codes:
            msg = f"indicator {name!r} names 2 rows; give their labels distinct names"
            raise ValueError(msg)
        codes[name] = row.codes
    return MappingProxyType(codes)


DIABETES_INDICATORS: Final[Mapping[str, tuple[str, ...]]] = indicator_codes(ICD9_MAPPINGS)
"""Each indicator column's ICD-9 categories, in `ICD9_MAPPINGS` order, as `load_diabetes` takes
them."""

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
    """The field row of a lab row, refusing a field or state the frame does not code."""
    if row.field not in STATE_CODES:
        msg = f"lab row {row.label!r}: field {row.field!r} is not a coded frame column"
        raise ValueError(msg)
    codes = STATE_CODES[row.field]
    unknown = sorted(row.states - set(codes))
    if unknown:
        msg = f"lab row {row.label!r}: state {unknown[0]!r} is not one of {row.field}'s states"
        raise ValueError(msg)
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
