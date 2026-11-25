"""Offline inputs for the TableShift partition tests: an invented `diabetic_data.csv` archive.

Every row is `tests._diabetes_fixtures.REAL_LAYOUT` with invented values. The archive holds
`KEPT` rows with a recorded race, `IN_DOMAIN` of them admitted from a source other than 7, and
`MISSING_RACE` rows whose race is `?`, spread through the file.
"""

from __future__ import annotations

from typing import Final

from tests._diabetes_fixtures import archive, diabetic_line, text


__all__ = ["IN_DOMAIN", "KEPT", "MISSING_RACE", "line", "tableshift_archive", "tableshift_lines"]

KEPT: Final = 250
"""The fixture's rows with a recorded race."""

IN_DOMAIN: Final = 200
"""The kept rows admitted from a source other than 7."""

MISSING_RACE: Final = 12
"""The rows whose race is `?`, which TableShift drops."""

_RACES: Final = ("Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian")
_GENDERS: Final = ("Female", "Male", "Female", "Male", "Unknown/Invalid")
_SOURCES: Final = ("1", "2", "4", "6", "17")
_TOKENS: Final = ("250.41", "401", "428.0", "8", "V45", "?", "E888", "486")
_STATES: Final = ("No", "Steady", "Up", "Down")
_READMITTED: Final = ("NO", ">30", "<30")


def line(number: int, *, race: str | None = None, ood: bool = False, **fields: str) -> str:
    """The fixture's row `number`, its fields varied by `number`, then `fields` in place.

    `ood` admits it from source 7.
    """
    decade = number % 10
    values = {
        "encounter_id": str(9000 + 11 * number),
        "patient_nbr": str(700_000 + number // 2),
        "race": _RACES[number % len(_RACES)] if race is None else race,
        "gender": _GENDERS[number % len(_GENDERS)] if number % 50 else "Unknown/Invalid",
        "age": f"[{10 * decade}-{10 * (decade + 1)})",
        "weight": ("?", "[75-100)", "[50-75)")[number % 3],
        "admission_type_id": str(1 + number % 3),
        "discharge_disposition_id": str(1 + number % 5),
        "admission_source_id": "7" if ood else _SOURCES[number % len(_SOURCES)],
        "time_in_hospital": str(1 + number % 14),
        "payer_code": ("?", "MC", "HM")[number % 3],
        "medical_specialty": ("?", "InternalMedicine", "Cardiology", "Surgery-General")[number % 4],
        "num_lab_procedures": str(1 + 7 * number % 80),
        "num_procedures": str(number % 7),
        "num_medications": str(1 + 3 * number % 40),
        "number_outpatient": str(number % 4),
        "number_emergency": str(number // 3 % 3),
        "number_inpatient": str(number // 2 % 5),
        "diag_1": _TOKENS[number % len(_TOKENS)],
        "diag_2": _TOKENS[(3 * number + 1) % len(_TOKENS)],
        "diag_3": _TOKENS[(5 * number + 2) % len(_TOKENS)],
        "number_diagnoses": str(1 + number % 9),
        "max_glu_serum": ("None", "Norm", ">200", ">300")[number // 4 % 4],
        "A1Cresult": ("None", "Norm", ">7", ">8")[number % 4],
        "metformin": _STATES[number % 4],
        "insulin": _STATES[number // 2 % 4],
        "acarbose": _STATES[number // 3 % 2],
        "change": ("No", "Ch")[number % 2],
        "diabetesMed": ("No", "Yes")[number // 2 % 2],
        "readmitted": _READMITTED[number % 3],
    }
    return diabetic_line(**(values | fields))


def tableshift_lines() -> list[str]:
    """The fixture's rows in file order: every 22nd row's race is `?`, and every fifth kept row
    is admitted from source 7, so the domains interleave with the drops."""
    lines: list[str] = []
    kept = 0
    for number in range(KEPT + MISSING_RACE):
        if number % 22 == 5:
            lines.append(line(number, race="?", ood=number % 2 == 1))
        else:
            lines.append(line(number, ood=kept % 5 == 4))
            kept += 1
    return lines


def tableshift_archive() -> bytes:
    """The fixture's archive."""
    return archive(text(*tableshift_lines()))
