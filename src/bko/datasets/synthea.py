"""A generated Synthea population, loaded as the patients alive on its snapshot date.

Only the columns the ablation and the cell counts read are loaded (`PATIENT_COLUMNS`,
`CONDITION_COLUMNS`), so the synthetic SSNs, names and addresses Synthea writes never enter a
frame. Codes are read as text and matched exactly; descriptions are not read. Dates must be
`YYYY-MM-DD`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL
from bko.datasets.errors import ContractError, UnavailableError


logger = logging.getLogger(__name__)

PATIENT_COLUMNS: Final = ("Id", "BIRTHDATE", "DEATHDATE", "GENDER")
"""The `patients.csv` columns read; every other column stays on disk."""

CONDITION_COLUMNS: Final = ("START", "PATIENT", "CODE")
"""The `conditions.csv` columns read."""

SEX_CODES: Final = {"F": 0, "M": 1}
"""Synthea's `GENDER` mapped to the UCI `sex` codes."""

SYNTHEA_AGE_BAND_EDGES: Final = (40, 50, 60, 70, 80)
"""Decade bands over the ages Synthea's ischemic heart disease model starts in (40 to 79)."""

ISCHEMIC_HEART_DISEASE: Final = "414545008"
"""SNOMED-CT Ischemic heart disease (disorder), as Synthea v4.0.0 codes it."""


@dataclass(frozen=True, slots=True, eq=False)
class Population:
    """Patients alive on `snapshot_date`, and their condition onsets.

    Attributes:
        patients: Indexed by patient id (text, sorted); `birth_date` (datetime64) and `sex` (0/1).
        conditions: `patient` (text), `code` (text) and `start` (datetime64), one row per onset.
        snapshot_date: The date ages are taken on.
    """

    patients: pd.DataFrame
    conditions: pd.DataFrame
    snapshot_date: date


def load_population(directory: Path, *, snapshot_date: date) -> Population:
    """Load the population in `directory/csv/` as the patients alive on `snapshot_date`.

    Args:
        directory: A population directory from `bko.datasets.synthea_run`.
        snapshot_date: The date ages are taken on.

    Returns:
        The population.

    Raises:
        UnavailableError: If a table is absent.
        ContractError: If a column is missing, an id repeats, a gender is not F or M, a condition
            names no patient, or a date is not `YYYY-MM-DD`.
    """
    patients = _read(directory / "csv" / "patients.csv", PATIENT_COLUMNS)
    conditions = _read(directory / "csv" / "conditions.csv", CONDITION_COLUMNS)
    if not patients["Id"].is_unique:
        msg = "patients.csv: duplicate Id"
        raise ContractError(msg)
    unknown = ~patients["GENDER"].isin(list(SEX_CODES))
    if unknown.any():
        msg = f"patients.csv: GENDER {patients['GENDER'][unknown].iloc[0]!r} is not F or M"
        raise ContractError(msg)
    orphans = ~conditions["PATIENT"].isin(patients["Id"])
    if orphans.any():
        msg = f"conditions.csv: {int(orphans.sum())} rows name no patient"
        raise ContractError(msg)
    birth = _dates(patients["BIRTHDATE"], "patients.csv BIRTHDATE")
    death = _dates(patients["DEATHDATE"], "patients.csv DEATHDATE")
    snapshot = pd.Timestamp(snapshot_date)
    alive = ((birth <= snapshot) & (death.isna() | (death > snapshot))).to_numpy()
    logger.info("patients=%d alive_on_snapshot=%d", len(patients), int(alive.sum()))
    people = pd.DataFrame(
        {
            "birth_date": birth[alive].to_numpy(),
            "sex": patients["GENDER"][alive].map(SEX_CODES).astype("int64").to_numpy(),
        },
        index=pd.Index(patients["Id"][alive].to_numpy(), name="patient"),
    ).sort_index()
    onsets = pd.DataFrame(
        {
            "patient": conditions["PATIENT"],
            "code": conditions["CODE"],
            "start": _dates(conditions["START"], "conditions.csv START"),
        }
    )
    onsets = onsets.loc[onsets["patient"].isin(people.index)].reset_index(drop=True)
    return Population(patients=people, conditions=onsets, snapshot_date=snapshot_date)


def ages(birth: pd.Series, on: date) -> pd.Series:
    """Return completed years from each `birth` to `on`."""
    before_birthday = (birth.dt.month > on.month) | (
        (birth.dt.month == on.month) & (birth.dt.day > on.day)
    )
    return (on.year - birth.dt.year - before_birthday.astype("int64")).astype("int64")


def subjects(population: Population, *, codes: frozenset[str]) -> pd.DataFrame:
    """Return one row per patient, shaped for `bko.datasets.cells`.

    Args:
        population: The population.
        codes: The condition codes that make a case.

    Returns:
        `age` on the snapshot date, `sex`, and `target` 1 where the patient had one of `codes` by
        then, indexed by patient id.
    """
    onsets = population.conditions
    snapshot = pd.Timestamp(population.snapshot_date)
    cases = onsets.loc[onsets["code"].isin(list(codes)) & (onsets["start"] <= snapshot), "patient"]
    people = population.patients
    return pd.DataFrame(
        {
            "age": ages(people["birth_date"], population.snapshot_date),
            "sex": people["sex"],
            LABEL: people.index.isin(cases.unique()).astype("int64"),
        },
        index=people.index,
    )


def _read(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=list(columns), dtype=str, keep_default_na=False)
    except FileNotFoundError as exc:
        msg = f"{path} is absent"
        raise UnavailableError(msg) from exc
    except ValueError as exc:
        msg = f"{path}: {exc}"
        raise ContractError(msg) from exc


def _dates(values: pd.Series, what: str) -> pd.Series:
    try:
        return pd.to_datetime(values.mask(values.eq("")), format="%Y-%m-%d")
    except ValueError as exc:
        msg = f"{what}: {exc}"
        raise ContractError(msg) from exc
