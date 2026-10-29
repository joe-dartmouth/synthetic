"""An invented Synthea population for the frame and E3 tests, and a release over its concepts.

Every patient is invented. The population's snapshot is 2026-01-01: 310 patients alive and aged 40
and over, 35 per cell of E3's `(65,)` grid but 65 in `(>=65, female, 1)`, the removed cell, beside
15 patients under 40 and 10 who died before the snapshot. Condition codes are assigned by a
patient's number, so the indicators vary, and some onsets fall after the snapshot. The release
names each concept `bko.ontology.mapping.CODE_MAPPINGS` maps under an invented root; its labels
are those rows' labels, and no stanza copies release content.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import BinaryIO, Final

from bko.datasets.sources import Source
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE
from bko.datasets.synthea_run import SyntheaRun, ensure_population
from bko.ontology.mapping import CODE_MAPPINGS
from tests._datasets_fixtures import FakeSynthea, condition, patient, pin


FRAME_RUN: Final = SyntheaRun(
    seed=3, clinician_seed=4, population=300, snapshot_date=date(2026, 1, 1)
)
"""The run the invented population stands for."""

JAR: Final = b"not a jar"
JAR_PIN: Final = pin(JAR, "synthea-jar")
"""The jar the population's record names."""

CELLS: Final = {
    (False, "F", 0): 35,
    (False, "F", 1): 35,
    (False, "M", 0): 35,
    (False, "M", 1): 35,
    (True, "F", 0): 35,
    (True, "F", 1): 65,
    (True, "M", 0): 35,
    (True, "M", 1): 35,
}
"""Alive patients aged 40 and over per (65 or over, gender, case)."""

HYPERTENSION, DIABETES, LIPIDS, MI, STEMI, STROKE = (row.snomed for row in CODE_MAPPINGS[1:])
NSTEMI: Final = "401314000"


def _birth(age: int) -> str:
    """A birth date whose age on the snapshot is `age`."""
    return f"{2025 - age}-06-15"


def _conditions(pid: str, number: int, *, case: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if case:
        rows.append(condition(pid, ISCHEMIC_HEART_DISEASE, "2019-03-01"))
    for code, every in ((HYPERTENSION, 3), (DIABETES, 5), (LIPIDS, 4), (MI, 7), (STROKE, 11)):
        if number % every == 0:
            rows.append(condition(pid, code, "2015-05-01"))
    if number % 14 == 0:
        rows.append(condition(pid, STEMI, "2016-05-01"))
    if number % 6 == 0:
        rows.append(condition(pid, NSTEMI, "2017-05-01"))
    if number % 9 == 1:
        rows.append(condition(pid, STROKE, "2026-06-01"))
    return rows


def population_tables() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The patients and conditions tables, patients listed in an order other than their ids'."""
    patients: list[dict[str, str]] = []
    conditions: list[dict[str, str]] = []
    number = 0
    for (old, gender, case), count in CELLS.items():
        for index in range(count):
            number += 1
            age = 65 + index % 31 if old else 40 + index % 25
            pid = f"p{(number * 7919) % 100003:06d}"
            patients.append(patient(pid, _birth(age), gender))
            conditions += _conditions(pid, number, case=bool(case))
    for index in range(15):
        number += 1
        pid = f"y{index:03d}"
        patients.append(patient(pid, _birth(20 + index), "F" if index % 2 else "M"))
        conditions += _conditions(pid, number, case=index % 3 == 0)
    for index in range(10):
        number += 1
        pid = f"d{index:03d}"
        patients.append(patient(pid, _birth(70 + index), "F", death="2024-02-02"))
        conditions += _conditions(pid, number, case=True)
    return patients, conditions


def write_population(data_dir: Path) -> Path:
    """Write the population under `data_dir` as `ensure_population` publishes one, with no Java."""
    return ensure_population(
        FRAME_RUN,
        data_dir,
        jar_source=JAR_PIN,
        open_url=_jar_opener,
        resolve_java=lambda: (Path("/fake/java"), 'openjdk version "21"'),
        runner=FakeSynthea(tables=population_tables()),
    )


def _jar_opener(url: str, timeout: float) -> BinaryIO:
    """Serve the fake jar for any URL."""
    del url, timeout
    return io.BytesIO(JAR)


def _parent(snomed: str) -> str:
    """The release parent of the concept `snomed` maps to."""
    if snomed == STEMI:
        return "MONDO:0005068"
    if snomed in {ISCHEMIC_HEART_DISEASE, MI}:
        return "MONDO:9000002"
    return "MONDO:9000001"


RELEASE: Final = "\n".join(
    [
        "format-version: 1.2",
        "data-version: releases/2026-01-01",
        "ontology: mondo",
        "",
        "[Term]",
        "id: BFO:0000016",
        "name: disposition",
        "",
        "[Term]",
        "id: MONDO:9000001",
        "name: disease",
        "is_a: BFO:0000016",
        "",
        "[Term]",
        "id: MONDO:9000002",
        "name: heart disorder",
        "is_a: MONDO:9000001",
        "",
        *(
            line
            for row in CODE_MAPPINGS
            for line in (
                "[Term]",
                f"id: {row.concept}",
                f"name: {row.label}",
                f"is_a: {_parent(row.snomed)}",
                "",
            )
        ),
    ]
)
"""A MONDO-shaped release holding every concept `CODE_MAPPINGS` maps."""

RELEASE_PIN: Final[Source] = pin(RELEASE.encode(), name="mondo")
"""The release's pin."""
