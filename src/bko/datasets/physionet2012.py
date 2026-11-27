"""PhysioNet/CinC Challenge 2012: one frame record per ICU stay, labelled by in-hospital death.

`load_physionet2012` reads the three sets' stays from their verified archives, never extracting
them (`bko.datasets.archive`), and each set's outcomes file. A stay is an ASCII CSV of
`Time,Parameter,Value` lines over its first 48 hours; `-1` means missing, and a 0 of a variable
of `IMPLAUSIBLE_ZERO` is no reading. Each field reduces its variables' readings, over the whole 48
hours and over the values at or above 0, to their minimum or their maximum, `map_min` over `MAP`
and `NIMAP` together, and is coded at its cited cutoff (`bko.datasets.states.coded`), `0` = not
measured. A stay whose `Age` or `Gender` is -1 is left out and counted; an empty `Parameter`,
which names no variable, is skipped and counted. The outcome-related descriptors but the label
never enter. ADR `research/physionet2012-frame` records the frame.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.archive import members, tar_members
from bko.datasets.cells import LABEL
from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.sources import Source
from bko.datasets.states import Cutoffs, coded


logger = logging.getLogger(__name__)

PHYSIONET2012_NAME: Final = "physionet2012"
"""The frame's name, its `Dataset` value and its catalog name."""

SETS: Final = ("a", "b", "c")
"""The three sets, in the order `load_physionet2012` reads their archives and outcomes."""

TARRED: Final = frozenset({"c"})
"""The sets served only as a gzipped tar; the others are zips."""

HEADER: Final = "Time,Parameter,Value"
"""A stay's header line."""

OUTCOMES_HEADER: Final = "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death"
"""An outcomes file's header line."""

DESCRIPTORS: Final = ("RecordID", "Age", "Gender", "Height", "ICUType", "Weight")
"""The general descriptors, recorded at `00:00`."""

ONCE: Final = DESCRIPTORS[:-1]
"""The descriptors a stay holds exactly once, at `00:00`; `Weight` is also a time series."""

SERIES: Final = (
    *("Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine"),
    *("DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg", "MAP"),
    *("MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets"),
    *("RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight"),
)
"""The 37 time series, as the files name them."""

ICU_TYPES: Final[Mapping[int, str]] = MappingProxyType(
    {
        1: "Coronary Care Unit",
        2: "Cardiac Surgery Recovery Unit",
        3: "Medical ICU",
        4: "Surgical ICU",
    }
)
"""`ICUType`'s codes and meanings, as the challenge page gives them."""

GENDERS: Final = frozenset({0, 1})
"""`Gender`'s codes, 0 female and 1 male, which are `sex`'s (`bko.datasets.cells.SEXES`)."""

MISSING: Final = -1.0
"""The value the files write for missing or unknown data."""

_SINGER: Final = "singer-2016"
REDUCED: Final[Mapping[str, tuple[tuple[str, ...], str, Cutoffs]]] = MappingProxyType(
    {
        "hr_max": (("HR",), "max", Cutoffs(above=(90.0,), source=_SINGER)),
        "resprate_max": (("RespRate",), "max", Cutoffs(above=(20.0,), source=_SINGER)),
        "temp_max": (("Temp",), "max", Cutoffs(above=(38.0,), source=_SINGER)),
        "temp_min": (("Temp",), "min", Cutoffs(at_or_above=(36.0,), source=_SINGER)),
        "wbc_max": (("WBC",), "max", Cutoffs(above=(12.0,), source=_SINGER)),
        "wbc_min": (("WBC",), "min", Cutoffs(at_or_above=(4.0,), source=_SINGER)),
        "map_min": (("MAP", "NIMAP"), "min", Cutoffs(at_or_above=(70.0,), source=_SINGER)),
        "creatinine_max": (("Creatinine",), "max", Cutoffs(at_or_above=(1.2,), source=_SINGER)),
        "bilirubin_max": (("Bilirubin",), "max", Cutoffs(at_or_above=(1.2,), source=_SINGER)),
        "platelets_min": (("Platelets",), "min", Cutoffs(at_or_above=(150.0,), source=_SINGER)),
        "gcs_min": (("GCS",), "min", Cutoffs(at_or_above=(15.0,), source=_SINGER)),
    }
)
"""Each coded field: the variables whose readings together it reduces, `max` or `min`, and its
cutoffs, the SIRS criteria (`bone-1992`) and SOFA's first abnormal score (`vincent-1996`) as
Singer et al. reproduce them (`singer-2016`, Box 1 and Table 1)."""

MEASURED: Final[Mapping[str, Cutoffs]] = MappingProxyType(
    {name: cut for name, (_, _, cut) in REDUCED.items()}
)
"""Every coded field, in record order, and its cutoffs."""

IMPLAUSIBLE_ZERO: Final[tuple[str, ...]] = (
    "HR",
    "Temp",
    "WBC",
    "MAP",
    "NIMAP",
    "Creatinine",
    "Bilirubin",
    "Platelets",
    "GCS",
)
"""The read variables whose recorded 0 is physiologically implausible, so no reading: `GCS`'s
scale runs from 3. `RespRate` keeps its 0, which a ventilated patient can record."""

MECHVENT: Final = "MechVent"
"""The variable `mechvent` reads: 1 when the stay holds a value of 1, else 0."""

READ: Final = frozenset(
    {variable for variables, _, _ in REDUCED.values() for variable in variables} | {MECHVENT}
)
"""The time series the frame reads."""

FEATURES: Final = ("age", "sex", "ICUType", *MEASURED, "mechvent")
"""The frame's features, in record order."""

_TIME: Final = re.compile(r"([0-9][0-9]):([0-5][0-9])")
_LAST_MINUTE: Final = 48 * 60
_NAME: Final = re.compile(r"set-([abc])/([0-9]+)\.txt")
_NUMBER: Final = re.compile(r"-?[0-9]+(\.[0-9]+)?(e\+[0-9]+)?")
"""A value as the files write it: decimal digits, or the exponent form such as `1.213e+04`."""


@dataclass(slots=True)
class Stay:
    """One stay as read: its descriptors and the readings of each variable it reads.

    Attributes:
        label: Its set, `a`, `b` or `c`, from its member's name.
        record_id: Its `RecordID`, equal to its member's name.
        age: `Age`, -1 when unknown.
        gender: `Gender`, -1 when unknown.
        icu_type: `ICUType`.
        values: Each read variable's readings, in file order: its values at or above 0, but a
            0 of `IMPLAUSIBLE_ZERO`.
        unnamed: Its lines with an empty `Parameter`, skipped.
        missing: Its lines of a read variable whose value is -1, missing by the page's rule.
        below_zero: Its lines of a read variable whose value is below 0 but not -1. Only
            `MechVent`'s values are checked against a set of codes; any other read variable's
            value below 0 is counted here, not refused, and no value is checked against a range.
        zeros: Per variable of `IMPLAUSIBLE_ZERO` holding one, its lines whose value is 0.
    """

    label: str
    record_id: int
    age: float
    gender: float
    icu_type: float
    values: dict[str, list[float]] = field(default_factory=dict)
    unnamed: int = 0
    missing: int = 0
    below_zero: int = 0
    zeros: dict[str, int] = field(default_factory=dict)

    def reduced(self, variables: tuple[str, ...], how: str) -> float:
        """The minimum or maximum of `variables`' readings together; NaN if none."""
        values = [value for variable in variables for value in self.values.get(variable, [])]
        if not values:
            return math.nan
        return max(values) if how == "max" else min(values)


def _value(where: str, text: str) -> float:
    if not _NUMBER.fullmatch(text):
        msg = f"{where}: value {text!r} is not a decimal number"
        raise ContractError(msg)
    value = float(text)
    if not math.isfinite(value):
        msg = f"{where}: value {text!r} is not finite"
        raise ContractError(msg)
    return value


def _minute(where: str, text: str) -> int:
    match = _TIME.fullmatch(text)
    minute = int(match[1]) * 60 + int(match[2]) if match else _LAST_MINUTE + 1
    if minute > _LAST_MINUTE:
        msg = f"{where}: time {text!r} is not hh:mm from 00:00 to 48:00"
        raise ContractError(msg)
    return minute


def _read(
    where: str,
    parameter: str,
    value: float,
    values: dict[str, list[float]],
    counts: dict[str, int],
    zeros: dict[str, int],
) -> None:
    """File one value of a read variable: a reading, a listed 0, a -1 or another value below 0."""
    if parameter == MECHVENT and value not in {0.0, 1.0, MISSING}:
        msg = f"{where}: MechVent is {value:g}, not 0, 1 or -1"
        raise ContractError(msg)
    if value == 0 and parameter in IMPLAUSIBLE_ZERO:
        zeros[parameter] = zeros.get(parameter, 0) + 1
    elif value >= 0:
        values[parameter].append(value)
    else:
        counts["missing" if value == MISSING else "below_zero"] += 1


def parse_stay(name: str, text: str) -> Stay:
    """Read one stay's member `name`, `set-x/<RecordID>.txt`, and its text.

    Raises:
        ContractError: If the name is not a stay's, the text is not ASCII, does not end in a
            newline or start with `HEADER`, a line does not hold three fields, a time from `00:00`
            to `48:00` and a finite decimal number (`_NUMBER`), a `Parameter` is neither empty
            nor a descriptor or time series, a descriptor of `ONCE` is missing, repeated or not at
            `00:00`, `RecordID` is not the name's, `Age` is not whole, `Gender` is not 0, 1 or -1,
            `ICUType` is not 1 to 4, or `MechVent` is not 0, 1 or -1.
    """
    named = _NAME.fullmatch(name)
    if named is None:
        msg = f"{name}: not a stay's name set-x/<RecordID>.txt"
        raise ContractError(msg)
    if not text.isascii():
        msg = f"{name}: not ASCII text"
        raise ContractError(msg)
    if not text.endswith("\n"):
        msg = f"{name}: does not end in a newline"
        raise ContractError(msg)
    lines = text[:-1].split("\n")
    if lines[0] != HEADER:
        msg = f"{name}: the header is not {HEADER}"
        raise ContractError(msg)
    found: dict[str, float] = {}
    values: dict[str, list[float]] = {variable: [] for variable in READ}
    counts = {"unnamed": 0, "missing": 0, "below_zero": 0}
    zeros: dict[str, int] = {}
    for number, line in enumerate(lines[1:], start=2):
        where = f"{name} line {number}"
        fields = line.split(",")
        if len(fields) != 3:
            msg = f"{where}: {len(fields)} fields, not 3"
            raise ContractError(msg)
        minute = _minute(where, fields[0])
        parameter, value = fields[1], _value(where, fields[2])
        if not parameter:
            counts["unnamed"] += 1
        elif parameter in ONCE:
            if minute or parameter in found:
                msg = f"{where}: {parameter} must appear once, at 00:00"
                raise ContractError(msg)
            found[parameter] = value
        elif parameter not in SERIES:
            msg = f"{where}: {parameter!r} is no descriptor or time series"
            raise ContractError(msg)
        elif parameter in READ:
            _read(where, parameter, value, values, counts, zeros)
    return _checked(name, (named[1], int(named[2])), found, values, counts, zeros)


def _checked(
    name: str,
    member: tuple[str, int],
    found: Mapping[str, float],
    values: dict[str, list[float]],
    counts: Mapping[str, int],
    zeros: dict[str, int],
) -> Stay:
    """The stay from its parsed descriptors, refusing one that is missing or out of its codes."""
    absent = [descriptor for descriptor in ONCE if descriptor not in found]
    if absent:
        msg = f"{name}: no {absent[0]} at 00:00"
        raise ContractError(msg)
    label, record_id = member
    checks = (
        ("RecordID", found["RecordID"] == record_id, f"not the name's {record_id}"),
        (
            "Age",
            found["Age"] == MISSING or (found["Age"] >= 0 and found["Age"].is_integer()),
            "not a whole age or -1",
        ),
        ("Gender", found["Gender"] in {*GENDERS, MISSING}, "not 0, 1 or -1"),
        ("ICUType", found["ICUType"] in set(ICU_TYPES), "not 1 to 4"),
    )
    for descriptor, fits, why in checks:
        if not fits:
            msg = f"{name}: {descriptor} is {found[descriptor]:g}, {why}"
            raise ContractError(msg)
    return Stay(
        label=label,
        record_id=record_id,
        age=found["Age"],
        gender=found["Gender"],
        icu_type=found["ICUType"],
        values=values,
        unnamed=counts["unnamed"],
        missing=counts["missing"],
        below_zero=counts["below_zero"],
        zeros=zeros,
    )


def parse_outcomes(text: str, where: str) -> dict[int, int]:
    """Each `RecordID` of an outcomes file and its `In-hospital_death`; nothing else is read.

    Raises:
        ContractError: If the text does not end in a newline or start with `OUTCOMES_HEADER`, a
            line does not hold six fields, its `RecordID` is not whole or repeats, or its death
            is not 0 or 1.
    """
    if not text.endswith("\n"):
        msg = f"{where}: does not end in a newline"
        raise ContractError(msg)
    lines = text[:-1].split("\n")
    if lines[0] != OUTCOMES_HEADER:
        msg = f"{where}: the header is not {OUTCOMES_HEADER}"
        raise ContractError(msg)
    deaths: dict[int, int] = {}
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        if len(fields) != 6 or not fields[0].isdigit() or fields[5] not in {"0", "1"}:
            msg = f"{where} line {number}: not a RecordID, four fields and a death of 0 or 1"
            raise ContractError(msg)
        record_id = int(fields[0])
        if record_id in deaths:
            msg = f"{where} line {number}: RecordID {record_id} is listed more than once"
            raise ContractError(msg)
        deaths[record_id] = int(fields[5])
    return deaths


def _stays(data: bytes, label: str, source: Source) -> Iterator[Stay]:
    pattern = re.compile(rf"set-{label}/[0-9]+\.txt")
    read = tar_members if label in TARRED else members
    for name, text in read(data, pattern, source):
        yield parse_stay(name, text)


def _matched(label: str, stays: Sequence[Stay], deaths: Mapping[int, int]) -> None:
    """Refuse a stay of set `label` with no outcome line, or an outcome line with no stay."""
    ids = {stay.record_id for stay in stays}
    no_outcome = sorted(ids - set(deaths))
    if no_outcome:
        msg = f"set {label}: stay {no_outcome[0]} has no outcome line"
        raise ContractError(msg)
    no_stay = sorted(set(deaths) - ids)
    if no_stay:
        msg = f"set {label}: the outcome line of {no_stay[0]} names no stay"
        raise ContractError(msg)


def frame_records(stays: Sequence[Stay], deaths: Mapping[int, int]) -> pd.DataFrame:
    """Return the records of the stays whose age and sex are known, in `RecordID` order.

    Logs a `zeros` line per variable of `IMPLAUSIBLE_ZERO`, its 0 lines and the stays holding one
    (`records`), counted over every stay as `missing` and `below_zero` are, then one `cohort`
    line: the stays, in all and per set, those of unknown age and of unknown sex, the stays left
    out, the records and their deaths, the skipped empty-`Parameter` lines, the read values of -1
    and those below 0 but not -1, per coded field the records it was measured on, the records
    whose `map_min` is a `NIMAP` reading with no `MAP` reading (`map_min_nimap`), those holding
    both whose lowest `NIMAP` is strictly the lower (`map_min_nimap_lower`) and those with neither
    (`map_unmeasured`), and the records ventilated.

    Returns:
        `FEATURES` and `LABEL`, all int64, indexed by `record_id`.
    """
    kept = sorted(
        (stay for stay in stays if MISSING not in {stay.age, stay.gender}),
        key=lambda stay: stay.record_id,
    )
    index = pd.Index([stay.record_id for stay in kept], name="record_id", dtype="int64")
    columns: dict[str, pd.Series] = {
        "age": pd.Series([stay.age for stay in kept], index=index),
        "sex": pd.Series([stay.gender for stay in kept], index=index),
        "ICUType": pd.Series([stay.icu_type for stay in kept], index=index),
        **{
            name: coded(
                pd.Series([stay.reduced(variables, how) for stay in kept], index=index),
                at_or_above=cut.at_or_above,
                above=cut.above,
            )
            for name, (variables, how, cut) in REDUCED.items()
        },
        "mechvent": pd.Series([int(1.0 in stay.values[MECHVENT]) for stay in kept], index=index),
        LABEL: pd.Series([deaths[stay.record_id] for stay in kept], index=index),
    }
    records = pd.DataFrame(columns, index=index).astype("int64")
    for variable in IMPLAUSIBLE_ZERO:
        logger.info(
            "zeros dataset=%s field=%s readings=%d records=%d",
            PHYSIONET2012_NAME,
            variable,
            sum(stay.zeros.get(variable, 0) for stay in stays),
            sum(variable in stay.zeros for stay in stays),
        )
    arterial = [bool(stay.values["MAP"]) for stay in kept]
    cuff = [bool(stay.values["NIMAP"]) for stay in kept]
    logger.info(
        "cohort dataset=%s stays=%d %s unknown_age=%d unknown_sex=%d left_out=%d records=%d "
        "died=%d unnamed=%d missing=%d below_zero=%d %s map_min_nimap=%d map_min_nimap_lower=%d "
        "map_unmeasured=%d mechvent=%d",
        PHYSIONET2012_NAME,
        len(stays),
        " ".join(f"stays_{label}={sum(s.label == label for s in stays)}" for label in SETS),
        sum(stay.age == MISSING for stay in stays),
        sum(stay.gender == MISSING for stay in stays),
        len(stays) - len(kept),
        len(records),
        int(records[LABEL].sum()),
        sum(stay.unnamed for stay in stays),
        sum(stay.missing for stay in stays),
        sum(stay.below_zero for stay in stays),
        " ".join(f"measured_{name}={int((records[name] != 0).sum())}" for name in MEASURED),
        sum(c and not a for a, c in zip(arterial, cuff, strict=True)),
        sum(
            a and c and min(stay.values["NIMAP"]) < min(stay.values["MAP"])
            for stay, a, c in zip(kept, arterial, cuff, strict=True)
        ),
        sum(not (a or c) for a, c in zip(arterial, cuff, strict=True)),
        int(records["mechvent"].sum()),
    )
    return records


def _ascii(data: bytes, source: Source) -> str:
    try:
        return data.decode("ascii")
    except UnicodeDecodeError as exc:
        msg = f"{source.name}: not ASCII text: {exc}"
        raise ContractError(msg) from exc


def load_physionet2012(data_dir: Path, *, sources: Sequence[Source]) -> pd.DataFrame:
    """Load the frame records from the verified snapshots of `sources`.

    This never fetches; `bko.datasets.fetch.ensure` does.

    Args:
        data_dir: The data root.
        sources: The three sets' archives in `SETS` order, then their outcomes files.

    Raises:
        ValueError: If `sources` holds other than six sources.
        UnavailableError: If a snapshot is absent.
        ContractError: If a file contradicts its pin or schema, a `RecordID` repeats across the
            sets, a set's stays and outcome lines differ, or the cohort holds no record or no
            death.
    """
    if len(sources) != 2 * len(SETS):
        msg = f"PhysioNet 2012 reads {2 * len(SETS)} files, not {len(sources)}"
        raise ValueError(msg)
    stays: list[Stay] = []
    deaths: dict[int, int] = {}
    for label, archive, outcomes in zip(SETS, sources[:3], sources[3:], strict=True):
        read = list(_stays(read_verified(archive, data_dir), label, archive))
        found = parse_outcomes(_ascii(read_verified(outcomes, data_dir), outcomes), outcomes.name)
        _matched(label, read, found)
        repeated = sorted({stay.record_id for stay in read} & set(deaths))
        if repeated:
            msg = f"set {label}: RecordID {repeated[0]} is a stay of an earlier set"
            raise ContractError(msg)
        stays.extend(read)
        deaths.update(found)
    records = frame_records(stays, deaths)
    if records.empty:
        msg = f"{PHYSIONET2012_NAME}: the cohort holds no records"
        raise ContractError(msg)
    if not records[LABEL].any():
        msg = f"{PHYSIONET2012_NAME}: the cohort holds no in-hospital death"
        raise ContractError(msg)
    return records
