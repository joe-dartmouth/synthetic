"""NHANES 2021-2023: one frame record per adult, labelled by a cardiovascular composite.

The survey is NHANES August 2021 to August 2023. `load_nhanes` reads seven SAS XPORT component
files from their verified snapshots, each whole from its bytes by `pandas.read_sas`, never written
elsewhere. Each file's columns must be the
published file's in order, and its respondent sequence numbers, `SEQN`, whole and unique. The
demographics and the medical-conditions file join inner on `SEQN`, the exam and laboratory files
left, so an adult who was only interviewed stays a record with every measurement not measured.
The cohort keeps adults of 20 and over, the universe the four items are asked of. `LABEL` is 1
when any of the four items is 1 (yes), 0 when all four are 2 (no), and the record is left out
otherwise. A record holds `age` as published (top-coded at 80), `sex`, and each lab or vital as a
coded state (`bko.datasets.states.coded`). The four items and the survey weights never enter. ADR
`research/nhanes-frame` records the frame.
"""

from __future__ import annotations

import io
import logging
import struct
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from bko.datasets.cells import LABEL
from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.sources import Source
from bko.datasets.states import Cutoffs, coded


logger = logging.getLogger(__name__)

NHANES_NAME: Final = "nhanes"
"""The frame's name, its `Dataset` value and its catalog name."""

COMPONENTS: Final = ("DEMO_L", "MCQ_L", "BPXO_L", "BMX_L", "TCHOL_L", "HDL_L", "GHB_L")
"""The component files, in the order `load_nhanes` reads its sources."""

COLUMNS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "DEMO_L": (
            *("SEQN", "SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDAGEMN", "RIDRETH1"),
            *("RIDRETH3", "RIDEXMON", "RIDEXAGM", "DMQMILIZ", "DMDBORN4", "DMDYRUSR", "DMDEDUC2"),
            *("DMDMARTZ", "RIDEXPRG", "DMDHHSIZ", "DMDHRGND", "DMDHRAGZ", "DMDHREDZ", "DMDHRMAZ"),
            *("DMDHSEDZ", "WTINT2YR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU", "INDFMPIR"),
        ),
        "MCQ_L": (
            *("SEQN", "MCQ010", "MCQ035", "MCQ040", "MCQ050", "AGQ030", "MCQ053", "MCQ149"),
            *("MCQ160A", "MCQ195", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"),
            *("MCQ160M", "MCQ170M", "MCQ160P", "MCQ160L", "MCQ170L", "MCQ500", "MCQ510A"),
            *("MCQ510B", "MCQ510C", "MCQ510D", "MCQ510E", "MCQ510F", "MCQ550", "MCQ560"),
            *("MCQ220", "MCQ230A", "MCQ230B", "MCQ230C", "MCQ230D", "OSQ230"),
        ),
        "BPXO_L": (
            *("SEQN", "BPAOARM", "BPAOCSZ", "BPXOSY1", "BPXODI1", "BPXOSY2", "BPXODI2"),
            *("BPXOSY3", "BPXODI3", "BPXOPLS1", "BPXOPLS2", "BPXOPLS3"),
        ),
        "BMX_L": (
            *("SEQN", "BMDSTATS", "BMXWT", "BMIWT", "BMXRECUM", "BMIRECUM", "BMXHEAD", "BMIHEAD"),
            *("BMXHT", "BMIHT", "BMXBMI", "BMDBMIC", "BMXLEG", "BMILEG", "BMXARML", "BMIARML"),
            *("BMXARMC", "BMIARMC", "BMXWAIST", "BMIWAIST", "BMXHIP", "BMIHIP"),
        ),
        "TCHOL_L": ("SEQN", "WTPH2YR", "LBXTC", "LBDTCSI"),
        "HDL_L": ("SEQN", "WTPH2YR", "LBDHDD", "LBDHDDSI"),
        "GHB_L": ("SEQN", "WTPH2YR", "LBXGH"),
    }
)
"""Each component file's columns, in file order; a file must hold exactly these."""

QUESTIONS: Final = ("MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F")
"""Ever told of congestive heart failure, coronary heart disease, a heart attack and a stroke."""

YES: Final = 1.0
NO: Final = 2.0
ANSWERS: Final = frozenset({YES, NO, 7.0, 9.0})
"""An item's codes: 1 yes, 2 no, 7 refused, 9 don't know; a missing answer is NaN."""

ADULT_AGE: Final = 20
"""The youngest age kept: the four items are asked of respondents 20 and over."""

TOP_CODE: Final = 80
"""`RIDAGEYR`'s top code: 80 stands for 80 years and over."""

SEX_CODES: Final[Mapping[float, int]] = MappingProxyType({1.0: 1, 2.0: 0})
"""`RIAGENDR` 1 (male) and 2 (female) and their codes (`bko.datasets.cells.SEXES`)."""

READINGS: Final[Mapping[str, tuple[str, str, tuple[str, ...]]]] = MappingProxyType(
    {
        "BPXOSY": ("BPXO_L", "mean", ("BPXOSY1", "BPXOSY2", "BPXOSY3")),
        "BPXODI": ("BPXO_L", "mean", ("BPXODI1", "BPXODI2", "BPXODI3")),
        "LBXGH": ("GHB_L", "value", ("LBXGH",)),
        "LBXTC": ("TCHOL_L", "value", ("LBXTC",)),
        "LBDHDD": ("HDL_L", "value", ("LBDHDD",)),
        "BMXBMI": ("BMX_L", "value", ("BMXBMI",)),
    }
)
"""Each coded field's component file, how it is read, and the columns it reads: a blood pressure
is the mean of the readings present."""


def check_readings(
    readings: Mapping[str, tuple[str, str, tuple[str, ...]]],
    columns: Mapping[str, tuple[str, ...]],
) -> None:
    """Refuse a reading whose component file or column the published files do not hold.

    Raises:
        ValueError: Naming the first field whose component is not in `columns`, or whose column
            is not among its component's columns.
    """
    for field, (component, _, read) in readings.items():
        if component not in columns:
            msg = f"{field}: {component} is not a component file"
            raise ValueError(msg)
        absent = [name for name in read if name not in columns[component]]
        if absent:
            msg = f"{field}: {component} holds no column {absent[0]}"
            raise ValueError(msg)


check_readings(READINGS, COLUMNS)

CITED: Final[Mapping[str, Cutoffs]] = MappingProxyType(
    {
        "BPXOSY": Cutoffs(at_or_above=(130.0,), source="muntner-2018"),
        "BPXODI": Cutoffs(at_or_above=(80.0,), source="muntner-2018"),
        "LBXGH": Cutoffs(at_or_above=(5.7, 6.5), source="ada-2026"),
        "LBXTC": Cutoffs(at_or_above=(240.0,), source="nhlbi-2001"),
        "LBDHDD": Cutoffs(at_or_above=(40.0,), source="nhlbi-2001"),
    }
)
"""The labs and vitals coded at cited cutoffs, each the source it is read in: the 2017 ACC/AHA
definition of hypertension (`whelton-2018`) in `muntner-2018`'s Table 1, the ADA's prediabetes and
diabetes ranges (`ada-2026`, Tables 2.1 and 2.2), and ATP III's high total and low HDL cholesterol
(`ncep-2001`) in `nhlbi-2001`'s Table 2."""

TERTILES: Final[Mapping[str, Cutoffs]] = MappingProxyType(
    {"BMXBMI": Cutoffs(at_or_above=(23.4, 29.6))}
)
"""The numeric fields with missing values and no cited cutoff, coded at the tertiles
(`bko.datasets.states.tertiles`) of their measured values over every record of the published
file, each value read and each tertile rounded at `DECIMALS`; their cutoffs cite no source."""

MEASURED: Final[Mapping[str, Cutoffs]] = MappingProxyType({**CITED, **TERTILES})
"""Every coded measured field, in record order, and its cutoffs."""

NO_RESULT: Final = frozenset({"LBXGH"})
"""The fields whose codebook gives 0 as "No Lab Result", read as not measured."""

DECIMALS: Final = 2
"""The decimals a measured value, and a mean of readings, is read at before it is coded: XPORT
stores IBM floats, so a value such as 5.7 may read a hair below itself."""

FEATURES: Final = ("age", "sex", *MEASURED)
"""The frame's features, in record order."""

_READ_ERRORS: Final = (
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    OSError,
    StopIteration,
    struct.error,
    UserWarning,
)
"""What `pandas.read_sas` raises on bytes that are not a readable XPORT file, its warning that a
file may be corrupted raised as an error."""

_CORRUPTED: Final = "xport file may be corrupted"
"""The start of the warning `pandas.read_sas` gives on a file cut inside its observations."""


def read_component(data: bytes, component: str) -> pd.DataFrame:
    """Read one component file's XPORT bytes, after checking its columns and its `SEQN`s.

    Returns:
        Every column but `SEQN`, as read, indexed by `record_id`, the whole `SEQN`, ascending.

    Raises:
        ContractError: If the bytes are not a readable XPORT file, `pandas.read_sas` warns that
            they may be corrupted, its columns are not `COLUMNS[component]` in order, or a `SEQN`
            is missing, not whole or repeated.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=_CORRUPTED)
            frame = pd.read_sas(io.BytesIO(data), format="xport")
    except _READ_ERRORS as exc:
        msg = f"{component}: not a readable XPORT file: {type(exc).__name__}: {exc}"
        raise ContractError(msg) from exc
    if tuple(frame.columns) != COLUMNS[component]:
        msg = f"{component}: the columns are not the published file's {len(COLUMNS[component])}"
        raise ContractError(f"{msg} names in order")
    seqn = frame["SEQN"].to_numpy(dtype="float64")
    bad = ~np.isfinite(seqn) | (seqn != np.floor(seqn))
    if bad.any():
        first = int(bad.argmax())
        msg = f"{component} row {first + 1}: SEQN {float(seqn[first])} is not whole"
        raise ContractError(msg)
    index = pd.Index(seqn.astype("int64"), name="record_id")
    if index.has_duplicates:
        msg = f"{component}: respondent {index[index.duplicated()][0]} is listed more than once"
        raise ContractError(msg)
    return frame.drop(columns="SEQN").set_axis(index).sort_index(kind="stable")


def measured(values: pd.Series) -> pd.Series:
    """A measured column as float64 at `DECIMALS`, a missing value as NaN."""
    return values.astype("float64").round(DECIMALS)


def reading(joined: pd.DataFrame, field: str) -> pd.Series:
    """`field`'s value on each joined record at `DECIMALS`, NaN where it was not measured."""
    _, how, columns = READINGS[field]
    if how == "mean":
        return measured(joined[list(columns)].astype("float64").mean(axis=1)).rename(field)
    [column] = columns
    values = measured(joined[column])
    if field in NO_RESULT:
        values = values.mask(values == 0)
    return values.rename(field)


def _checked(joined: pd.DataFrame) -> None:
    """Refuse a joined record whose read values contradict the codebooks, naming the first."""
    age = measured(joined["RIDAGEYR"])
    checks: list[tuple[str, pd.Series, str]] = [
        ("RIDAGEYR", (age == age.round()) & age.between(0, TOP_CODE), "not a whole age 0 to 80"),
        ("RIAGENDR", joined["RIAGENDR"].isin(list(SEX_CODES)), "not 1 or 2"),
        *(
            (name, joined[name].isin(list(ANSWERS)) | joined[name].isna(), "not 1, 2, 7, 9 or .")
            for name in QUESTIONS
        ),
        *(
            (name, ~(joined[name].astype("float64") < 0), "negative")
            for _, _, columns in READINGS.values()
            for name in columns
        ),
    ]
    for name, fits, why in checks:
        bad = ~fits.to_numpy(dtype=bool)
        if bad.any():
            first = int(bad.argmax())
            value = float(joined[name].iloc[first])
            msg = f"respondent {joined.index[first]}: {name} is {value}, {why}"
            raise ContractError(msg)


def _within_demographics(components: Mapping[str, pd.DataFrame]) -> None:
    """Refuse a component file listing a respondent the demographics file does not."""
    demo = components["DEMO_L"].index
    for name in COMPONENTS[1:]:
        absent = components[name].index.difference(demo)
        if len(absent):
            msg = f"{name}: {len(absent)} respondents are absent from DEMO_L, first {absent[0]}"
            raise ContractError(msg)


def frame_records(components: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the frame records of the adults whose answers settle the label, in `SEQN` order.

    Args:
        components: Each component file's output of `read_component`, by `COMPONENTS` name.

    Logs one `cohort` line: the demographics records, those with a medical-conditions record
    (`with_conditions`), the adults among them, each label's count, the adults left out, and per
    coded field the kept records it was measured on.

    Returns:
        `FEATURES` and `LABEL`, all int64, indexed by `record_id`.

    Raises:
        ContractError: If a component file lists a respondent the demographics file does not, or
            a joined value contradicts its codebook.
    """
    _within_demographics(components)
    demo = components["DEMO_L"]
    joined = demo[["RIAGENDR", "RIDAGEYR"]].join(components["MCQ_L"][list(QUESTIONS)], how="inner")
    for name in COMPONENTS[2:]:
        read = [c for component, _, cs in READINGS.values() if component == name for c in cs]
        joined = joined.join(components[name][read], how="left")
    _checked(joined)
    adults = joined.loc[measured(joined["RIDAGEYR"]) >= ADULT_AGE]
    answers = adults[list(QUESTIONS)]
    any_yes = (answers == YES).any(axis=1)
    all_no = (answers == NO).all(axis=1)
    kept = adults.loc[any_yes | all_no]
    columns: dict[str, pd.Series] = {
        "age": measured(kept["RIDAGEYR"]),
        "sex": kept["RIAGENDR"].map(SEX_CODES),
        **{
            field: coded(reading(kept, field), at_or_above=cut.at_or_above, above=cut.above)
            for field, cut in MEASURED.items()
        },
        LABEL: any_yes.loc[kept.index].astype("int64"),
    }
    records = pd.DataFrame(columns).astype("int64")
    records.index.name = "record_id"
    logger.info(
        "cohort dataset=%s demographics=%d with_conditions=%d adults=%d any_yes=%d all_no=%d "
        "left_out=%d records=%d composite=%d %s",
        NHANES_NAME,
        len(demo),
        len(joined),
        len(adults),
        int(any_yes.sum()),
        int(all_no.sum()),
        len(adults) - len(kept),
        len(records),
        int(records[LABEL].sum()),
        " ".join(f"measured_{field}={int((records[field] != 0).sum())}" for field in MEASURED),
    )
    return records


def load_nhanes(data_dir: Path, *, sources: Sequence[Source]) -> pd.DataFrame:
    """Load the frame records from the verified snapshots of `sources`, one per component.

    This never fetches; `bko.datasets.fetch.ensure` does. Each file is read from its bytes once
    they match the pin.

    Args:
        data_dir: The data root.
        sources: The seven component files' pins, in `COMPONENTS` order.

    Raises:
        ValueError: If `sources` holds other than one source per component.
        UnavailableError: If a snapshot is absent.
        ContractError: If a file contradicts its pin or schema, or the cohort holds no record or
            no positive label.
    """
    if len(sources) != len(COMPONENTS):
        msg = f"NHANES reads {len(COMPONENTS)} component files, not {len(sources)}"
        raise ValueError(msg)
    read = {
        name: read_component(read_verified(source, data_dir), name)
        for name, source in zip(COMPONENTS, sources, strict=True)
    }
    records = frame_records(read)
    if records.empty:
        msg = f"{NHANES_NAME}: the cohort holds no records"
        raise ContractError(msg)
    if not records[LABEL].any():
        msg = f"{NHANES_NAME}: the cohort holds no adult with a component of the composite"
        raise ContractError(msg)
    return records
