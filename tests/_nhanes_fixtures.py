"""Offline NHANES inputs: SAS XPORT component files built in memory by a hand-built writer.

pandas reads XPORT but writes none, so `xpt_bytes` writes version 5 files: the library, member,
descriptor and NAMESTR headers, one 140-byte NAMESTR record per column and the observations,
numbers as IBM double-precision floats. The real data here is `REAL_COLUMNS` and `REAL_ROWS`, each
component file's column names and first record as `pandas.read_sas` reads them, kept so the reader
is proven against the published layout. Every other respondent is built from them with invented
values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from bko.datasets.nhanes import COMPONENTS


REAL_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
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
"""Each component file's column names, in file order, as the published file holds them."""

_ = None
REAL_ROWS: Final[Mapping[str, tuple[float | bytes | None, ...]]] = {
    "DEMO_L": (
        *(130378.0, 12.0, 2.0, 1.0, 43.0, _, 5.0, 6.0, 2.0, _, 2.0, 2.0, 6.0, 5.0, 1.0, _, 4.0),
        *(_, _, _, _, _, 50055.450807, 54374.463898, 173.0, 2.0, 5.0),
    ),
    "MCQ_L": (
        *(130378.0, 2.0, _, _, _, 2.0, 2.0, _, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, _, 2.0),
        *(2.0, _, _, _, _, _, _, _, _, 2.0, 2.0, 2.0, _, _, _, _, 2.0),
    ),
    "BPXO_L": (130378.0, b"R", 4.0, 135.0, 98.0, 131.0, 96.0, 132.0, 94.0, 82.0, 79.0, 82.0),
    "BMX_L": (
        *(130378.0, 1.0, 86.9, _, _, _, _, _, 179.5, _, 27.0, _, 42.8, _, 42.0, _, 35.7, _),
        *(98.3, _, 102.9, _),
    ),
    "TCHOL_L": (130378.0, 56042.129409583235, 264.0, 6.83),
    "HDL_L": (130378.0, 56042.129409583235, 45.0, 1.16),
    "GHB_L": (130378.0, 56042.129409583235, 5.6),
}
"""Each component file's first record, respondent 130378, as `pandas.read_sas` reads it; None is a
missing value."""

QUESTIONS: Final = ("MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F")
"""The four medical-conditions items the label reads."""


def _ibm(value: float) -> bytes:
    """`value` as an IBM hexadecimal double: sign, excess-64 base-16 exponent, 56-bit fraction."""
    if value == 0:
        return bytes(8)
    sign = 0x80 if value < 0 else 0
    fraction, exponent = abs(value), 0
    while fraction >= 1:
        fraction, exponent = fraction / 16, exponent + 1
    while fraction < 1 / 16:
        fraction, exponent = fraction * 16, exponent - 1
    mantissa = round(fraction * 2**56)
    if mantissa >= 2**56:
        mantissa, exponent = mantissa >> 4, exponent + 1
    return bytes([sign | (exponent + 64)]) + mantissa.to_bytes(7, "big")


def _card(text: str) -> bytes:
    """One 80-byte header record: `text` padded with blanks."""
    assert len(text) <= 80, text
    return text.ljust(80).encode("ascii")


def _namestr(number: int, name: str, *, character: bool, length: int, position: int) -> bytes:
    """One 140-byte NAMESTR record for the column `name` at byte `position` of a row."""
    record = (
        (2 if character else 1).to_bytes(2, "big")
        + bytes(2)
        + length.to_bytes(2, "big")
        + number.to_bytes(2, "big")
        + name.ljust(8).encode("ascii")
        + name.ljust(40).encode("ascii")
        + bytes(8 + 8)
        + b" " * 8
        + bytes(4)
        + position.to_bytes(4, "big")
        + bytes(52)
    )
    assert len(record) == 140
    return record


def _padded(data: bytes) -> bytes:
    return data + b" " * (-len(data) % 80)


def xpt_bytes(frame: pd.DataFrame, member: str = "DATA") -> bytes:
    """`frame` as a SAS XPORT version 5 file with one member.

    A column of bytes is a character column as wide as its widest value; every other column is
    numeric, a missing value (None or NaN) written as SAS's `.`.
    """
    stamp = "11JAN24:06:56:54"
    names = [str(name) for name in frame.columns]
    character = {
        name: bool(frame[name].map(lambda v: isinstance(v, bytes)).any()) for name in names
    }
    widths: dict[str, int] = {
        name: max(1, *(len(v) for v in frame[name] if isinstance(v, bytes)))
        if character[name]
        else 8
        for name in character
    }
    starts: dict[str, int] = {}
    position = 0
    for name, width in widths.items():
        starts[name] = position
        position += width
    namestrs = b"".join(
        _namestr(n + 1, name, character=character[name], length=widths[name], position=starts[name])
        for n, name in enumerate(names)
    )
    rows = b"".join(
        b"".join(
            (value or b"").ljust(widths[name])
            if character[name]
            else (b"." + bytes(7) if value is None or math.isnan(value) else _ibm(float(value)))
            for name, value in zip(names, record, strict=True)
        )
        for record in frame.itertuples(index=False, name=None)
    )
    header = (
        _card("HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!" + "0" * 30)
        + _card("SAS     SAS     SASLIB  9.4     X64_10PR" + " " * 24 + stamp)
        + _card(stamp)
        + _card(
            "HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!" + "0" * 17 + "16" + "0" * 8 + "140"
        )
        + _card("HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!" + "0" * 30)
        + _card(f"SAS     {member:<8}SASDATA 9.4     X64_10PR" + " " * 24 + stamp)
        + _card(stamp)
        + _card(
            f"HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!000000{len(frame.columns):04d}"
            + "0" * 20
        )
    )
    observations = _card("HEADER RECORD*******OBS     HEADER RECORD!!!!!!!" + "0" * 30)
    return header + _padded(namestrs) + observations + _padded(rows)


@dataclass(frozen=True, slots=True)
class Respondent:
    """One invented respondent: what each component file holds for them.

    `answers` are the four items' codes in `QUESTIONS` order, None for a missing answer.
    `systolic` and `diastolic` are the three readings each. `examined` False leaves the respondent
    out of every exam and laboratory file; `None` for a single lab leaves them out of that file.
    """

    seqn: int
    age: float = 43.0
    sex: float = 1.0
    answers: tuple[float | None, ...] = (2.0, 2.0, 2.0, 2.0)
    systolic: tuple[float | None, ...] = (135.0, 131.0, 132.0)
    diastolic: tuple[float | None, ...] = (98.0, 96.0, 94.0)
    bmi: float | None = 27.0
    cholesterol: float | None = 264.0
    hdl: float | None = 45.0
    hba1c: float | None = 5.6
    examined: bool = True
    interviewed: bool = True
    extra: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)


def _row(component: str, seqn: int, values: Mapping[str, float | bytes | None]) -> dict:
    row = dict(zip(REAL_COLUMNS[component], REAL_ROWS[component], strict=True))
    row["SEQN"] = float(seqn)
    for name, value in values.items():
        assert name in row, name
        row[name] = value
    return row


def components(people: Sequence[Respondent]) -> dict[str, pd.DataFrame]:
    """Each component file's records for `people`, in `COMPONENTS` order, columns the real ones."""
    rows: dict[str, list[dict]] = {name: [] for name in COMPONENTS}
    for person in people:
        own = {
            "DEMO_L": {"RIDAGEYR": person.age, "RIAGENDR": person.sex},
            "MCQ_L": dict(zip(QUESTIONS, person.answers, strict=True)),
            "BPXO_L": {
                **dict(zip(("BPXOSY1", "BPXOSY2", "BPXOSY3"), person.systolic, strict=True)),
                **dict(zip(("BPXODI1", "BPXODI2", "BPXODI3"), person.diastolic, strict=True)),
            },
            "BMX_L": {"BMXBMI": person.bmi},
            "TCHOL_L": {"LBXTC": person.cholesterol},
            "HDL_L": {"LBDHDD": person.hdl},
            "GHB_L": {"LBXGH": person.hba1c},
        }
        for name in COMPONENTS:
            if name == "MCQ_L" and not person.interviewed:
                continue
            if name not in {"DEMO_L", "MCQ_L"} and not person.examined:
                continue
            values = {**own[name], **person.extra.get(name, {})}
            rows[name].append(_row(name, person.seqn, values))
    return {name: pd.DataFrame(rows[name], columns=list(REAL_COLUMNS[name])) for name in COMPONENTS}


def files(people: Sequence[Respondent]) -> dict[str, bytes]:
    """Each component file of `people` as XPORT bytes, in `COMPONENTS` order."""
    return {name: xpt_bytes(frame, name) for name, frame in components(people).items()}


def mixed() -> list[Respondent]:
    """Twelve respondents across the age bands, both sexes, both labels and missing values.

    Three are left out: one aged 19, one with no yes and a don't-know, one not asked the items.
    The items beside the four, such as `MCQ160A` and the angina item `MCQ160D`, vary too, so a
    label that read one of them would move.
    """
    return [
        Respondent(130378),
        Respondent(130379, age=19.0, answers=(1.0, 2.0, 2.0, 2.0)),
        Respondent(130380, age=20.0, sex=2.0, answers=(2.0, 1.0, 2.0, 2.0)),
        Respondent(
            130381, age=35.0, sex=2.0, hba1c=6.5, hdl=39.0, extra={"MCQ_L": {"MCQ160D": 1.0}}
        ),
        Respondent(130382, age=45.0, answers=(2.0, 2.0, 9.0, 2.0)),
        Respondent(130383, age=52.0, sex=2.0, answers=(9.0, 2.0, 1.0, 2.0), cholesterol=None),
        Respondent(130384, age=61.0, answers=(2.0, 2.0, 2.0, 1.0), examined=False),
        Respondent(
            130385,
            age=70.0,
            sex=2.0,
            systolic=(128.0, 130.0, None),
            bmi=None,
            extra={"MCQ_L": {"MCQ160A": 2.0, "MCQ160D": 9.0}},
        ),
        Respondent(130386, age=80.0, answers=(1.0, 1.0, 1.0, 1.0), hba1c=5.7),
        Respondent(130387, age=66.0, sex=2.0, answers=(None, None, None, None)),
        Respondent(130388, age=58.0, systolic=(None, None, None), diastolic=(80.0, 79.0, 81.0)),
        Respondent(
            130389,
            age=29.0,
            sex=2.0,
            hdl=40.0,
            cholesterol=240.0,
            extra={"MCQ_L": {"MCQ160A": 9.0, "MCQ160D": 7.0}},
        ),
    ]
