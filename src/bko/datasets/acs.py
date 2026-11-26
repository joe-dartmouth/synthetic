"""ACS PUMS: one frame record per Massachusetts person of the 2024 file in Folktables' cohort.

`read_kept` reads `psam_p25.csv` from the verified `acs-ma-2024` snapshot without extracting it and
returns the rows Folktables' ACSIncome filters keep, as text: every column is read as a string, so
`OCCP` keeps its leading zeros, and no value is read as missing. The header must be the published
286 names and every row must hold 286 fields. `occupation_pairs` gives each kept `OCCP` code's
`SOCP` code and refuses a file where one `OCCP` code carries two. ADR `research/acs-frame` records
the frame.
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.archive import member_text
from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.sources import Source


logger = logging.getLogger(__name__)

ACS_NAME: Final = "acs"
"""The frame's name, its `Dataset` value and its catalog name."""

MEMBER: Final = "psam_p25.csv"
"""The archive member holding Massachusetts' person records."""

_HEADER: Final = (
    "RT,SERIALNO,DIVISION,SPORDER,PUMA,REGION,STATE,ADJINC,PWGTP,AGEP,CIT,CITWP,COW,DDRS,"
    "DEAR,DEYE,DOUT,DPHY,DRAT,DRATX,DREM,ENG,FER,GCL,GCM,GCR,HIMRKS,HINS1,HINS2,HINS3,HINS4,"
    "HINS5,HINS6,HINS7,INTP,JWMNP,JWRIP,JWTRNS,LANX,MAR,MARHD,MARHM,MARHT,MARHW,MARHYP,MIG,"
    "MIL,MLPA,MLPB,MLPCD,MLPE,MLPFG,MLPIK,MLPHJ,NWAB,NWAV,NWLA,NWLK,NWRE,OIP,PAP,RELSHIPP,"
    "RETP,SCH,SCHG,SCHL,SEMP,SEX,SSIP,SSP,WAGP,WKHP,WKL,WKWN,WRK,YOEP,ANC,ANC1P,ANC2P,"
    "DECADE,DIS,DRIVESP,ESP,ESR,FOD1P,FOD2P,HICOV,HISP,INDP,JWAP,JWDP,LANP,MIGPUMA,MIGSP,"
    "MSP,NAICSP,NATIVITY,NOP,OC,OCCP,PAOC,PERNP,PINCP,POBP,POVPIP,POWPUMA,POWSP,PRIVCOV,"
    "PUBCOV,QTRBIR,RAC1P,RAC2P,RAC3P,RACAIAN,RACASN,RACBLK,RACNH,RACNUM,RACPI,RACSOR,RACWHT,"
    "RC,SCIENGP,SCIENGRLP,SFN,SFR,SOCP,VPSP,WAOB,FAGEP,FANCP,FCITP,FCITWP,FCOWP,FDDRSP,"
    "FDEARP,FDEYEP,FDISP,FDOUTP,FDPHYP,FDRATP,FDRATXP,FDREMP,FENGP,FESRP,FFERP,FFODP,FGCLP,"
    "FGCMP,FGCRP,FHICOVP,FHIMRKSP,FHINS1P,FHINS2P,FHINS3C,FHINS3P,FHINS4C,FHINS4P,FHINS5C,"
    "FHINS5P,FHINS6P,FHINS7P,FHISP,FINDP,FINTP,FJWDP,FJWMNP,FJWRIP,FJWTRNSP,FLANP,FLANXP,"
    "FMARP,FMARHDP,FMARHMP,FMARHTP,FMARHWP,FMARHYP,FMIGP,FMIGSP,FMILPP,FMILSP,FOCCP,FOIP,"
    "FPAP,FPERNP,FPINCP,FPOBP,FPOWSP,FPRIVCOVP,FPUBCOVP,FRACP,FRELSHIPP,FRETP,FSCHGP,FSCHLP,"
    "FSCHP,FSEMP,FSEXP,FSSIP,FSSP,FWAGP,FWKHP,FWKLP,FWKWNP,FWRKP,FYOEP,PWGTP1,PWGTP2,PWGTP3,"
    "PWGTP4,PWGTP5,PWGTP6,PWGTP7,PWGTP8,PWGTP9,PWGTP10,PWGTP11,PWGTP12,PWGTP13,PWGTP14,"
    "PWGTP15,PWGTP16,PWGTP17,PWGTP18,PWGTP19,PWGTP20,PWGTP21,PWGTP22,PWGTP23,PWGTP24,"
    "PWGTP25,PWGTP26,PWGTP27,PWGTP28,PWGTP29,PWGTP30,PWGTP31,PWGTP32,PWGTP33,PWGTP34,"
    "PWGTP35,PWGTP36,PWGTP37,PWGTP38,PWGTP39,PWGTP40,PWGTP41,PWGTP42,PWGTP43,PWGTP44,"
    "PWGTP45,PWGTP46,PWGTP47,PWGTP48,PWGTP49,PWGTP50,PWGTP51,PWGTP52,PWGTP53,PWGTP54,"
    "PWGTP55,PWGTP56,PWGTP57,PWGTP58,PWGTP59,PWGTP60,PWGTP61,PWGTP62,PWGTP63,PWGTP64,"
    "PWGTP65,PWGTP66,PWGTP67,PWGTP68,PWGTP69,PWGTP70,PWGTP71,PWGTP72,PWGTP73,PWGTP74,"
    "PWGTP75,PWGTP76,PWGTP77,PWGTP78,PWGTP79,PWGTP80"
)

HEADER: Final = tuple(_HEADER.split(","))
"""The 286 names of `psam_p25.csv`'s header, in file order; the header must be exactly these."""

FEATURES_READ: Final = (
    "AGEP",
    "SEX",
    "COW",
    "SCHL",
    "MAR",
    "OCCP",
    "POBP",
    "WKHP",
    "RAC1P",
    "RELSHIPP",
    "DEAR",
    "DEYE",
    "DREM",
    "DPHY",
    "DDRS",
    "DOUT",
)
"""The columns that become features, in record order: Folktables' ACSIncome features with
`RELSHIPP` for `RELP`, then the six disability items."""

DISABILITY: Final = ("DEAR", "DEYE", "DREM", "DPHY", "DDRS", "DOUT")
"""The six disability items, each 1 yes and 2 no."""

READ: Final = (*FEATURES_READ, "PINCP", "PWGTP", "SOCP")
"""Every column the loader reads: the features, income, the person weight and the SOC code."""

THRESHOLD: Final = 50_000
"""The label is 1 when unadjusted `PINCP` is strictly above this."""

FILTERS: Final = (
    ("AGEP", 16, False),
    ("PINCP", 100, False),
    ("WKHP", 0, False),
    ("PWGTP", 1, True),
)
"""Folktables' ACSIncome filters in its order, each `(column, limit, inclusive)`: a value above
the limit, or at it when inclusive, is kept; a blank value is kept by none."""

_WHOLE: Final = r"[0-9]{1,7}"
_SIGNED: Final = r"-?[0-9]{1,7}"

FILTER_PATTERNS: Final[Mapping[str, str]] = MappingProxyType(
    {"AGEP": _WHOLE, "PINCP": f"(?:{_SIGNED})?", "WKHP": f"(?:{_WHOLE})?", "PWGTP": _WHOLE}
)
"""What every row's filter columns must look like: `PINCP` and `WKHP` may be blank."""

CODES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "SEX": frozenset({"1", "2"}),
        "COW": frozenset(str(code) for code in range(1, 10)),
        "SCHL": frozenset(f"{code:02d}" for code in range(1, 25)),
        "MAR": frozenset(str(code) for code in range(1, 6)),
        "RAC1P": frozenset(str(code) for code in range(1, 10)),
        "RELSHIPP": frozenset(str(code) for code in range(20, 39)),
        **dict.fromkeys(DISABILITY, frozenset({"1", "2"})),
    }
)
"""The 2024 data dictionary's codes of each coded column a kept row reads, its N/A blank left
out."""

KEPT_PATTERNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "OCCP": r"[0-9]{4}",
        "POBP": r"[0-9]{3}",
        "SOCP": r"[0-9]{2}[0-9]{0,4}[XY]{0,4}",
        "AGEP": r"[0-9]{1,2}",
        "WKHP": r"[0-9]{1,2}",
    }
)
"""What a kept row's other read columns must look like: `SOCP` six characters, digits then any
trailing `X` or `Y` (its length is checked apart)."""


def parse_acs(text: str) -> pd.DataFrame:
    """Parse `psam_p25.csv` into its `READ` columns as strings, after checking its shape.

    Returns:
        One row per data line, in file order, indexed by its line number, the header line 1.

    Raises:
        ContractError: If a line is not readable CSV, the header is not `HEADER`, or a line has
            other than 286 fields.
    """
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = tuple(next(reader, ()))
        read = [(reader.line_num, row) for row in reader]
    except csv.Error as exc:
        msg = f"{MEMBER} line {reader.line_num}: {exc}"
        raise ContractError(msg) from exc
    if header != HEADER:
        msg = f"{MEMBER}: the header is not the published {len(HEADER)} names in order"
        raise ContractError(msg)
    wanted = [HEADER.index(name) for name in READ]
    lines: list[int] = []
    rows: list[list[str]] = []
    for line, row in read:
        if len(row) != len(HEADER):
            msg = f"{MEMBER} line {line}: {len(row)} fields, expected {len(HEADER)}"
            raise ContractError(msg)
        lines.append(line)
        rows.append([row[i] for i in wanted])
    index = pd.Index(lines, dtype="int64", name="line")
    return pd.DataFrame(rows, columns=list(READ), index=index, dtype=str)


def _refuse(rows: pd.DataFrame, name: str, fits: pd.Series, why: str) -> None:
    bad = ~fits.to_numpy(dtype=bool)
    if bad.any():
        first = int(bad.argmax())
        value = rows[name].iloc[first]
        msg = f"{MEMBER} line {rows.index[first]}: {name} is {value!r}, {why}"
        raise ContractError(msg)


def kept_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Apply `FILTERS` in order to `raw`, the output of `parse_acs`, and check the kept rows.

    Returns:
        The kept rows as strings, indexed by `record_id`, their position among the kept rows in
        file order from 1; and the row count before the filters and after each.

    Raises:
        ContractError: If a row's filter column does not look like `FILTER_PATTERNS` says, or a
            kept row's read value is blank or not one of its codes, naming its line.
    """
    for name, pattern in FILTER_PATTERNS.items():
        _refuse(raw, name, raw[name].str.fullmatch(pattern), "not a whole number or blank")
    keep = pd.Series(True, index=raw.index)
    counts = [len(raw)]
    for name, limit, inclusive in FILTERS:
        values = pd.to_numeric(raw[name].mask(raw[name].str.len() == 0), errors="raise")
        keep &= values >= limit if inclusive else values > limit
        counts.append(int(keep.sum()))
    rows = raw.loc[keep.to_numpy(dtype=bool)]
    for name, codes in CODES.items():
        _refuse(rows, name, rows[name].isin(list(codes)), "not one of its codes")
    for name, pattern in KEPT_PATTERNS.items():
        _refuse(rows, name, rows[name].str.fullmatch(pattern), f"not {pattern}")
    _refuse(rows, "SOCP", rows["SOCP"].str.len() == 6, "not six characters")
    records = rows.reset_index(drop=True)
    records.index = pd.RangeIndex(1, len(records) + 1, name="record_id")
    return records, tuple(counts)


def occupation_pairs(rows: pd.DataFrame) -> dict[str, str]:
    """Each `OCCP` code among `rows` and the one `SOCP` code its rows carry, in code order.

    Raises:
        ContractError: If an `OCCP` code's rows carry more than one `SOCP` code, naming the first
            such code and its `SOCP` codes.
    """
    held = rows.groupby("OCCP", sort=True)["SOCP"].unique()
    for code, socs in held.items():
        if len(socs) > 1:
            msg = f"{MEMBER}: OCCP {code} carries SOCP {', '.join(sorted(socs))}, not one code"
            raise ContractError(msg)
    return {str(code): str(socs[0]) for code, socs in held.items()}


def read_kept(data_dir: Path, source: Source) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """`kept_rows` of the verified snapshot of `source`; never fetches or extracts.

    Raises:
        UnavailableError: If the snapshot is absent.
        ContractError: If the archive or its member contradicts its pin or schema, or the member
            holds more than `bko.datasets.archive.MEMBER_LIMIT` bytes.
    """
    return kept_rows(parse_acs(member_text(read_verified(source, data_dir), MEMBER, source)))
