"""ACS PUMS: one frame record per Massachusetts person of the 2024 file in Folktables' cohort.

`load_acs` reads the frame records through `read_kept`, which reads `psam_p25.csv` from the verified
`acs-ma-2024` snapshot without extracting it and returns the rows Folktables' ACSIncome filters
keep, as text: every column is read as a string, so `OCCP` keeps its leading zeros, and no value is
read as missing. The header must be the published 286 names and every row must hold 286 fields.
`occupation_pairs` gives each kept `OCCP` code's `SOCP` code and refuses a file where one `OCCP`
code carries two. A record holds `age` (`AGEP`), `sex` (`SEX` 1 to 1, 2 to 0), the other features as
the file codes them, and `LABEL`, 1 when the unadjusted `PINCP` is above `THRESHOLD`. ADR
`research/acs-frame` records the frame.
"""

from __future__ import annotations

import _csv
import csv
import io
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.archive import member_text
from bko.datasets.cells import LABEL
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

FEATURES: Final = ("age", "sex", *FEATURES_READ[2:])
"""The frame's features, in record order: `AGEP` as `age`, `SEX` as `sex`, then the rest."""

SEX_CODES: Final[Mapping[str, int]] = MappingProxyType({"2": 0, "1": 1})
"""Each `SEX` code and its frame code (`bko.datasets.cells.SEXES`)."""

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

_COW: Final[Mapping[int, str]] = MappingProxyType(
    {
        1: (
            "Employee of a private for-profit company or business, or of an individual, for "
            "wages, salary, or commissions"
        ),
        2: "Employee of a private not-for-profit, tax-exempt, or charitable organization",
        3: "Local government employee (city, county, etc.)",
        4: "State government employee",
        5: "Federal government employee",
        6: "Self-employed in own not incorporated business, professional practice, or farm",
        7: "Self-employed in own incorporated business, professional practice or farm",
        8: "Working without pay in family business or farm",
        9: "Unemployed and last worked 5 years ago or earlier or never worked",
    }
)

_SCHL: Final[Mapping[int, str]] = MappingProxyType(
    {
        1: "No schooling completed",
        2: "Nursery school, preschool",
        3: "Kindergarten",
        4: "Grade 1",
        5: "Grade 2",
        6: "Grade 3",
        7: "Grade 4",
        8: "Grade 5",
        9: "Grade 6",
        10: "Grade 7",
        11: "Grade 8",
        12: "Grade 9",
        13: "Grade 10",
        14: "Grade 11",
        15: "12th grade - no diploma",
        16: "Regular high school diploma",
        17: "GED or alternative credential",
        18: "Some college, but less than 1 year",
        19: "1 or more years of college credit, no degree",
        20: "Associate's degree",
        21: "Bachelor's degree",
        22: "Master's degree",
        23: "Professional degree beyond a bachelor's degree",
        24: "Doctorate degree",
    }
)

_MAR: Final[Mapping[int, str]] = MappingProxyType(
    {
        1: "Married",
        2: "Widowed",
        3: "Divorced",
        4: "Separated",
        5: "Never married or under 15 years old",
    }
)

_RAC1P: Final[Mapping[int, str]] = MappingProxyType(
    {
        1: "White alone",
        2: "Black or African American alone",
        3: "American Indian alone",
        4: "Alaska Native alone",
        5: (
            "American Indian and Alaska Native tribes specified; or American Indian or "
            "Alaska Native, not specified and no other races"
        ),
        6: "Asian alone",
        7: "Native Hawaiian and Other Pacific Islander alone",
        8: "Some Other Race alone",
        9: "Two or More Races",
    }
)

_RELSHIPP: Final[Mapping[int, str]] = MappingProxyType(
    {
        20: "Reference person",
        21: "Opposite-sex husband/wife/spouse",
        22: "Opposite-sex unmarried partner",
        23: "Same-sex husband/wife/spouse",
        24: "Same-sex unmarried partner",
        25: "Biological son or daughter",
        26: "Adopted son or daughter",
        27: "Stepson or stepdaughter",
        28: "Brother or sister",
        29: "Father or mother",
        30: "Grandchild",
        31: "Parent-in-law",
        32: "Son-in-law or daughter-in-law",
        33: "Other relative",
        34: "Roommate or housemate",
        35: "Foster child",
        36: "Other nonrelative",
        37: "Institutionalized group quarters population",
        38: "Noninstitutionalized group quarters population",
    }
)

DISABILITY_LABELS: Final[Mapping[int, str]] = MappingProxyType({1: "Yes", 2: "No"})
"""Each disability item's codes and meanings."""

CODEBOOK: Final[Mapping[str, Mapping[int, str]]] = MappingProxyType(
    {
        "COW": _COW,
        "SCHL": _SCHL,
        "MAR": _MAR,
        "RAC1P": _RAC1P,
        "RELSHIPP": _RELSHIPP,
        **dict.fromkeys(DISABILITY, DISABILITY_LABELS),
    }
)
"""The 2024 data dictionary's codes and meanings of each coded feature it fixes, its N/A blank
left out; `sex`, `OCCP` and `POBP` are coded apart."""

_PADDED: Final[Mapping[str, int]] = MappingProxyType({"SCHL": 2})
"""The columns whose codes the file writes zero-padded, and their width."""

CODES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "SEX": frozenset(SEX_CODES),
        **{
            name: frozenset(f"{code:0{_PADDED.get(name, 1)}d}" for code in codes)
            for name, codes in CODEBOOK.items()
        },
    }
)
"""Each coded column a kept row reads and the codes the file may write in it."""

KEPT_PATTERNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "OCCP": r"[0-9]{4}",
        "POBP": r"[0-9]{3}",
        "SOCP": r"[0-9]{2}[0-9]{0,4}[XY]{0,3}",
        "AGEP": r"[0-9]{1,2}",
        "WKHP": r"[0-9]{1,2}",
    }
)
"""What a kept row's other read columns must look like: `SOCP` six characters, digits then at
most three trailing `X` or `Y`, the codes `bko.ontology.occupations.soc_ids` reads (its length is
checked apart)."""


def parse_acs(text: str) -> pd.DataFrame:
    """Parse `psam_p25.csv` into its `READ` columns as strings, after checking its shape.

    Returns:
        One row per data line, in file order, indexed by its line number, the header line 1.

    Raises:
        ContractError: If a line is not readable CSV, the header is not `HEADER`, or a line has
            other than 286 fields.
    """
    reader = csv.reader(io.StringIO(text, newline=""))
    read = _readable(reader)
    if tuple(next(read, ())) != HEADER:
        msg = f"{MEMBER}: the header is not the published {len(HEADER)} names in order"
        raise ContractError(msg)
    wanted = [HEADER.index(name) for name in READ]
    lines: list[int] = []
    rows: list[list[str]] = []
    for row in read:
        if len(row) != len(HEADER):
            msg = f"{MEMBER} line {reader.line_num}: {len(row)} fields, expected {len(HEADER)}"
            raise ContractError(msg)
        lines.append(reader.line_num)
        rows.append([row[i] for i in wanted])
    index = pd.Index(lines, dtype="int64", name="line")
    return pd.DataFrame(rows, columns=list(READ), index=index, dtype=str)


def _readable(reader: _csv.Reader) -> Iterator[list[str]]:
    """`reader`'s rows, a line that is not readable CSV raising `ContractError` naming it."""
    try:
        yield from reader
    except csv.Error as exc:
        msg = f"{MEMBER} line {reader.line_num}: {exc}"
        raise ContractError(msg) from exc


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


def frame_records(rows: pd.DataFrame, counts: Sequence[int]) -> pd.DataFrame:
    """Return the frame records of `rows`, the kept rows `kept_rows` returns with `counts`.

    Logs one `cohort` line with the row count before the filters and after each, the records and
    those above `THRESHOLD`.

    Returns:
        `FEATURES` and `LABEL`, all int64, indexed by `record_id`.
    """
    columns: dict[str, pd.Series] = {
        "age": rows["AGEP"],
        "sex": rows["SEX"].map(SEX_CODES),
        **{name: rows[name] for name in FEATURES_READ[2:]},
        LABEL: pd.to_numeric(rows["PINCP"]) > THRESHOLD,
    }
    records = pd.DataFrame(columns).astype("int64")
    records.index.name = "record_id"
    logger.info(
        "cohort dataset=%s rows=%d age_over_16=%d income_over_100=%d hours_over_0=%d "
        "weight_at_least_1=%d records=%d income_over_50000=%d",
        ACS_NAME,
        *counts,
        len(records),
        int(records[LABEL].sum()),
    )
    return records


def load_acs(data_dir: Path, *, sources: Sequence[Source]) -> pd.DataFrame:
    """Load the frame records from the verified snapshot of the one source in `sources`.

    This never fetches; `bko.datasets.fetch.ensure` does. Nothing is extracted: the member is read
    from the archive's bytes once they match the pin.

    Raises:
        ValueError: If `sources` holds other than one source.
        UnavailableError: If the snapshot is absent.
        ContractError: As `read_kept` and `occupation_pairs`, or if the cohort holds no record or
            no income above `THRESHOLD`.
    """
    if len(sources) != 1:
        msg = f"ACS reads one archive, not {len(sources)}"
        raise ValueError(msg)
    [source] = sources
    rows, counts = read_kept(data_dir, source)
    # Called for its refusal alone: a file where one OCCP code carries two SOCP codes is refused.
    occupation_pairs(rows)
    records = frame_records(rows, counts)
    if records.empty:
        msg = f"{source.name}: the cohort holds no records"
        raise ContractError(msg)
    if not records[LABEL].any():
        msg = f"{source.name}: the cohort holds no income above {THRESHOLD}"
        raise ContractError(msg)
    return records
