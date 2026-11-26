"""Offline ACS inputs: `psam_p25.csv` text and its archive, built in memory.

The real data here is `REAL_HEADER` and `REAL_ROW`, the file's header and its fifth data row byte
for byte, a kept row whose `OCCP` has a leading zero, kept so the reader is proven against the
real layout. Every other row is built from `REAL_ROW` with invented values.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.acs import MEMBER
from tests._datasets_fixtures import archive


REAL_HEADER: Final = (
    "RT,SERIALNO,DIVISION,SPORDER,PUMA,REGION,STATE,ADJINC,PWGTP,AGEP,CIT,CITWP,COW,DDRS,DEAR"
    ",DEYE,DOUT,DPHY,DRAT,DRATX,DREM,ENG,FER,GCL,GCM,GCR,HIMRKS,HINS1,HINS2,HINS3,HINS4,HINS5"
    ",HINS6,HINS7,INTP,JWMNP,JWRIP,JWTRNS,LANX,MAR,MARHD,MARHM,MARHT,MARHW,MARHYP,MIG,MIL"
    ",MLPA,MLPB,MLPCD,MLPE,MLPFG,MLPIK,MLPHJ,NWAB,NWAV,NWLA,NWLK,NWRE,OIP,PAP,RELSHIPP,RETP"
    ",SCH,SCHG,SCHL,SEMP,SEX,SSIP,SSP,WAGP,WKHP,WKL,WKWN,WRK,YOEP,ANC,ANC1P,ANC2P,DECADE,DIS"
    ",DRIVESP,ESP,ESR,FOD1P,FOD2P,HICOV,HISP,INDP,JWAP,JWDP,LANP,MIGPUMA,MIGSP,MSP,NAICSP"
    ",NATIVITY,NOP,OC,OCCP,PAOC,PERNP,PINCP,POBP,POVPIP,POWPUMA,POWSP,PRIVCOV,PUBCOV,QTRBIR"
    ",RAC1P,RAC2P,RAC3P,RACAIAN,RACASN,RACBLK,RACNH,RACNUM,RACPI,RACSOR,RACWHT,RC,SCIENGP"
    ",SCIENGRLP,SFN,SFR,SOCP,VPSP,WAOB,FAGEP,FANCP,FCITP,FCITWP,FCOWP,FDDRSP,FDEARP,FDEYEP"
    ",FDISP,FDOUTP,FDPHYP,FDRATP,FDRATXP,FDREMP,FENGP,FESRP,FFERP,FFODP,FGCLP,FGCMP,FGCRP"
    ",FHICOVP,FHIMRKSP,FHINS1P,FHINS2P,FHINS3C,FHINS3P,FHINS4C,FHINS4P,FHINS5C,FHINS5P"
    ",FHINS6P,FHINS7P,FHISP,FINDP,FINTP,FJWDP,FJWMNP,FJWRIP,FJWTRNSP,FLANP,FLANXP,FMARP"
    ",FMARHDP,FMARHMP,FMARHTP,FMARHWP,FMARHYP,FMIGP,FMIGSP,FMILPP,FMILSP,FOCCP,FOIP,FPAP"
    ",FPERNP,FPINCP,FPOBP,FPOWSP,FPRIVCOVP,FPUBCOVP,FRACP,FRELSHIPP,FRETP,FSCHGP,FSCHLP,FSCHP"
    ",FSEMP,FSEXP,FSSIP,FSSP,FWAGP,FWKHP,FWKLP,FWKWNP,FWRKP,FYOEP,PWGTP1,PWGTP2,PWGTP3,PWGTP4"
    ",PWGTP5,PWGTP6,PWGTP7,PWGTP8,PWGTP9,PWGTP10,PWGTP11,PWGTP12,PWGTP13,PWGTP14,PWGTP15"
    ",PWGTP16,PWGTP17,PWGTP18,PWGTP19,PWGTP20,PWGTP21,PWGTP22,PWGTP23,PWGTP24,PWGTP25,PWGTP26"
    ",PWGTP27,PWGTP28,PWGTP29,PWGTP30,PWGTP31,PWGTP32,PWGTP33,PWGTP34,PWGTP35,PWGTP36,PWGTP37"
    ",PWGTP38,PWGTP39,PWGTP40,PWGTP41,PWGTP42,PWGTP43,PWGTP44,PWGTP45,PWGTP46,PWGTP47,PWGTP48"
    ",PWGTP49,PWGTP50,PWGTP51,PWGTP52,PWGTP53,PWGTP54,PWGTP55,PWGTP56,PWGTP57,PWGTP58,PWGTP59"
    ",PWGTP60,PWGTP61,PWGTP62,PWGTP63,PWGTP64,PWGTP65,PWGTP66,PWGTP67,PWGTP68,PWGTP69,PWGTP70"
    ",PWGTP71,PWGTP72,PWGTP73,PWGTP74,PWGTP75,PWGTP76,PWGTP77,PWGTP78,PWGTP79,PWGTP80"
)
"""The real `psam_p25.csv` header: 286 names."""

REAL_ROW: Final = (
    "P,2024GQ0000120,1,01,00802,1,25,1015250,38,20,1,,1,2,2,2,2,2,,,2,,,,,,0,1,2,2,2,2,2,2,0,"
    ",,,2,5,,,,,,1,4,,,,,,,,3,5,3,3,3,0,0,38,0,3,15,19,0,1,0,0,3600,40,1,7,,,4,999,999,,2,,,3"
    ",,,1,01,8370,,,,,,6,6241,1,,,0420,,3600,3600,025,,,,1,2,4,1,1073,001,0,0,0,0,1,0,0,1,,,,"
    ",,119151,,1,0,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,1,0,1,1,,1,,1,,1,1,1,0,1,1,0,0,0,0"
    ",0,1,1,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,1,1,1,0,0,5,41,7,34,33"
    ",35,72,35,72,35,41,78,43,7,69,66,6,45,38,6,6,37,7,41,37,39,67,40,64,42,33,70,37,6,79,73"
    ",5,36,45,84,70,34,68,44,44,40,7,41,6,42,38,6,35,65,6,7,70,35,37,72,77,41,61,35,38,35,6"
    ",35,6,35,45,5,42,69,5,6,78,40,33,5"
)
"""The real file's fifth data row: a 20-year-old man kept by every filter, `OCCP` `0420`, `SOCP`
`119151`, `PINCP` 3600."""

_NAMES: Final = tuple(REAL_HEADER.split(","))


def row(**values: str) -> str:
    """`REAL_ROW` with the named columns replaced."""
    fields: dict[str, str] = dict(zip(_NAMES, REAL_ROW.split(","), strict=True))
    for name, value in values.items():
        assert name in fields, name
        fields[name] = value
    return ",".join(fields[name] for name in _NAMES)


def text(*rows: str, header: str = REAL_HEADER) -> str:
    """The file's text: `header`, then `rows`, each line ended by a newline."""
    return "".join(f"{line}\n" for line in (header, *rows))


def mixed_rows() -> list[str]:
    """Twelve kept rows across age bands, sexes, labels, occupations and disability items."""
    occupations = (("0420", "119151"), ("0010", "1110XX"), ("1530", "1721YY"), ("6240", "472040"))
    return [
        row(
            AGEP=str(25 + 4 * n),
            SEX="2" if n % 2 else "1",
            PINCP=str(50_000 + (n % 3) * 1000 - 1000),
            OCCP=occupations[n % len(occupations)][0],
            SOCP=occupations[n % len(occupations)][1],
            DEAR="1" if n == 3 else "2",
            DPHY="1" if n == 5 else "2",
            DDRS="1" if n == 7 else "2",
        )
        for n in range(12)
    ]


def acs_archive(*rows: str) -> bytes:
    """A zip holding `text(*rows)`, or of `mixed_rows()` when none is given, as `MEMBER`."""
    return archive(text(*(rows or mixed_rows())), member=MEMBER)
