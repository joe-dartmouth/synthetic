"""Offline SUPPORT2 inputs: `support2.csv` text and its archive, built in memory.

The real data here is `REAL_HEADER` and `REAL_ROW`, the file's header and first data row byte for
byte, kept so the parser is proven against the real spacing and number formats. Every other row is
built from `REAL_ROW` with invented values.
"""

from __future__ import annotations

from typing import Final

from bko.datasets.support2 import COLUMNS, MEMBER
from tests._datasets_fixtures import archive


REAL_HEADER: Final = (
    "age,death,sex,hospdead,slos,d.time,dzgroup,dzclass,num.co,edu,income,scoma,charges,totcst,"
    "totmcst,avtisst,race,sps,aps,surv2m,surv6m,hday,diabetes,dementia,ca,prg2m,prg6m,dnr,dnrday,"
    "meanbp,wblc,hrt,resp,temp,pafi,alb,bili,crea,sod,ph,glucose,bun,urine,adlp,adls,sfdm2,adlsc"
)
"""The real `support2.csv` header: 47 names, none for the leading row number."""

REAL_ROW: Final = (
    "1, 62.84998,0,male,0,  5,2029,Lung Cancer,Cancer,0,11,$11-$25k,  0,   9715.000,,, 7.000000,"
    "other,33.8984375, 20,0.2629394530,0.0369949341,  1,0,0,metastatic,0.500000000,2.500000e-01,"
    "no dnr,  5, 97.0,  6.00000000, 69.000000,22,36.00000,388.00000, 1.7998047, 0.19998169, "
    "1.19995117,141,7.459961,,,, 7, 7,,7.0000000"
)
"""The real file's first data row: the row number, then 47 values."""


def row(number: int, **values: str) -> str:
    """`REAL_ROW` with its row number set to `number` and the named columns replaced.

    A column name with a dot is passed with `_` in its place (`num_co` for `num.co`).
    """
    fields: dict[str, str] = dict(zip(COLUMNS, REAL_ROW.split(",")[1:], strict=True))
    for name, value in values.items():
        column = name.replace("_", ".")
        assert column in fields, column
        fields[column] = value
    return ",".join([str(number), *(fields[name] for name in COLUMNS)])


def text(*rows: str, header: str = REAL_HEADER) -> str:
    """The file's text: `header`, then `rows`, each line ended by a newline."""
    return "".join(f"{line}\n" for line in (header, *rows))


def mixed_rows() -> list[str]:
    """Twelve rows across age bands, sexes, labels, groups, missing values and cutoff edges."""
    groups = ("COPD", "CHF", "Cirrhosis", "Coma", "Colon Cancer", "Lung Cancer")
    return [
        row(
            n + 1,
            age=f" {40 + 5 * n}.{n}1",
            sex="female" if n % 2 else "male",
            hospdead=str(int(n % 3 == 0)),
            dzgroup=groups[n % len(groups)],
            hrt=" 91.0" if n % 2 else " 90.0",
            crea="" if n == 4 else " 1.19995117",
            race="" if n == 5 else "white",
            income="" if n == 6 else "under $11k",
            edu="" if n == 7 else str(8 + n),
        )
        for n in range(12)
    ]


def support2_archive(*rows: str) -> bytes:
    """A zip holding `text(*rows)`, or of `mixed_rows()` when none is given, as `MEMBER`."""
    return archive(text(*(rows or mixed_rows())), member=MEMBER)
