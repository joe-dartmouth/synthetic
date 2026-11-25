"""SUPPORT2: one frame record per seriously ill hospitalized adult, labelled by in-hospital death.

`load_support2` reads `support2.csv` from the verified `support2` snapshot without extracting it.
The header holds 47 names and every data row 48 fields: the first, unnamed, is the row number,
which becomes `record_id`. Values carry leading spaces, a missing value is an empty field and the
categoricals are text. The cohort drops the records whose `scoma` is missing. A record holds
`age` floored to whole years, the coded demographics and comorbidities, `num.co`, `hday`,
`scoma`, each measured lab or vital as a coded state (`bko.datasets.states.coded`) and `LABEL`,
`hospdead`. The columns the publisher warns against using as predictors, those known only at or
after the outcome, `dzclass` (a grouping of `dzgroup`) and the partly imputed activity scores never
enter. ADR `research/support2-frame` records the frame.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.archive import member_text
from bko.datasets.cells import LABEL
from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.sources import Source
from bko.datasets.states import coded


logger = logging.getLogger(__name__)

SUPPORT2_NAME: Final = "support2"
"""The frame's name, its `Dataset` value and its catalog name."""

MEMBER: Final = "support2.csv"
"""The archive member holding the records."""

COLUMNS: Final = (
    "age",
    "death",
    "sex",
    "hospdead",
    "slos",
    "d.time",
    "dzgroup",
    "dzclass",
    "num.co",
    "edu",
    "income",
    "scoma",
    "charges",
    "totcst",
    "totmcst",
    "avtisst",
    "race",
    "sps",
    "aps",
    "surv2m",
    "surv6m",
    "hday",
    "diabetes",
    "dementia",
    "ca",
    "prg2m",
    "prg6m",
    "dnr",
    "dnrday",
    "meanbp",
    "wblc",
    "hrt",
    "resp",
    "temp",
    "pafi",
    "alb",
    "bili",
    "crea",
    "sod",
    "ph",
    "glucose",
    "bun",
    "urine",
    "adlp",
    "adls",
    "sfdm2",
    "adlsc",
)
"""The 47 names of `support2.csv`'s header, in file order; the header must be exactly these."""

FIELDS: Final = len(COLUMNS) + 1
"""The fields of every data row: the unnamed row number, then one per header name."""

WARNED: Final = ("aps", "sps", "surv2m", "surv6m", "prg2m", "prg6m", "dnr", "dnrday")
"""The columns the publisher's page warns against using as predictors."""

AFTER_OUTCOME: Final = (
    "death",
    "slos",
    "d.time",
    "charges",
    "totcst",
    "totmcst",
    "avtisst",
    "sfdm2",
)
"""The columns known only at or after the outcome."""

LEFT_OUT: Final = ("dzclass", "adlp", "adls", "adlsc")
"""`dzclass`, a grouping of `dzgroup`, and the activity scores, partly imputed by the publisher."""

SEX_CODES: Final[Mapping[str, int]] = {"female": 0, "male": 1}
"""Each `sex` value and its code (`bko.datasets.cells.SEXES`)."""

DZGROUP_CODES: Final[Mapping[str, int]] = {
    "ARF/MOSF w/Sepsis": 0,
    "COPD": 1,
    "CHF": 2,
    "Cirrhosis": 3,
    "Coma": 4,
    "Colon Cancer": 5,
    "Lung Cancer": 6,
    "MOSF w/Malig": 7,
}
"""Each `dzgroup` level, in the codebook's order, and its code."""

CA_CODES: Final[Mapping[str, int]] = {"no": 0, "yes": 1, "metastatic": 2}
"""Each `ca` level, in the codebook's order, and its code."""

RACE_CODES: Final[Mapping[str, int]] = {
    "": 0,
    "white": 1,
    "black": 2,
    "asian": 3,
    "other": 4,
    "hispanic": 5,
}
"""Each `race` level, in the codebook's order after code 0, not recorded (an empty field)."""

INCOME_CODES: Final[Mapping[str, int]] = {
    "": 0,
    "under $11k": 1,
    "$11-$25k": 2,
    "$25-$50k": 3,
    ">$50k": 4,
}
"""Each `income` level, in the codebook's order after code 0, not recorded (an empty field)."""

PRESENCE_CODES: Final[Mapping[str, int]] = {"0": 0, "1": 1}
"""`diabetes`, `dementia` and `hospdead`: 1 where present."""

CATEGORIES: Final[Mapping[str, Mapping[str, int]]] = MappingProxyType(
    {
        "sex": SEX_CODES,
        "hospdead": PRESENCE_CODES,
        "dzgroup": DZGROUP_CODES,
        "diabetes": PRESENCE_CODES,
        "dementia": PRESENCE_CODES,
        "ca": CA_CODES,
        "race": RACE_CODES,
        "income": INCOME_CODES,
    }
)
"""Every text column a record reads and its codes."""


@dataclass(frozen=True, slots=True)
class Cutoffs:
    """Where a measured field's states split (`bko.datasets.states.coded`).

    Attributes:
        at_or_above: Cutoffs a value at or above reaches.
        above: Cutoffs only a value strictly above reaches.
        source: Where the cutoffs come from: a `docs/literature.md` key, or `tertiles` for the
            published file's tertiles.
    """

    at_or_above: tuple[float, ...] = ()
    above: tuple[float, ...] = ()
    source: str = "tertiles"


CITED: Final[Mapping[str, Cutoffs]] = MappingProxyType(
    {
        "hrt": Cutoffs(above=(90.0,), source="singer-2016"),
        "resp": Cutoffs(above=(20.0,), source="singer-2016"),
        "temp": Cutoffs(at_or_above=(36.0,), above=(38.0,), source="singer-2016"),
        "wblc": Cutoffs(at_or_above=(4.0,), above=(12.0,), source="singer-2016"),
        "meanbp": Cutoffs(at_or_above=(70.0,), source="singer-2016"),
        "crea": Cutoffs(at_or_above=(1.2,), source="singer-2016"),
        "bili": Cutoffs(at_or_above=(1.2,), source="singer-2016"),
        "pafi": Cutoffs(at_or_above=(400.0,), source="singer-2016"),
    }
)
"""The labs and vitals coded at cited cutoffs: the SIRS criteria (`bone-1992`) for `hrt`, `resp`,
`temp` and `wblc` (in thousands per cubic millimetre), and SOFA's first abnormal score
(`vincent-1996`) for `meanbp`, `crea`, `bili` and `pafi`, each read in `singer-2016`'s
reproduction (Box 1 and Table 1)."""

TERTILES: Final[Mapping[str, Cutoffs]] = MappingProxyType(
    {
        "edu": Cutoffs(at_or_above=(11.0, 12.0)),
        "sod": Cutoffs(at_or_above=(135.0, 140.0)),
        "alb": Cutoffs(at_or_above=(2.5, 3.3)),
        "ph": Cutoffs(at_or_above=(7.4, 7.45)),
        "glucose": Cutoffs(at_or_above=(111.0, 166.33)),
        "bun": Cutoffs(at_or_above=(17.0, 33.0)),
        "urine": Cutoffs(at_or_above=(1430.0, 2595.0)),
    }
)
"""The numeric fields with missing values and no cited cutoff, coded at the tertiles of their
measured values, read at `DECIMALS`, over every record of the published file
(`bko.datasets.states.tertiles`), each rounded to `DECIMALS`."""

MEASURED: Final[Mapping[str, Cutoffs]] = MappingProxyType({**CITED, **TERTILES})
"""Every coded measured field, in record order, and its cutoffs."""

DECIMALS: Final = 2
"""The decimals a measured value is read at before it is coded: the file stores values truncated
in binary, so a creatinine of 1.2 reads `1.19995117`."""

WHOLE_FIELDS: Final = ("num.co", "hday", "scoma")
"""The numeric fields kept as whole numbers, beside `age`."""

FEATURES: Final = (
    "age",
    "sex",
    "dzgroup",
    "num.co",
    "diabetes",
    "dementia",
    "ca",
    "hday",
    "scoma",
    "race",
    "income",
    *MEASURED,
)
"""The frame's features, in record order."""

_WHOLE: Final = r"[0-9]{1,18}"
"""An ASCII whole number that fits int64."""

_DECIMAL: Final = r"[0-9]{1,9}(?:\.[0-9]{1,12})?"
"""An ASCII decimal number as the file writes a measured value."""


def parse_support2(text: str) -> pd.DataFrame:
    """Parse `support2.csv` into one row of stripped strings per record, after checking its header.

    Returns:
        The 47 `COLUMNS` as strings with their leading and trailing spaces dropped, in file order,
        indexed by `record_id`, the row number each row starts with.

    Raises:
        ContractError: If the header is not `COLUMNS`, a line has other than `FIELDS` fields, or a
            row number is not a whole number or repeats.
    """
    reader = csv.reader(io.StringIO(text, newline=""))
    header = tuple(next(reader, ()))
    if header != COLUMNS:
        msg = f"{MEMBER}: the header is not the archive's {len(COLUMNS)} names in order"
        raise ContractError(msg)
    read: list[tuple[int, list[str]]] = []
    try:
        read.extend((reader.line_num, row) for row in reader)
    except csv.Error as exc:
        msg = f"{MEMBER} line {reader.line_num}: {exc}"
        raise ContractError(msg) from exc
    ids: list[int] = []
    rows: list[list[str]] = []
    for line, row in read:
        if len(row) != FIELDS:
            msg = f"{MEMBER} line {line}: {len(row)} fields, expected {FIELDS}"
            raise ContractError(msg)
        number = row[0].strip()
        if not re.fullmatch(_WHOLE, number):
            msg = f"{MEMBER} line {line}: row number {row[0]!r} is not a whole number"
            raise ContractError(msg)
        ids.append(int(number))
        rows.append([value.strip() for value in row[1:]])
    index = pd.Index(ids, dtype="int64", name="record_id")
    if index.has_duplicates:
        msg = f"{MEMBER} record {index[index.duplicated()][0]}: listed more than once"
        raise ContractError(msg)
    return pd.DataFrame(rows, columns=list(COLUMNS), index=index, dtype=str)


def checked(raw: pd.DataFrame) -> pd.DataFrame:
    """Return `raw` in record order once every value it is read for fits the schema.

    Raises:
        ContractError: If a read value contradicts the codebook, naming the first record in
            record order.
    """
    rows = raw.sort_index(kind="stable")
    checks: list[tuple[str, pd.Series, str]] = [
        ("age", rows["age"].str.fullmatch(_DECIMAL), "not a number"),
        *(
            (name, rows[name].str.fullmatch(_WHOLE), "not a whole number")
            for name in ("num.co", "hday")
        ),
        ("scoma", rows["scoma"].str.fullmatch(f"(?:{_WHOLE})?"), "not a whole number or empty"),
        *(
            (name, rows[name].isin(list(codes)), f"not one of {sorted(codes)}")
            for name, codes in CATEGORIES.items()
        ),
        *(
            (name, rows[name].str.fullmatch(f"(?:{_DECIMAL})?"), "not a number or empty")
            for name in MEASURED
        ),
    ]
    for name, fits, why in checks:
        bad = ~fits.to_numpy(dtype=bool)
        if bad.any():
            first = int(bad.argmax())
            msg = (
                f"{MEMBER} record {rows.index[first]}: {name} is {rows[name].iloc[first]!r}, {why}"
            )
            raise ContractError(msg)
    return rows


def measured(values: pd.Series) -> pd.Series:
    """A measured column as float64 at `DECIMALS`, an empty field as NaN."""
    numbers = pd.to_numeric(values.mask(values.str.len() == 0), errors="raise")
    return numbers.astype("float64").round(DECIMALS)


def frame_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the frame records of `raw`'s cohort, in record order.

    Every row is checked before any is dropped. The cohort drops the records whose `scoma` is
    missing, and logs one `cohort` line that accounts for every row.

    Args:
        raw: The output of `parse_support2`.

    Returns:
        `FEATURES` and `LABEL`, all int64, indexed by `record_id`.

    Raises:
        ContractError: As `checked`.
    """
    rows = checked(raw)
    kept = rows.loc[rows["scoma"] != ""]
    columns: dict[str, pd.Series] = {
        "age": kept["age"].astype("float64").floordiv(1),
        **{name: kept[name].map(CATEGORIES[name]) for name in ("sex", "dzgroup")},
        "num.co": kept["num.co"],
        **{name: kept[name].map(CATEGORIES[name]) for name in ("diabetes", "dementia", "ca")},
        "hday": kept["hday"],
        "scoma": kept["scoma"],
        **{name: kept[name].map(CATEGORIES[name]) for name in ("race", "income")},
        **{
            name: coded(measured(kept[name]), at_or_above=cut.at_or_above, above=cut.above)
            for name, cut in MEASURED.items()
        },
        LABEL: kept["hospdead"].map(PRESENCE_CODES),
    }
    records = pd.DataFrame(columns).astype("int64")
    records.index.name = "record_id"
    logger.info(
        "cohort rows=%d missing_scoma=%d records=%d hospital_deaths=%d",
        len(rows),
        len(rows) - len(kept),
        len(records),
        int(records[LABEL].sum()),
    )
    return records


def load_support2(data_dir: Path, *, sources: Sequence[Source]) -> pd.DataFrame:
    """Load the frame records from the verified snapshot of the one source in `sources`.

    This never fetches; `bko.datasets.fetch.ensure` does. Nothing is extracted: the member is read
    from the archive's bytes once they match the pin.

    Raises:
        ValueError: If `sources` holds other than one source.
        UnavailableError: If the snapshot is absent.
        ContractError: If the archive or its member contradicts its pin or schema, the member
            holds more than `bko.datasets.archive.MEMBER_LIMIT` bytes, or the cohort holds no
            record or no in-hospital death.
    """
    if len(sources) != 1:
        msg = f"SUPPORT2 reads one archive, not {len(sources)}"
        raise ValueError(msg)
    [source] = sources
    data = read_verified(source, data_dir)
    records = frame_records(parse_support2(member_text(data, MEMBER, source)))
    if records.empty:
        msg = f"{source.name}: the cohort holds no records"
        raise ContractError(msg)
    if not records[LABEL].any():
        msg = f"{source.name}: the cohort holds no in-hospital death"
        raise ContractError(msg)
    return records
