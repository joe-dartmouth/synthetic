"""NHAMCS 2022 ED: one frame record per sampled emergency-department visit, labelled by admission.

`load_nhamcs` reads `ed2022` from the verified `nhamcs-ed-2022` snapshot without extracting it. The
member is fixed-width ASCII, one visit per line of `WIDTH` characters, each line ended by CRLF;
`POSITIONS` holds the columns of every item read, from the Stata dictionary `eddict2022.dct`.
`record_id` is `HOSPCODE` x 1000 + `PATCODE`, the hospital and the visit within it. Every visit
enters. A record holds `age` as published (top-coded at 94), `sex`, the five diagnosis tokens as
numbers (`bko.datasets.icd10_tokens.token_number`), the chronic-condition flags, each 2 (not
recorded) where `NOCHRON` says the whole item is blank, and `LABEL`, `ADMITHOS`. The rest of the
disposition block, every item recorded once the patient is admitted or leaves, and the weights
never enter. ADR `research/nhamcs-frame` records the frame.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.archive import member_text
from bko.datasets.cells import LABEL
from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.icd10_tokens import BLANK, NOT_ICD, is_token, token_number
from bko.datasets.sources import Source


logger = logging.getLogger(__name__)

NHAMCS_NAME: Final = "nhamcs"
"""The frame's name, its `Dataset` value and its catalog name."""

MEMBER: Final = "ed2022"
"""The archive member holding the records."""

WIDTH: Final = 2382
"""The characters of every record, before its CRLF."""

DIAGNOSES: Final = ("DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5")
"""The diagnosis items, in the order the visit lists them."""

ORDINALS: Final = ("first", "second", "third", "fourth", "fifth")
"""Each diagnosis item's position in the visit's list, in `DIAGNOSES` order."""

FLAGS: Final = (
    *("ETOHAB", "ALZHD", "ASTHMA", "CANCER", "CEBVD", "CKD", "COPD", "CHF", "CAD", "DEPRN"),
    *("DIABTYP1", "DIABTYP2", "DIABTYP0", "ESRD", "HPE", "EDHIV", "HYPLIPID", "HTN", "OBESITY"),
    *("OSA", "OSTPRSIS", "SUBSTAB"),
)
"""The chronic-condition items, each one column from 152 to 173, in file order."""

POSITIONS: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {
        "AGE": (16, 18),
        "SEX": (25, 25),
        **{name: (122 + 4 * k, 125 + 4 * k) for k, name in enumerate(DIAGNOSES)},
        **{name: (152 + k, 152 + k) for k, name in enumerate(FLAGS)},
        "NOCHRON": (174, 174),
        "ADMITHOS": (499, 499),
        "HOSPCODE": (544, 546),
        "PATCODE": (547, 549),
    }
)
"""Each item read and its first and last column, 1-based and inclusive, as `eddict2022.dct`
gives them."""

AGE_TOP: Final = 94
"""`AGE`'s top code: 94 means 94 years or older."""

SEX_CODES: Final[Mapping[str, int]] = MappingProxyType({"1": 0, "2": 1})
"""`SEX`'s codes, 1 female and 2 male, and their codes in the frame (`bko.datasets.cells.SEXES`)."""

FLAG_CODES: Final[Mapping[int, str]] = MappingProxyType({0: "no", 1: "yes", 2: "not recorded"})
"""A chronic flag's codes: the file's 0 and 1, and 2 where the whole item is blank."""

NOT_RECORDED: Final = 2
"""A flag's code, and `NOCHRON`'s, where the whole chronic item is blank."""

FEATURES: Final = ("age", "sex", *DIAGNOSES, *FLAGS)
"""The frame's features, in record order."""

_THREE_DIGITS: Final = re.compile(r"[0-9]{3}")
_LINE_END: Final = "\r\n"


def parse_nhamcs(text: str) -> pd.DataFrame:
    """Parse `ed2022` into one row of raw strings per record, its items as `POSITIONS` cuts them.

    Returns:
        Every item of `POSITIONS` but `HOSPCODE` and `PATCODE` as a string, in file order,
        indexed by `record_id`.

    Raises:
        ContractError: If the text does not end with CRLF, a line is not `WIDTH` characters,
            `HOSPCODE` or `PATCODE` is not three digits, or an id repeats.
    """
    if text and not text.endswith(_LINE_END):
        msg = f"{MEMBER}: the last line has no CRLF"
        raise ContractError(msg)
    lines = text.split(_LINE_END)[:-1]
    ids: list[int] = []
    rows: list[list[str]] = []
    names = [name for name in POSITIONS if name not in {"HOSPCODE", "PATCODE"}]
    for number, line in enumerate(lines, 1):
        if len(line) != WIDTH:
            msg = f"{MEMBER} line {number}: {len(line)} characters, expected {WIDTH}"
            raise ContractError(msg)
        hospital, visit = (_cut(line, name) for name in ("HOSPCODE", "PATCODE"))
        if not (_THREE_DIGITS.fullmatch(hospital) and _THREE_DIGITS.fullmatch(visit)):
            msg = (
                f"{MEMBER} line {number}: HOSPCODE {hospital!r} or PATCODE {visit!r} is not "
                "three digits"
            )
            raise ContractError(msg)
        ids.append(int(hospital) * 1000 + int(visit))
        rows.append([_cut(line, name) for name in names])
    index = pd.Index(ids, dtype="int64", name="record_id")
    if index.has_duplicates:
        msg = f"{MEMBER} record {index[index.duplicated()][0]}: listed more than once"
        raise ContractError(msg)
    return pd.DataFrame(rows, columns=names, index=index, dtype=str)


def _cut(line: str, name: str) -> str:
    first, last = POSITIONS[name]
    return line[first - 1 : last]


def checked(raw: pd.DataFrame) -> pd.DataFrame:
    """Return `raw` in record order once every item fits the documentation's codes.

    Raises:
        ContractError: If an item holds a value outside its codes, or `NOCHRON` contradicts the
            flags (0 with none set, 1 or 2 with one set). Checks run item by item in file order,
            and the first that fails names its first failing record in record order.
    """
    rows = raw.sort_index(kind="stable")
    ages = rows["AGE"].where(rows["AGE"].str.fullmatch(_THREE_DIGITS.pattern), "999")
    any_flag = (rows[list(FLAGS)] == "1").any(axis=1)
    checks: list[tuple[str, pd.Series, str]] = [
        ("AGE", ages.astype("int64") <= AGE_TOP, f"not 000 to {AGE_TOP:03d}"),
        ("SEX", rows["SEX"].isin(list(SEX_CODES)), "not 1 or 2"),
        *((name, rows[name].map(is_token), "not a diagnosis token") for name in DIAGNOSES),
        *((name, rows[name].isin(["0", "1"]), "not 0 or 1") for name in FLAGS),
        ("NOCHRON", rows["NOCHRON"].isin(["0", "1", "2"]), "not 0, 1 or 2"),
        ("NOCHRON", (rows["NOCHRON"] == "0") == any_flag, "at odds with the flags"),
        ("ADMITHOS", rows["ADMITHOS"].isin(["0", "1"]), "not 0 or 1"),
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


def frame_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the frame records of `raw`, in record order, after checking every row.

    Logs one `cohort` line with the records, the admissions, the visits whose chronic item is
    blank, the visits whose first diagnosis is blank, and the visits holding one of NCHS's own
    `ZZZ` tokens.

    Args:
        raw: The output of `parse_nhamcs`.

    Returns:
        `FEATURES` and `LABEL`, all int64, indexed by `record_id`.

    Raises:
        ContractError: As `checked`.
    """
    rows = checked(raw)
    blank = rows["NOCHRON"] == str(NOT_RECORDED)
    columns: dict[str, pd.Series] = {
        "age": rows["AGE"].astype("int64"),
        "sex": rows["SEX"].map(SEX_CODES),
        **{name: rows[name].map(token_number) for name in DIAGNOSES},
        **{name: rows[name].astype("int64").mask(blank, NOT_RECORDED) for name in FLAGS},
        LABEL: rows["ADMITHOS"].astype("int64"),
    }
    records = pd.DataFrame(columns).astype("int64")
    records.index.name = "record_id"
    logger.info(
        "cohort dataset=%s records=%d admitted=%d chronic_blank=%d first_blank=%d not_icd=%d",
        NHAMCS_NAME,
        len(records),
        int(records[LABEL].sum()),
        int(blank.sum()),
        int((rows["DIAG1"] == BLANK).sum()),
        int(rows[list(DIAGNOSES)].isin(NOT_ICD).any(axis=1).sum()),
    )
    return records


def load_nhamcs(data_dir: Path, *, sources: Sequence[Source]) -> pd.DataFrame:
    """Load the frame records from the verified snapshot of the one source in `sources`.

    This never fetches; `bko.datasets.fetch.ensure` does. Nothing is extracted: the member is read
    from the archive's bytes once they match the pin.

    Raises:
        ValueError: If `sources` holds other than one source.
        UnavailableError: If the snapshot is absent.
        ContractError: If the archive or its member contradicts its pin or schema, the member
            holds more than `bko.datasets.archive.MEMBER_LIMIT` bytes, or the records hold none or
            no admission.
    """
    if len(sources) != 1:
        msg = f"NHAMCS reads one archive, not {len(sources)}"
        raise ValueError(msg)
    [source] = sources
    data = read_verified(source, data_dir)
    records = frame_records(parse_nhamcs(member_text(data, MEMBER, source)))
    if records.empty:
        msg = f"{source.name}: the file holds no records"
        raise ContractError(msg)
    if not records[LABEL].any():
        msg = f"{source.name}: the file holds no admission"
        raise ContractError(msg)
    return records
