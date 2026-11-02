"""UCI Diabetes 130-US Hospitals: one frame record per adult patient's first encounter.

`load_diabetes` reads `diabetic_data.csv` and `IDS_mapping.csv` from the verified
`uci-diabetes-130` snapshot without extracting it. The cohort keeps each patient's first encounter
(the lowest `encounter_id`), then drops encounters discharged expired or to hospice, then
`Unknown/Invalid` gender, then age brackets under 20. A record holds coded demographics, the eight
count fields, the lab and drug states, `change`, `diabetesMed`, one indicator per caller-given
ICD-9 category list and `LABEL`, 1 for readmission in under 30 days. `encounter_id` and
`patient_nbr` never enter it. `docs/DEVELOPMENT.md` (Data, The Diabetes frame) states the contract.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL
from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.sources import DIABETES_130, Source


logger = logging.getLogger(__name__)

MEMBER: Final = "diabetic_data.csv"
"""The archive member holding the encounters."""

IDS_MEMBER: Final = "IDS_mapping.csv"
"""The archive member naming each admission type, discharge disposition and admission source."""

COLUMNS: Final = (
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmitted",
)
"""The 50 fields of `diabetic_data.csv`, in file order; the header must be exactly these."""

COUNTS: Final = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)
"""The eight count fields, kept as whole numbers."""

DRUGS: Final = (
    "insulin",
    "metformin",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "glimepiride",
    "repaglinide",
)
"""The eight drug columns most often other than `No` across the archive's encounters, most first.

The archive's other 15 drug columns are never read."""

DIAGNOSES: Final = ("diag_1", "diag_2", "diag_3")
"""The diagnosis columns an indicator reads."""

AGE_BRACKETS: Final[Mapping[str, int]] = {f"[{10 * d}-{10 * (d + 1)})": d for d in range(10)}
"""Each `age` bracket and its decade index, `[0-10)` 0 to `[90-100)` 9."""

ADULTS_FROM: Final = 2
"""The lowest decade index kept: brackets under 20 are dropped."""

SEX_CODES: Final[Mapping[str, int]] = {"Female": 0, "Male": 1}
"""The `gender` values kept and their `sex` codes (`bko.datasets.cells.SEXES`)."""

UNKNOWN_GENDER: Final = "Unknown/Invalid"
"""The `gender` value whose encounters are dropped."""

RACE_CODES: Final[Mapping[str, int]] = {
    "?": 0,
    "Caucasian": 1,
    "AfricanAmerican": 2,
    "Hispanic": 3,
    "Asian": 4,
    "Other": 5,
}
"""Each `race` value and its code; `?` is code 0, not recorded."""

A1C_CODES: Final[Mapping[str, int]] = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
"""Each `A1Cresult` state and its code; `None` (not measured) is a state, not missing."""

GLU_CODES: Final[Mapping[str, int]] = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
"""Each `max_glu_serum` state and its code; `None` (not measured) is a state, not missing."""

DRUG_CODES: Final[Mapping[str, int]] = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
"""Each drug column's states and their codes."""

CHANGE_CODES: Final[Mapping[str, int]] = {"No": 0, "Ch": 1}
"""`change`: 1 where a diabetes medication's dose changed."""

DIABETES_MED_CODES: Final[Mapping[str, int]] = {"No": 0, "Yes": 1}
"""`diabetesMed`: 1 where any diabetes medication was prescribed."""

READMITTED_LABELS: Final[Mapping[str, int]] = {"NO": 0, ">30": 0, "<30": 1}
"""Each `readmitted` value and its `LABEL`: 1 for readmission in under 30 days."""

EXCLUDED_DISCHARGES: Final = frozenset({11, 13, 14, 19, 20, 21})
"""The `discharge_disposition_id`s `IDS_mapping.csv` names Expired or Hospice; `load_diabetes`
refuses an archive whose mapping names others."""

STATE_CODES: Final[Mapping[str, Mapping[str, int]]] = {
    "race": RACE_CODES,
    "A1Cresult": A1C_CODES,
    "max_glu_serum": GLU_CODES,
    **dict.fromkeys(DRUGS, DRUG_CODES),
    "change": CHANGE_CODES,
    "diabetesMed": DIABETES_MED_CODES,
}
"""Every coded column of a record besides `age` and `sex`, in record order, and its codes."""

MEMBER_LIMIT: Final = 64 * 2**20
"""The most bytes a member may hold once decompressed; the archive's largest holds under 20 MiB."""

_WHOLE: Final = r"[0-9]{1,18}"
"""An ASCII whole number that fits int64; `\\d` and `int` also read other scripts' digits."""
_NUMERIC_CODE: Final = r"([0-9]{1,3})(?:\.([0-9]{1,2}))?"
_DIAGNOSIS: Final = rf"\?|{_NUMERIC_CODE}|V[0-9]{{2}}(?:\.[0-9]{{1,2}})?|E[0-9]{{3}}(?:\.[0-9])?"


def parse_diabetic(text: str) -> pd.DataFrame:
    """Parse `diabetic_data.csv` into one row of strings per line, after checking its header.

    Every field stays a string, so `None` and `?` are values, never missing.

    Args:
        text: The member's text.

    Returns:
        The 50 `COLUMNS` as strings, in file order, indexed by `line`, the physical line each row
        ends on, so a quoted line break inside a field keeps later rows' numbers right.

    Raises:
        ContractError: If the header is not `COLUMNS` or a line has other than 50 fields.
    """
    reader = csv.reader(io.StringIO(text, newline=""))
    header = tuple(next(reader, ()))
    if header != COLUMNS:
        msg = f"{MEMBER}: the header is not the archive's {len(COLUMNS)} columns in order"
        raise ContractError(msg)
    rows: list[list[str]] = []
    lines: list[int] = []
    try:
        for row in reader:
            if len(row) != len(COLUMNS):
                msg = f"{MEMBER} line {reader.line_num}: {len(row)} fields, expected {len(COLUMNS)}"
                raise ContractError(msg)
            rows.append(row)
            lines.append(reader.line_num)
    except csv.Error as exc:
        msg = f"{MEMBER} line {reader.line_num}: {exc}"
        raise ContractError(msg) from exc
    index = pd.Index(lines, dtype="int64", name="line")
    return pd.DataFrame(rows, columns=list(COLUMNS), index=index, dtype=str)


def excluded_discharges(ids: str) -> frozenset[int]:
    """Return the `discharge_disposition_id`s whose description names Expired or Hospice.

    Args:
        ids: The text of `IDS_mapping.csv`: blocks of `<id column>,description` rows, each ended
            by a blank row.

    Returns:
        The ids of the discharge block's rows whose description holds `expired` or `hospice`, in
        any case.

    Raises:
        ContractError: If there is no discharge block, a block's header row has an empty first
            cell, one of the discharge ids is not a whole number, or the text is not CSV.
    """
    found: set[int] = set()
    block: str | None = None
    seen = False
    reader = csv.reader(io.StringIO(ids, newline=""))
    try:
        rows = list(reader)
    except csv.Error as exc:
        msg = f"{IDS_MEMBER} line {reader.line_num}: {exc}"
        raise ContractError(msg) from exc
    for row in rows:
        cells = [cell.strip() for cell in row]
        if not any(cells):
            block = None
        elif block is None:
            if not cells[0]:
                msg = f"{IDS_MEMBER}: a block header with an empty first cell: {row!r}"
                raise ContractError(msg)
            block = cells[0]
            seen = seen or block == "discharge_disposition_id"
        elif block == "discharge_disposition_id":
            if not re.fullmatch(_WHOLE, cells[0]):
                msg = f"{IDS_MEMBER}: discharge_disposition_id {cells[0]!r} is not a whole number"
                raise ContractError(msg)
            description = " ".join(cells[1:]).lower()
            if "expired" in description or "hospice" in description:
                found.add(int(cells[0]))
    if not seen:
        msg = f"{IDS_MEMBER}: no discharge_disposition_id block"
        raise ContractError(msg)
    return frozenset(found)


def frame_records(raw: pd.DataFrame, *, indicators: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    """Return the frame records of `raw`'s cohort, in encounter order.

    Every row is checked before any is dropped, in encounter order, so an error names the same
    encounter whatever the file order. The cohort keeps each patient's lowest `encounter_id`, then
    drops `EXCLUDED_DISCHARGES`, `UNKNOWN_GENDER` and brackets under `ADULTS_FROM`, and logs one
    `cohort` line that accounts for every row.

    Args:
        raw: The output of `parse_diabetic`.
        indicators: Each indicator column's numeric ICD-9 categories, in column order. A column is
            1 where any of `DIAGNOSES` falls in one of its categories: `250.4` holds `250.4` and
            `250.41`, `401` holds `401` and `401.9`, never `4010` or `40`; `?` and V and E codes
            fall in none.

    Returns:
        `age` (the decade index), `sex`, `race`, `COUNTS`, `A1Cresult`, `max_glu_serum`, `DRUGS`,
        `change`, `diabetesMed`, each indicator and `LABEL`, all int64, indexed by `record_id`,
        the 1-based position in encounter order.

    Raises:
        ValueError: If an indicator's name is taken or one of its codes is not a numeric ICD-9
            category.
        ContractError: If a value contradicts the archive's schema.
    """
    categories = _categories(indicators)
    rows = _checked(raw)
    patients = rows["patient_nbr"].astype("int64")
    first = rows.loc[~patients.duplicated()]
    kept = first.loc[~first["discharge_disposition_id"].astype("int64").isin(EXCLUDED_DISCHARGES)]
    discharged = len(kept)
    kept = kept.loc[kept["gender"] != UNKNOWN_GENDER]
    gendered = len(kept)
    kept = kept.loc[kept["age"].map(AGE_BRACKETS) >= ADULTS_FROM]
    columns: dict[str, pd.Series] = {
        "age": kept["age"].map(AGE_BRACKETS),
        "sex": kept["gender"].map(SEX_CODES),
        "race": kept["race"].map(RACE_CODES),
        **{name: kept[name].astype("int64") for name in COUNTS},
        **{name: kept[name].map(codes) for name, codes in STATE_CODES.items() if name != "race"},
        **_indicators(kept, categories),
        LABEL: kept["readmitted"].map(READMITTED_LABELS),
    }
    records = pd.DataFrame({name: values.to_numpy() for name, values in columns.items()})
    records = records.astype("int64")
    records.index = pd.RangeIndex(1, len(records) + 1, name="record_id")
    logger.info(
        "cohort encounters=%d repeat_encounters=%d excluded_discharges=%d unknown_gender=%d "
        "under_20=%d records=%d readmitted_under_30=%d",
        len(rows),
        len(rows) - len(first),
        len(first) - discharged,
        discharged - gendered,
        gendered - len(records),
        len(records),
        int(records[LABEL].sum()),
    )
    return records


def load_diabetes(
    data_dir: Path,
    *,
    indicators: Mapping[str, tuple[str, ...]],
    source: Source = DIABETES_130,
) -> pd.DataFrame:
    """Load the frame records from the verified snapshot of `source` under `data_dir`.

    This never fetches; `bko.datasets.fetch.ensure` does. Nothing is extracted: both members are
    read from the archive's bytes once they match the pin.

    Args:
        data_dir: The data root.
        indicators: As `frame_records` takes them.
        source: The pinned archive.

    Returns:
        The records `frame_records` returns.

    Raises:
        UnavailableError: If the snapshot is absent.
        ContractError: If the archive or a member contradicts its pin or schema, a member holds
            more than `MEMBER_LIMIT` bytes, `IDS_mapping.csv` names other Expired or Hospice
            discharges than `EXCLUDED_DISCHARGES`, or the cohort holds no record or no record
            readmitted in under 30 days.
        ValueError: As `frame_records` raises it.
    """
    data = read_verified(source, data_dir)
    texts = {member: _member(data, member, source) for member in (MEMBER, IDS_MEMBER)}
    named = excluded_discharges(texts[IDS_MEMBER])
    if named != EXCLUDED_DISCHARGES:
        msg = (
            f"{source.name}: {IDS_MEMBER} names discharges {sorted(named)} Expired or Hospice, "
            f"not {sorted(EXCLUDED_DISCHARGES)}"
        )
        raise ContractError(msg)
    records = frame_records(parse_diabetic(texts[MEMBER]), indicators=indicators)
    if records.empty:
        msg = f"{source.name}: the cohort holds no records"
        raise ContractError(msg)
    if not records[LABEL].any():
        msg = f"{source.name}: the cohort holds no record readmitted in under 30 days"
        raise ContractError(msg)
    return records


def _member(data: bytes, member: str, source: Source) -> str:
    """Return `member` of the zip `data` as UTF-8 text, refusing one over `MEMBER_LIMIT` bytes."""
    where = f"{source.name}: cannot read {member}"
    try:
        return _capped(data, member, where).decode("utf-8")
    except KeyError as exc:
        msg = f"{where}: the archive has no such member"
        raise ContractError(msg) from exc
    except (
        zipfile.BadZipFile,
        zlib.error,
        UnicodeDecodeError,
        NotImplementedError,
        RuntimeError,
        EOFError,
    ) as exc:
        msg = f"{where}: {type(exc).__name__}: {exc}"
        raise ContractError(msg) from exc


def _capped(data: bytes, member: str, where: str) -> bytes:
    """Return `member`'s bytes, refusing one whose stated or read size passes `MEMBER_LIMIT`."""
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        size = bundle.getinfo(member).file_size
        if size > MEMBER_LIMIT:
            msg = f"{where}: {size} bytes, over the {MEMBER_LIMIT}-byte cap"
            raise ContractError(msg)
        with bundle.open(member) as handle:
            body = handle.read(MEMBER_LIMIT + 1)
    if len(body) > MEMBER_LIMIT:
        msg = f"{where}: over the {MEMBER_LIMIT}-byte cap"
        raise ContractError(msg)
    return body


def _categories(
    indicators: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[tuple[int, str], ...]]:
    """Return each indicator's categories as (whole part, decimal digits), after checking them."""
    taken = {"age", "sex", *COUNTS, *STATE_CODES, LABEL}
    parsed: dict[str, tuple[tuple[int, str], ...]] = {}
    for name, codes in indicators.items():
        if not name or name in taken:
            msg = f"indicator {name!r}: the name is empty or a frame column"
            raise ValueError(msg)
        if not codes:
            msg = f"indicator {name!r}: no ICD-9 categories"
            raise ValueError(msg)
        parts: list[tuple[int, str]] = []
        for code in codes:
            match = re.fullmatch(_NUMERIC_CODE, code)
            if match is None:
                msg = f"indicator {name!r}: {code!r} is not a numeric ICD-9 category"
                raise ValueError(msg)
            parts.append((int(match[1]), match[2] or ""))
        parsed[name] = tuple(parts)
    return parsed


def _checked(raw: pd.DataFrame) -> pd.DataFrame:
    """Return `raw` sorted by `encounter_id` once every value it is read for fits the schema."""
    malformed = ~raw["encounter_id"].str.fullmatch(_WHOLE).to_numpy(dtype=bool)
    if malformed.any():
        first = int(malformed.argmax())
        value = raw["encounter_id"].iloc[first]
        msg = f"{MEMBER} line {raw.index[first]}: encounter_id is {value!r}, not a whole number"
        raise ContractError(msg)
    encounters = raw["encounter_id"].astype("int64")
    order = encounters.sort_values(kind="stable").index
    rows = raw.loc[order].reset_index(drop=True)
    ids = encounters.loc[order].to_numpy()
    repeated = pd.Series(ids).duplicated()
    if repeated.any():
        msg = f"{MEMBER} encounter {ids[repeated.to_numpy().argmax()]}: listed more than once"
        raise ContractError(msg)
    allowed: dict[str, Mapping[str, int] | frozenset[str]] = {
        "gender": frozenset({*SEX_CODES, UNKNOWN_GENDER}),
        "age": AGE_BRACKETS,
        **STATE_CODES,
        "readmitted": READMITTED_LABELS,
    }
    checks: list[tuple[str, pd.Series, str]] = [
        *(
            (name, rows[name].str.fullmatch(_WHOLE), "not a whole number")
            for name in ("patient_nbr", "discharge_disposition_id", *COUNTS)
        ),
        *(
            (name, rows[name].isin(list(values)), f"not one of {sorted(values)}")
            for name, values in allowed.items()
        ),
        *(
            (name, rows[name].str.fullmatch(_DIAGNOSIS), "not an ICD-9 code or ?")
            for name in DIAGNOSES
        ),
    ]
    for name, fits, why in checks:
        bad = ~fits.to_numpy(dtype=bool)
        if bad.any():
            first = int(bad.argmax())
            msg = f"{MEMBER} encounter {ids[first]}: {name} is {rows[name].iloc[first]!r}, {why}"
            raise ContractError(msg)
    return rows


def _indicators(
    rows: pd.DataFrame, categories: Mapping[str, tuple[tuple[int, str], ...]]
) -> dict[str, pd.Series]:
    """Return each indicator: 1 where any diagnosis falls in one of its categories."""
    parts = [rows[name].str.extract(rf"^{_NUMERIC_CODE}$") for name in DIAGNOSES]
    numbers = [(pd.to_numeric(part[0]), part[1].fillna("")) for part in parts]
    held: dict[str, pd.Series] = {}
    for name, codes in categories.items():
        hit = pd.Series(False, index=rows.index)
        for whole, digits in numbers:
            for major, minor in codes:
                hit |= whole.eq(major) & digits.str.startswith(minor)
        held[name] = hit.astype("int64")
    return held
