"""UCI Heart Disease, Cleveland records: the evaluation frame and its codes.

The frame has one row per complete record, indexed by `record_id`, the record's 1-based line number
in `processed.cleveland.data`. Its columns are the 13 UCI predictors under their UCI names with
their UCI integer codes (`CODEBOOK`), and `target`: 1 where `num` is 1 to 4 (disease present), 0
where it is 0. `num` is dropped, because a model given it reads the label off it. ADR
`research/uci-evaluation-frame` records why.
"""

from __future__ import annotations

import io
import logging
import math
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.sources import UCI_HEART_DISEASE, Source


MEMBER: Final = "processed.cleveland.data"
"""The archive member the frame is read from."""

COLUMNS: Final = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
)
"""The 14 fields of a `processed.*.data` line, in file order."""

FEATURES: Final = COLUMNS[:-1]
"""The 13 predictors."""

LABEL: Final = "target"
"""The binary label column: 1 where `num` is 1 to 4."""

NUMERIC: Final = ("age", "trestbps", "chol", "thalach", "oldpeak")
"""The measured predictors."""

CATEGORICAL: Final = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
"""The coded predictors; each one's codes are in `CODEBOOK`."""

CODEBOOK: Final[Mapping[str, Mapping[int, str]]] = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    "fbs": {0: "fasting blood sugar <= 120 mg/dl", 1: "fasting blood sugar > 120 mg/dl"},
    "restecg": {0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
    "exang": {0: "no exercise induced angina", 1: "exercise induced angina"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "ca": {0: "0 major vessels", 1: "1 major vessel", 2: "2 major vessels", 3: "3 major vessels"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversable defect"},
}
"""Each coded predictor's codes and meanings, from the archive's `heart-disease.names`."""

NUM_CODES: Final = frozenset(range(5))
"""`num` runs from 0 (no disease) to 4."""

logger = logging.getLogger(__name__)

REQUIRED: Final = ("age", "sex", "num")
"""The fields a record's cell needs; a record missing one is refused, not dropped."""


def parse_cleveland(text: str) -> pd.DataFrame:
    """Parse `processed.cleveland.data` into one row per line, reading `?` as missing.

    Args:
        text: The member's text.

    Returns:
        The 14 `COLUMNS` as floats, indexed by `record_id`, the 1-based line number.

    Raises:
        ContractError: If a line has other than 14 fields, a field is neither a number nor `?`,
            a coded field is not a whole number in its codebook, or a `REQUIRED` field is missing.
    """
    ids: list[int] = []
    rows: list[list[float]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != len(COLUMNS):
            msg = f"{MEMBER} line {number}: {len(fields)} fields, expected {len(COLUMNS)}"
            raise ContractError(msg)
        rows.append([_value(f, name, number) for f, name in zip(fields, COLUMNS, strict=True)])
        ids.append(number)
    if not rows:
        msg = f"{MEMBER} holds no records"
        raise ContractError(msg)
    frame = pd.DataFrame(rows, columns=list(COLUMNS), index=pd.Index(ids, name="record_id"))
    _check_codes(frame)
    return frame


def complete_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop records with a missing value, derive `LABEL` from `num`, and drop `num`.

    Args:
        raw: The output of `parse_cleveland`.

    Returns:
        The evaluation frame the module docstring describes.
    """
    incomplete = raw.isna().any(axis=1)
    if incomplete.any():
        logger.warning(
            "dropped %d records with a missing value, record_ids=%s",
            int(incomplete.sum()),
            ",".join(str(record) for record in raw.index[incomplete]),
        )
    kept = raw.loc[~incomplete]
    integers = {name: "int64" for name in FEATURES if name != "oldpeak"}
    frame = kept.drop(columns="num").astype(integers)
    frame[LABEL] = (kept["num"] > 0).astype("int64")
    return frame


def load_uci(data_dir: Path, *, source: Source = UCI_HEART_DISEASE) -> pd.DataFrame:
    """Load the evaluation frame from the verified snapshot of `source` under `data_dir`.

    This never fetches; `bko.datasets.fetch.ensure` does.

    Args:
        data_dir: The data root.
        source: The pinned archive.

    Returns:
        The complete records: `FEATURES` as int64 (`oldpeak` as float64) and `LABEL`.

    Raises:
        UnavailableError: If the snapshot is absent.
        ContractError: If the archive or its member contradicts its pin or schema.
    """
    data = read_verified(source, data_dir)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            text = bundle.read(MEMBER).decode("ascii")
    except (KeyError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        msg = f"{source.name}: cannot read {MEMBER}: {exc}"
        raise ContractError(msg) from exc
    return complete_records(parse_cleveland(text))


def _value(field: str, name: str, number: int) -> float:
    raw = field.strip()
    if raw == "?":
        if name in REQUIRED:
            msg = f"{MEMBER} line {number}: {name} is missing"
            raise ContractError(msg)
        return math.nan
    try:
        value = float(raw)
    except ValueError:
        msg = f"{MEMBER} line {number}: {name} is {raw!r}, not a number"
        raise ContractError(msg) from None
    if not math.isfinite(value) or (name != "oldpeak" and not value.is_integer()):
        msg = f"{MEMBER} line {number}: {name} is {raw!r}, not a finite whole number"
        raise ContractError(msg)
    return value


def _check_codes(frame: pd.DataFrame) -> None:
    allowed = {name: sorted(codes) for name, codes in CODEBOOK.items()} | {"num": sorted(NUM_CODES)}
    for name, codes in allowed.items():
        present = frame[name].dropna()
        outside = present[~present.isin(codes)]
        if not outside.empty:
            msg = (
                f"{MEMBER} line {outside.index[0]}: {name} is {outside.iloc[0]:g}, outside {codes}"
            )
            raise ContractError(msg)
