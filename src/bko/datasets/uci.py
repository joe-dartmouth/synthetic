"""UCI Heart Disease, Cleveland records: the evaluation frame and its codes.

The frame has one row per complete record, indexed by `record_id`, the record's 1-based line number
in `processed.cleveland.data`. Its columns are the 13 UCI predictors under their UCI names with
their UCI integer codes (`CODEBOOK`), and `target`: 1 where `num` is 1 to 4 (disease present), 0
where it is 0. `num` is dropped, because a model given it reads the label off it. `UCI_FRAME`
holds what the stages read of it. ADR `research/uci-evaluation-frame` records why.
"""

from __future__ import annotations

import io
import logging
import math
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.cells import AGE_BAND_EDGES, LABEL, LABEL_CODES, LABEL_MEANINGS, SEXES
from bko.datasets.errors import ContractError
from bko.datasets.fetch import read_verified
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.datasets.sources import UCI_HEART_DISEASE, Source


logger = logging.getLogger(__name__)

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

NUMERIC: Final = ("age", "trestbps", "chol", "thalach", "oldpeak")
"""The measured predictors."""

FRACTIONAL: Final = ("oldpeak",)
"""The predictors that may hold a fraction; every other field is a whole number."""

CATEGORICAL: Final = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
"""The coded predictors; each one's codes are in `CODEBOOK`."""

CODEBOOK: Final[Mapping[str, Mapping[int, str]]] = {
    "sex": SEXES,
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    "fbs": {0: "fasting blood sugar <= 120 mg/dl", 1: "fasting blood sugar > 120 mg/dl"},
    "restecg": {0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
    "exang": {0: "no exercise induced angina", 1: "exercise induced angina"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "ca": {0: "0 major vessels", 1: "1 major vessel", 2: "2 major vessels", 3: "3 major vessels"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversable defect"},
}
"""Each coded predictor's codes and meanings, from the archive's `heart-disease.names`."""

DESCRIPTIONS: Final[Mapping[str, str]] = {
    "age": "age in years",
    "sex": "sex",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure (in mm Hg on admission to the hospital)",
    "chol": "serum cholestoral in mg/dl",
    "fbs": "fasting blood sugar > 120 mg/dl",
    "restecg": "resting electrocardiographic results",
    "thalach": "maximum heart rate achieved",
    "exang": "exercise induced angina",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "the slope of the peak exercise ST segment",
    "ca": "number of major vessels (0-3) colored by flourosopy",
    "thal": "thallium scintigraphy result",
    "target": "diagnosis of heart disease (angiographic disease status)",
}
"""What each frame column records: the archive's `heart-disease.names` text before its code list.

`thal` has only codes there, so its text reads it as Detrano's Cleveland Clinic protocol does
(`detrano-1985b`), as its row in `bko.ontology.mapping` does. `target` takes the text of `num`,
which it collapses.
"""

NUM_CODES: Final = frozenset(range(5))
"""`num` runs from 0 (no disease) to 4."""

REQUIRED: Final = ("age", "sex", "num")
"""The fields a record's cell needs; a record missing one is refused, not dropped."""

FRAME_COLUMNS: Final = (*FEATURES, LABEL)
"""The columns of the training split and of every added frame."""

FRAME_DTYPES: Final[Mapping[str, str]] = {
    name: "float64" if name in FRACTIONAL else "int64" for name in FRAME_COLUMNS
}
"""The evaluation frame's dtypes: whole numbers as int64, `oldpeak` as float64."""

FRACTION_DIGITS: Final = 1
"""The decimal places `oldpeak` has in the archive, which added values keep."""

RANGES: Final[tuple[RangeConstraint, ...]] = (
    RangeConstraint(
        field="age",
        unit="a",
        low=18,
        high=120,
        source="research/ontology-mapping-design",
        limits=(
            "team-specified: adults only; the team found no published plausibility bound; awaits "
            "clinician review"
        ),
    ),
    RangeConstraint(
        field="trestbps",
        unit="mm[Hg]",
        low=10,
        high=250,
        source="blacketer-2021",
        limits=(
            'OHDSI DQD v1.4.1 plausible range for concept 4161413 "Systolic blood pressure on '
            'admission" (`inst/csv/OMOP_CDMv5.3.1_Concept_Level.csv`); a tool setting, not a '
            "clinical threshold"
        ),
    ),
    RangeConstraint(
        field="chol",
        unit="mg/dL",
        low=32,
        high=600,
        source="harutyunyan-2019",
        limits=(
            'MIMIC-III benchmark valid range for "Cholesterol" '
            "(`mimic3benchmark/resources/variable_ranges.csv` at commit "
            "ea0314c7cbd369f62e2237ace6f683740f867e3a); DQD's 50 to 500 rejects a real Cleveland "
            "record"
        ),
    ),
    RangeConstraint(
        field="thalach",
        unit="/min",
        low=10,
        high=500,
        source="blacketer-2021",
        limits=(
            'OHDSI DQD v1.4.1 plausible range for concept 3027018 "Heart rate"; age-predicted '
            "maxima are means, not bounds"
        ),
    ),
    RangeConstraint(
        field="oldpeak",
        unit="mm",
        low=-10,
        high=10,
        source="research/ontology-mapping-design",
        limits=(
            "team-specified: the team found no published bound; negative values are ST elevation, "
            "which other UCI sites record; mm is the team's reading; awaits clinician review"
        ),
    ),
)
"""One inclusive plausibility range per numeric field, in UCUM units.

`bko.ontology.mapping` re-exports it beside the mapping table.
"""

UCI_FRAME: Final = FrameSpec(
    name="uci",
    features=FEATURES,
    numeric=NUMERIC,
    fractional=FRACTIONAL,
    fraction_digits=FRACTION_DIGITS,
    categorical=CATEGORICAL,
    codebook=CODEBOOK,
    descriptions=DESCRIPTIONS,
    ranges=RANGES,
    label_codes=tuple(sorted(LABEL_CODES)),
    label_meanings=LABEL_MEANINGS,
    edges=AGE_BAND_EDGES,
    label=LABEL,
)
"""UCI Heart Disease's frame: what every stage reads of it, built from this module's constants."""


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

    Raises:
        ContractError: If no record is complete.
    """
    incomplete = raw.isna().any(axis=1)
    if incomplete.any():
        logger.warning(
            "dropped %d records with a missing value, record_ids=%s",
            int(incomplete.sum()),
            ",".join(str(record) for record in raw.index[incomplete]),
        )
    kept = raw.loc[~incomplete]
    if kept.empty:
        msg = f"{MEMBER} holds no complete records"
        raise ContractError(msg)
    frame = kept.drop(columns="num").astype({name: FRAME_DTYPES[name] for name in FEATURES})
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
    except KeyError as exc:
        msg = f"{source.name}: cannot read {MEMBER}: the archive has no such member"
        raise ContractError(msg) from exc
    except (zipfile.BadZipFile, zlib.error, UnicodeDecodeError) as exc:
        msg = f"{source.name}: cannot read {MEMBER}: {type(exc).__name__}: {exc}"
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
    if not math.isfinite(value):
        msg = f"{MEMBER} line {number}: {name} is {raw!r}, not a finite number"
        raise ContractError(msg)
    if name not in FRACTIONAL and not value.is_integer():
        msg = f"{MEMBER} line {number}: {name} is {raw!r}, not a whole number"
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
