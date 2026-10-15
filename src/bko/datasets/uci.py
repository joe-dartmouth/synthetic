"""UCI Heart Disease, Cleveland records: the evaluation frame and its codes.

The frame has one row per complete record, indexed by `record_id`, the record's 1-based line number
in `processed.cleveland.data`. Its columns are the 13 UCI predictors under their UCI names with
their UCI integer codes (`CODEBOOK`), and `target`: 1 where `num` is 1 to 4 (disease present), 0
where it is 0. `num` is dropped, because a model given it reads the label off it. ADR
`research/uci-evaluation-frame` records why.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final


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
