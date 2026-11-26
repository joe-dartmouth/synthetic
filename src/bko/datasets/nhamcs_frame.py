"""The NHAMCS frame: the codebook, descriptions and age range of `bko.datasets.nhamcs`'s records.

The five diagnosis columns share one codebook, the tokens the loaded records hold, each number
meaning its token (`bko.datasets.icd10_tokens.token_text`), so `nhamcs_frame` builds it from the
records, as it takes `age`'s range. ADR `research/nhamcs-frame` records the frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.datasets.icd10_tokens import BLANK, UNKNOWN, token_text
from bko.datasets.nhamcs import DIAGNOSES, FEATURES, FLAG_CODES, FLAGS, NHAMCS_NAME


_FLAG_ITEMS: Final[Mapping[str, str]] = {
    "ETOHAB": "alcohol misuse, abuse, or dependence",
    "ALZHD": "Alzheimer's disease or dementia",
    "ASTHMA": "asthma",
    "CANCER": "cancer",
    "CEBVD": "cerebrovascular disease, history of stroke or transient ischemic attack",
    "CKD": "chronic kidney disease",
    "COPD": "chronic obstructive pulmonary disease",
    "CHF": "congestive heart failure",
    "CAD": "coronary artery disease, ischemic heart disease or history of myocardial infarction",
    "DEPRN": "depression",
    "DIABTYP1": "diabetes mellitus, type I",
    "DIABTYP2": "diabetes mellitus, type II",
    "DIABTYP0": "diabetes mellitus, type unspecified",
    "ESRD": "end-stage renal disease",
    "HPE": "history of pulmonary embolism, deep vein thrombosis or venous thromboembolism",
    "EDHIV": "HIV infection or AIDS",
    "HYPLIPID": "hyperlipidemia",
    "HTN": "hypertension",
    "OBESITY": "obesity (BMI at least 30)",
    "OSA": "obstructive sleep apnea",
    "OSTPRSIS": "osteoporosis",
    "SUBSTAB": "substance abuse or dependence",
}

_POSITIONS: Final = ("first", "second", "third", "fourth", "fifth")

_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "age": "age in whole years at the visit (94 = 94 and over)",
    "sex": "sex",
    **{
        name: (
            f"ICD-10-CM code of the visit's {position} diagnosis, cut to four characters "
            f"({BLANK.strip()} = none recorded)"
        )
        for name, position in zip(DIAGNOSES, _POSITIONS, strict=True)
    },
    **{
        name: f"the patient has {item} (not recorded = the whole chronic item left blank)"
        for name, item in _FLAG_ITEMS.items()
    },
    LABEL: "admitted to this hospital",
}


def nhamcs_frame(records: pd.DataFrame) -> FrameSpec:
    """The frame of `records`, the output of `load_nhamcs`.

    Args:
        records: The frame records; their diagnosis tokens make the diagnosis codebook, and their
            least and greatest `age` its range.

    Returns:
        The NHAMCS frame: numeric `age`, every other feature coded, admission to this hospital as
        label, age bands from 18, 45 and 65.
    """
    present = sorted({int(code) for name in DIAGNOSES for code in records[name].unique()})
    book = MappingProxyType(
        {
            code: f"{BLANK.strip()} (none recorded)" if code == UNKNOWN else token_text(code)
            for code in present
        }
    )
    return FrameSpec(
        name=NHAMCS_NAME,
        features=FEATURES,
        numeric=("age",),
        fractional=(),
        fraction_digits=0,
        categorical=FEATURES[1:],
        codebook={
            "sex": SEXES,
            **dict.fromkeys(DIAGNOSES, book),
            **dict.fromkeys(FLAGS, FLAG_CODES),
        },
        descriptions=_DESCRIPTIONS,
        ranges=(
            RangeConstraint(
                field="age",
                unit="a",
                low=int(records["age"].min()),
                high=int(records["age"].max()),
                source="nhamcs-nd",
                limits=(
                    "the least and greatest age over the loaded records; the file top-codes age "
                    "at 94, so 94 stands for 94 and over"
                ),
            ),
        ),
        label_codes=tuple(sorted(LABEL_CODES)),
        label_meanings={0: "not admitted to this hospital", 1: "admitted to this hospital"},
        edges=(18, 45, 65),
        study="hospital admission from a US emergency-department visit in a national survey",
        cohort="emergency-department visit sampled by NHAMCS in 2022, of any age",
    )
