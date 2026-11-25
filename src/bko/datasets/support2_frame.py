"""The SUPPORT2 frame: the codebook, descriptions and ranges of `bko.datasets.support2`'s records.

Each coded measured field's states are named from its cutoffs, state 0 `not measured`. The ranges
of the numeric fields are the least and greatest values of the records the frame is built from,
so `support2_frame` takes them. ADR `research/support2-frame` records the frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.datasets.support2 import (
    CA_CODES,
    DZGROUP_CODES,
    FEATURES,
    INCOME_CODES,
    MEASURED,
    RACE_CODES,
    SUPPORT2_NAME,
    WHOLE_FIELDS,
    Cutoffs,
)


def band_meanings(cut: Cutoffs) -> Mapping[int, str]:
    """Each state of a field coded at `cut` and what it means, state 0 `not measured`."""
    bounds = sorted([(c, True) for c in cut.at_or_above] + [(c, False) for c in cut.above])
    meanings = {0: "not measured"}
    for state in range(len(bounds) + 1):
        low = bounds[state - 1] if state else None
        high = bounds[state] if state < len(bounds) else None
        parts = []
        if low is not None:
            parts.append(f"at least {low[0]:g}" if low[1] else f"above {low[0]:g}")
        if high is not None:
            parts.append(f"below {high[0]:g}" if high[1] else f"at most {high[0]:g}")
        meanings[state + 1] = " and ".join(parts) or "measured"
    return MappingProxyType(meanings)


_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "age": "age in whole years at study entry",
    "sex": "sex",
    "dzgroup": "the patient's disease group at study entry",
    "num.co": "number of comorbidities",
    "diabetes": "diabetes as a comorbidity",
    "dementia": "dementia as a comorbidity",
    "ca": "cancer: none, present, or metastatic",
    "hday": "day in hospital at study entry",
    "scoma": "SUPPORT coma score based on the Glasgow scale on study day 3, 0 to 100",
    "race": "race (not recorded = missing)",
    "income": "income (not recorded = missing)",
    "edu": "years of education, coded at the file's tertiles",
    "hrt": "heart rate on study day 3, per minute",
    "resp": "respiration rate on study day 3, per minute",
    "temp": "temperature on study day 3, degrees Celsius",
    "wblc": "white blood cell count on study day 3, in thousands per cubic millimetre",
    "meanbp": "mean arterial blood pressure on study day 3, mm Hg",
    "crea": "serum creatinine on study day 3, mg/dL",
    "bili": "bilirubin on study day 3, mg/dL",
    "pafi": "PaO2/FiO2 ratio on study day 3, mm Hg",
    "sod": "serum sodium on study day 3, coded at the file's tertiles",
    "alb": "serum albumin on study day 3, coded at the file's tertiles",
    "ph": "arterial serum pH on study day 3, coded at the file's tertiles",
    "glucose": "glucose on study day 3, coded at the file's tertiles",
    "bun": "blood urea nitrogen on study day 3, coded at the file's tertiles",
    "urine": "urine output on study day 3, coded at the file's tertiles",
    LABEL: "died in hospital",
}

_UNITS: Final[Mapping[str, str]] = {
    "age": "a",
    "num.co": "{comorbidities}",
    "hday": "d",
    "scoma": "{score}",
}

_CODEBOOK: Final[Mapping[str, Mapping[int, str]]] = {
    "sex": SEXES,
    "dzgroup": {code: level for level, code in DZGROUP_CODES.items()},
    "diabetes": {0: "absent", 1: "present"},
    "dementia": {0: "absent", 1: "present"},
    "ca": {code: level for level, code in CA_CODES.items()},
    "race": {code: level or "not recorded" for level, code in RACE_CODES.items()},
    "income": {code: level or "not recorded" for level, code in INCOME_CODES.items()},
    **{name: band_meanings(cut) for name, cut in MEASURED.items()},
}


def support2_frame(records: pd.DataFrame) -> FrameSpec:
    """The frame of `records`, the output of `load_support2`.

    Args:
        records: The frame records; their least and greatest value of each numeric field make
            its range.

    Returns:
        The SUPPORT2 frame: numeric `age` and `WHOLE_FIELDS`, every other feature coded, label
        in-hospital death, age bands from 50, 65 and 80.
    """
    numeric = ("age", *WHOLE_FIELDS)
    return FrameSpec(
        name=SUPPORT2_NAME,
        features=FEATURES,
        numeric=numeric,
        fractional=(),
        fraction_digits=0,
        categorical=tuple(name for name in FEATURES if name not in numeric),
        codebook=_CODEBOOK,
        descriptions=_DESCRIPTIONS,
        ranges=tuple(
            RangeConstraint(
                field=name,
                unit=_UNITS[name],
                low=int(records[name].min()),
                high=int(records[name].max()),
                source="vanderbilt-2026",
                limits=(
                    "the least and greatest value over the loaded records; a value outside them "
                    "is unseen, not impossible"
                ),
            )
            for name in numeric
        ),
        label_codes=tuple(sorted(LABEL_CODES)),
        label_meanings={0: "survived the hospital stay", 1: "died in hospital"},
        edges=(50, 65, 80),
        study="in-hospital death of seriously ill hospitalized adults",
        cohort="seriously ill adult enrolled in SUPPORT at a US academic medical center",
    )
