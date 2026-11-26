"""The NHANES frame: the codebook, descriptions and age range of `bko.datasets.nhanes`'s records.

Each coded measured field's states are named from its cutoffs, state 0 `not measured`. The age
range is the least and greatest age of the records the frame is built from, so `nhanes_frame`
takes them. ADR `research/nhanes-frame` records the frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.datasets.nhanes import FEATURES, MEASURED, NHANES_NAME
from bko.datasets.states import band_meanings


_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "age": "age in years at screening, 80 standing for 80 and over",
    "sex": "sex",
    "BPXOSY": "systolic blood pressure, the mean of up to three oscillometric readings, mm Hg",
    "BPXODI": "diastolic blood pressure, the mean of up to three oscillometric readings, mm Hg",
    "LBXGH": "glycohemoglobin, %",
    "LBXTC": "total cholesterol, mg/dL",
    "LBDHDD": "direct HDL cholesterol, mg/dL",
    "BMXBMI": "body mass index, kg/m2, coded at the kept records' tertiles",
    LABEL: (
        "ever told by a doctor or other health professional of congestive heart failure, coronary "
        "heart disease, a heart attack or a stroke"
    ),
}

_CODEBOOK: Final[Mapping[str, Mapping[int, str]]] = {
    "sex": SEXES,
    **{name: band_meanings(cut) for name, cut in MEASURED.items()},
}


def nhanes_frame(records: pd.DataFrame) -> FrameSpec:
    """The frame of `records`, the output of `load_nhanes`.

    Args:
        records: The frame records; their least and greatest `age` make its range.

    Returns:
        The NHANES frame: numeric `age`, every other feature coded, the cardiovascular composite
        as label, age bands from 40 and 60.
    """
    return FrameSpec(
        name=NHANES_NAME,
        features=FEATURES,
        numeric=("age",),
        fractional=(),
        fraction_digits=0,
        categorical=FEATURES[1:],
        codebook=_CODEBOOK,
        descriptions=_DESCRIPTIONS,
        ranges=(
            RangeConstraint(
                field="age",
                unit="a",
                low=int(records["age"].min()),
                high=int(records["age"].max()),
                source="nhanes-2024",
                limits=(
                    "the least and greatest age over the loaded records; the file top-codes age "
                    "at 80, so 80 stands for 80 and over"
                ),
            ),
        ),
        label_codes=tuple(sorted(LABEL_CODES)),
        label_meanings={
            0: "answered no to all four conditions",
            1: "answered yes to at least one of the four conditions",
        },
        edges=(40, 60),
        study="self-reported cardiovascular disease of US adults in a national health survey",
        cohort="US adult of 20 and over interviewed in NHANES August 2021 to August 2023",
    )
