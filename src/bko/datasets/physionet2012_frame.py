"""The PhysioNet 2012 frame: the codebook, descriptions and age range of its records.

Each coded measured field's states are named from its cutoffs, state 0 `not measured`. The age
range is the least and greatest age of the records the frame is built from, so
`physionet2012_frame` takes them. ADR `research/physionet2012-frame` records the frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.datasets.physionet2012 import FEATURES, ICU_TYPES, MEASURED, PHYSIONET2012_NAME
from bko.datasets.states import band_meanings


_WINDOW: Final = "over the stay's first 48 hours"

_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "age": "age in years on admission to the ICU",
    "sex": "sex",
    "ICUType": "the type of ICU the patient was admitted to",
    "hr_max": f"highest heart rate {_WINDOW}, beats per minute",
    "resprate_max": f"highest respiration rate {_WINDOW}, breaths per minute",
    "temp_max": f"highest temperature {_WINDOW}, degrees Celsius",
    "temp_min": f"lowest temperature {_WINDOW}, degrees Celsius",
    "wbc_max": f"highest white blood cell count {_WINDOW}, cells per nL",
    "wbc_min": f"lowest white blood cell count {_WINDOW}, cells per nL",
    "map_min": (
        f"lowest invasive mean arterial pressure {_WINDOW}, mm Hg, the non-invasive one where "
        "the stay has no invasive reading"
    ),
    "creatinine_max": f"highest serum creatinine {_WINDOW}, mg/dL",
    "bilirubin_max": f"highest bilirubin {_WINDOW}, mg/dL",
    "platelets_min": f"lowest platelet count {_WINDOW}, cells per nL",
    "gcs_min": f"lowest Glasgow Coma Score {_WINDOW}",
    "mechvent": f"mechanical ventilation recorded {_WINDOW}",
    LABEL: "died in hospital",
}

_CODEBOOK: Final[Mapping[str, Mapping[int, str]]] = {
    "sex": SEXES,
    "ICUType": ICU_TYPES,
    **{name: band_meanings(cut) for name, cut in MEASURED.items()},
    "mechvent": {0: "no ventilation recorded", 1: "ventilated"},
}


def physionet2012_frame(records: pd.DataFrame) -> FrameSpec:
    """The frame of `records`, the output of `load_physionet2012`.

    Args:
        records: The frame records; their least and greatest `age` make its range.

    Returns:
        The PhysioNet 2012 frame: numeric `age`, every other feature coded, in-hospital death as
        label, age bands from 50, 65 and 80.
    """
    return FrameSpec(
        name=PHYSIONET2012_NAME,
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
                source="silva-2012a",
                limits=(
                    "the least and greatest age over the loaded records; the challenge page "
                    "states no top code"
                ),
            ),
        ),
        label_codes=tuple(sorted(LABEL_CODES)),
        label_meanings={0: "survived the hospital stay", 1: "died in hospital"},
        edges=(50, 65, 80),
        study="in-hospital death of adult intensive-care patients",
        cohort="adult ICU stay of at least 48 hours in the PhysioNet/CinC Challenge 2012",
    )
