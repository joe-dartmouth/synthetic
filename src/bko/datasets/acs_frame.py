"""The ACS frame: the codebook, descriptions and ranges of `bko.datasets.acs`'s records.

`OCCP` and `POBP` are coded by the codes the records hold, each meaning its code as the file writes
it, so the frame is built from the records, as are the ranges of `age` and `WKHP`. Every other
coded field's meanings are the 2024 data dictionary's (`bko.datasets.acs.CODEBOOK`). ADR
`research/acs-frame` records the frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pandas as pd

from bko.datasets.acs import ACS_NAME, CODEBOOK, FEATURES, THRESHOLD
from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.frames import FrameSpec, RangeConstraint


_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "age": "age in years (AGEP, top-coded)",
    "sex": "sex",
    "COW": "class of worker",
    "SCHL": "educational attainment",
    "MAR": "marital status",
    "OCCP": "occupation recode for 2018 and later based on 2018 OCC codes, as a whole number",
    "POBP": "place of birth (recode), as a whole number",
    "WKHP": "usual hours worked per week past 12 months",
    "RAC1P": "recoded detailed race code",
    "RELSHIPP": "relationship to the household's reference person",
    "DEAR": "hearing difficulty",
    "DEYE": "vision difficulty",
    "DREM": "cognitive difficulty",
    "DPHY": "ambulatory difficulty",
    "DDRS": "self-care difficulty",
    "DOUT": "independent living difficulty",
    LABEL: f"total person's income above ${THRESHOLD:,}, unadjusted",
}

_UNITS: Final[Mapping[str, str]] = {"age": "a", "WKHP": "h/wk"}

_WIDTHS: Final[Mapping[str, int]] = {"OCCP": 4, "POBP": 3}
"""The fields coded by the codes the records hold, and the width the file writes each code at."""


def acs_frame(records: pd.DataFrame) -> FrameSpec:
    """The frame of `records`, the output of `load_acs`.

    Args:
        records: The frame records; their `OCCP` and `POBP` codes make those fields' codebooks,
            and their least and greatest `age` and `WKHP` make those fields' ranges.

    Returns:
        The ACS frame: numeric `age` and `WKHP`, every other feature coded, label income above
        `THRESHOLD`, age bands from 30, 45 and 60.
    """
    numeric = tuple(_UNITS)
    held = {
        name: {int(code): f"{int(code):0{width}d}" for code in sorted(records[name].unique())}
        for name, width in _WIDTHS.items()
    }
    return FrameSpec(
        name=ACS_NAME,
        features=FEATURES,
        numeric=numeric,
        fractional=(),
        fraction_digits=0,
        categorical=tuple(name for name in FEATURES if name not in numeric),
        codebook={"sex": SEXES, **CODEBOOK, **held},
        descriptions=_DESCRIPTIONS,
        ranges=tuple(
            RangeConstraint(
                field=name,
                unit=unit,
                low=int(records[name].min()),
                high=int(records[name].max()),
                source="acs-nd",
                limits=(
                    "the least and greatest value over the loaded records; a value outside them "
                    "is unseen, not impossible"
                ),
            )
            for name, unit in _UNITS.items()
        ),
        label_codes=tuple(sorted(LABEL_CODES)),
        label_meanings={
            0: f"total person's income at most ${THRESHOLD:,}",
            1: f"total person's income above ${THRESHOLD:,}",
        },
        edges=(30, 45, 60),
        study="income above $50,000 among working adults",
        cohort=(
            "Massachusetts resident over 16 with income over $100 who usually worked in the "
            "past 12 months, from the ACS 2024 1-Year person file"
        ),
    )
