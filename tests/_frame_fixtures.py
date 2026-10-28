"""A toy evaluation frame: three features, other codes, ranges, decimals and band edges than UCI's.

It proves a stage reads the frame it is given rather than UCI's constants.
"""

from __future__ import annotations

import pandas as pd

from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.datasets.split import split


def _range(field: str, unit: str, low: float, high: float) -> RangeConstraint:
    return RangeConstraint(field=field, unit=unit, low=low, high=high, source="toy", limits="toy")


TOY = FrameSpec(
    name="toy",
    features=("age", "sex", "level"),
    numeric=("age", "level"),
    fractional=("level",),
    fraction_digits=2,
    categorical=("sex",),
    codebook={"sex": {0: "female", 1: "male"}},
    descriptions={
        "age": "age at the visit",
        "sex": "recorded sex",
        "level": "toy marker level",
        "target": "whether the toy condition is present",
    },
    ranges=(_range("age", "a", 30, 90), _range("level", "mg/dL", 0.5, 4.5)),
    label_codes=(0, 1),
    label_meanings={0: "toy condition absent", 1: "toy condition present"},
    edges=(45, 70),
    study="toy-condition prediction",
    cohort="adult seen at a toy clinic",
)
"""The toy frame."""


def toy_records() -> pd.DataFrame:
    """25 records per toy cell, but 6 in each `45-69` cell with the condition."""
    rows: list[dict[str, int | float]] = []
    for age in (35, 50, 75):
        for sex in (0, 1):
            for target in (0, 1):
                size = 6 if (age, target) == (50, 1) else 25
                for i in range(size):
                    level = round(0.5 + ((i * 37 + age + 11 * sex + 5 * target) % 400) / 100, 2)
                    rows.append({"age": age + i % 9, "sex": sex, "level": level, "target": target})
    frame = pd.DataFrame(rows).astype(dict(TOY.dtypes))
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="record_id")
    return frame


TOY_PARTS = split(toy_records(), seed=0, edges=TOY.edges)
"""The toy records' split at seed 0, under the toy edges."""

TOY_TRAIN = TOY_PARTS.train
"""Its training records."""

SHORT = (("45-69", "female", 1), ("45-69", "male", 1))
"""The toy cells under 20 training records, in grid order."""
