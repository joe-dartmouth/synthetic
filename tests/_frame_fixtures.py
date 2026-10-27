"""A toy evaluation frame: three features, other codes, ranges, decimals and band edges than UCI's.

It proves a stage reads the frame it is given rather than UCI's constants.
"""

from __future__ import annotations

from bko.datasets.frames import FrameSpec, RangeConstraint


def _range(field: str, unit: str, low: float, high: float) -> RangeConstraint:
    return RangeConstraint(field=field, unit=unit, low=low, high=high, source="toy", limits="toy")


TOY = FrameSpec(
    name="toy",
    features=("age", "sex", "level"),
    numeric=("age", "level"),
    fractional=("level",),
    fraction_digits=2,
    categorical=("sex",),
    codebook={"sex": {0: "woman", 1: "man"}},
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
)
"""The toy frame."""
