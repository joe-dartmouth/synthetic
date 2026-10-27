"""What the stages read of an evaluation frame.

A range constraint bounds one numeric field. Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RangeConstraint:
    """The plausible values of a numeric field, both bounds inclusive.

    Attributes:
        field: A numeric field of the frame.
        unit: The UCUM unit.
        low: The lowest plausible value.
        high: The highest plausible value.
        source: Where the bounds come from.
        limits: What the bounds do not guarantee.
    """

    field: str
    unit: str
    low: float
    high: float
    source: str
    limits: str
