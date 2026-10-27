"""What the stages read of an evaluation frame: one `FrameSpec` per dataset.

A frame's records have its `features` and its label, each a number. Validation, generation, the
baselines and evaluation take a frame as a keyword defaulting to UCI's,
`bko.datasets.uci.UCI_FRAME`, so another dataset runs through them without editing them. A cell is
always an age band, a sex and the label (`bko.datasets.cells`), so a frame holds only its band
edges. ADR `research/uci-evaluation-frame` records why. Standard library only, but for `LABEL`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from bko.datasets.cells import LABEL


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


@dataclass(frozen=True, slots=True, eq=False)
class FrameSpec:
    """A dataset's columns, codes, ranges, precision, label and age band edges.

    Frames compare by identity, so a stage can cache what it derives from one.

    Attributes:
        name: The frame's name, as reports write it.
        features: The predictors, in column order.
        numeric: The measured features; each has exactly one range.
        fractional: The numeric features that may hold a fraction; every other field is whole.
        fraction_digits: The decimal places a fractional value has, which added values keep.
        categorical: The coded features; each one's codes are in `codebook`.
        codebook: Each coded feature's codes and meanings.
        descriptions: What each column records, the label included.
        ranges: One plausible range per numeric feature.
        label_codes: The label's codes, ascending.
        label_meanings: Each label code's meaning.
        edges: The age band starts (`bko.datasets.cells.age_band_labels`).
        label: The binary label column.
    """

    name: str
    features: tuple[str, ...]
    numeric: tuple[str, ...]
    fractional: tuple[str, ...]
    fraction_digits: int
    categorical: tuple[str, ...]
    codebook: Mapping[str, Mapping[int, str]]
    descriptions: Mapping[str, str]
    ranges: tuple[RangeConstraint, ...]
    label_codes: tuple[int, ...]
    label_meanings: Mapping[int, str]
    edges: tuple[int, ...]
    label: str = LABEL

    def __post_init__(self) -> None:
        """Refuse values that contradict each other.

        Raises:
            ValueError: If a numeric field is not a feature, a fractional field is not numeric, a
                numeric field has other than one range, a range names a field that is not
                numeric, or a column has no description.
        """
        for name in self.numeric:
            if name not in self.features:
                msg = f"frame {self.name}: numeric field {name} is not a feature"
                raise ValueError(msg)
        for name in self.fractional:
            if name not in self.numeric:
                msg = f"frame {self.name}: fractional field {name} is not numeric"
                raise ValueError(msg)
        counts = Counter(bound.field for bound in self.ranges)
        for name in self.numeric:
            if counts[name] != 1:
                msg = f"frame {self.name}: numeric field {name} has {counts[name]} ranges, not 1"
                raise ValueError(msg)
        for name in counts:
            if name not in self.numeric:
                msg = f"frame {self.name}: range field {name} is not numeric"
                raise ValueError(msg)
        for name in self.columns:
            if name not in self.descriptions:
                msg = f"frame {self.name}: column {name} has no description"
                raise ValueError(msg)

    @property
    def columns(self) -> tuple[str, ...]:
        """The features, then the label: the columns of a training split and of added rows."""
        return (*self.features, self.label)

    @property
    def dtypes(self) -> Mapping[str, str]:
        """Each column's dtype: fractional fields as float64, every other column as int64."""
        return MappingProxyType(
            {name: "float64" if name in self.fractional else "int64" for name in self.columns}
        )
