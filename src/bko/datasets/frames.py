"""What the stages read of an evaluation frame: one `FrameSpec` per dataset.

A frame's records have its `features` and its label, each a number. Generation, the baselines and
evaluation take a frame as a keyword defaulting to UCI's, `bko.datasets.uci.UCI_FRAME`, and
validation requires it, so another dataset runs through them without editing them. A cell is
always an age band, a sex and the label (`bko.datasets.cells`), so a frame holds only its band
edges. ADR `research/uci-evaluation-frame` records why. Beside the standard library it imports
`bko.datasets.cells` (`LABEL`, `LABEL_CODES`, `SEXES`), which loads pandas.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES


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

    A cell is always an age band, a sex and the label (`bko.datasets.cells`), so every frame has a
    numeric `age`, a categorical `sex` with `SEXES`' meanings and `LABEL` with `LABEL_CODES`; only
    the band edges vary (ADR `research/uci-evaluation-frame`). Frames compare by identity, and
    their mappings are read-only copies, so a stage can cache what it derives from one.

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
        study: What the records are synthesized for, as the system prompt's opening names it
            after "a study of".
        cohort: Who one record is, as the opening names it after "Each record is one".
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
    study: str
    cohort: str
    label: str = LABEL

    def __post_init__(self) -> None:
        """Snapshot the mappings read-only, then refuse values that contradict each other.

        The snapshot copies `codebook`, each of its code maps, `descriptions` and
        `label_meanings`, so a cache keyed by the frame cannot go stale.

        Raises:
            ValueError: If the frame breaks the cell machinery's fixed shape (the label is not
                `LABEL`, its codes are not `LABEL_CODES`, `age` is not numeric, `sex` is not
                categorical with `SEXES`' meanings) or its own consistency: the label is a
                feature; a numeric or categorical field is not a feature; a field is both, or a
                feature neither; a fractional field is not numeric; a categorical field has no
                codes; a numeric field has other than one range, or a range names another field
                or runs from above its high; a column has no description; or the edges are not
                strictly increasing.
        """
        codebook = {name: MappingProxyType(dict(codes)) for name, codes in self.codebook.items()}
        object.__setattr__(self, "codebook", MappingProxyType(codebook))
        object.__setattr__(self, "descriptions", MappingProxyType(dict(self.descriptions)))
        object.__setattr__(self, "label_meanings", MappingProxyType(dict(self.label_meanings)))
        self._check_shape()
        self._check_fields()
        self._check_ranges()
        for name in self.columns:
            if name not in self.descriptions:
                self._refuse(f"column {name} has no description")
        if not self.edges or any(low >= high for low, high in pairwise(self.edges)):
            self._refuse(f"edges {self.edges} are not strictly increasing")

    def _refuse(self, problem: str) -> None:
        msg = f"frame {self.name}: {problem}"
        raise ValueError(msg)

    def _check_shape(self) -> None:
        if self.label != LABEL:
            self._refuse(f"label {self.label} is not {LABEL}")
        if self.label in self.features:
            self._refuse(f"label {self.label} is listed as a feature")
        codes = tuple(sorted(LABEL_CODES))
        if self.label_codes != codes:
            self._refuse(f"label codes {self.label_codes} are not {codes}")

    def _check_fields(self) -> None:
        for kind, names in (("numeric", self.numeric), ("categorical", self.categorical)):
            for name in names:
                if name not in self.features:
                    self._refuse(f"{kind} field {name} is not a feature")
        for name in self.features:
            if name in self.numeric and name in self.categorical:
                self._refuse(f"field {name} is both numeric and categorical")
            if name not in self.numeric and name not in self.categorical:
                self._refuse(f"feature {name} is neither numeric nor categorical")
        if "age" not in self.numeric:
            self._refuse("cell field age is not numeric")
        if "sex" not in self.categorical:
            self._refuse("cell field sex is not categorical")
        for name in self.fractional:
            if name not in self.numeric:
                self._refuse(f"fractional field {name} is not numeric")
        for name in self.categorical:
            if name not in self.codebook:
                self._refuse(f"categorical field {name} has no codes")
        if dict(self.codebook["sex"]) != dict(SEXES):
            self._refuse(f"sex codes {dict(self.codebook['sex'])} are not {dict(SEXES)}")

    def _check_ranges(self) -> None:
        counts = Counter(bound.field for bound in self.ranges)
        for name in self.numeric:
            if counts[name] != 1:
                self._refuse(f"numeric field {name} has {counts[name]} ranges, not 1")
        for bound in self.ranges:
            if bound.field not in self.numeric:
                self._refuse(f"range field {bound.field} is not numeric")
            if bound.low > bound.high:
                self._refuse(f"range of {bound.field} runs from {bound.low:g} to {bound.high:g}")

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
