"""The row types of the concept-to-field mapping table in `bko.ontology.mapping`.

A field mapping means one thing: when its clauses all hold for a record, the record shows its
concept (SSSOM `skos:broadMatch`, with the dataset state as subject). No row says a concept is
absent. `RangeConstraint` is defined with the frame it bounds, in `bko.datasets.frames`, and
re-exported here. That import loads `bko.datasets.cells`, and so pandas; nothing else is imported
but the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bko.datasets.frames import RangeConstraint  # ruff: ignore[unused-import]  (re-exported)


@dataclass(frozen=True, slots=True)
class Codes:
    """A categorical field holding one of some codes.

    Attributes:
        field: A categorical UCI field, or the label.
        codes: A non-empty subset of the field's codebook.
    """

    field: str
    codes: frozenset[int]

    def holds(self, value: float) -> bool:
        """Return whether `value` is one of the codes."""
        return value in self.codes


@dataclass(frozen=True, slots=True)
class AtLeast:
    """A numeric field at or above a limit.

    Attributes:
        field: A numeric UCI field.
        value: The inclusive limit, inside the field's range constraint.
    """

    field: str
    value: float

    def holds(self, value: float) -> bool:
        """Return whether `value` is at or above the limit."""
        return value >= self.value


type Clause = Codes | AtLeast


@dataclass(frozen=True, slots=True)
class FieldMapping:
    """Field states that show a concept.

    Attributes:
        when: Clauses that must all hold, each on a different field.
        concept: The MONDO or HP curie.
        label: The concept's name in its pinned release, verbatim.
        source: What establishes the mapping, from the vocabulary `bko.ontology.mapping` states.
        limits: Where the mapping stops being true.
    """

    when: tuple[Clause, ...]
    concept: str
    label: str
    source: str
    limits: str

    def applies(self, record: Mapping[str, float]) -> bool:
        """Return whether every clause holds for `record`, a value per field name."""
        return all(clause.holds(record[clause.field]) for clause in self.when)


@dataclass(frozen=True, slots=True)
class Unmapped:
    """Categorical field states no concept defensibly matches.

    Attributes:
        field: A categorical UCI field, or the label.
        codes: The codes left unmapped.
        reason: Why no concept matches.
        source: What supports the reason.
    """

    field: str
    codes: frozenset[int]
    reason: str
    source: str


@dataclass(frozen=True, slots=True)
class CodeMapping:
    """A SNOMED-CT code, as Synthea writes it, and the MONDO concept the release equates with it.

    Attributes:
        snomed: The code.
        concept: The MONDO curie.
        label: The concept's name in its pinned release, verbatim.
        source: The release that asserts the equivalence.
    """

    snomed: str
    concept: str
    label: str
    source: str


@dataclass(frozen=True, slots=True)
class OutOfScope:
    """A concept considered and rejected: no field grounds it defensibly.

    Attributes:
        concept: The MONDO or HP curie.
        label: The concept's name in its pinned release, verbatim.
        reason: Why no field grounds it.
        source: What supports the reason.
    """

    concept: str
    label: str
    reason: str
    source: str


type ConceptRow = FieldMapping | CodeMapping | OutOfScope
