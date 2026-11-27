"""Automatic mapping rows that `python -m bko.ontology.automatic lexical` wrote.

Regenerate this module rather than editing it (ADR `research/automatic-mapping`). `_ROWS` holds
one (field, code, concept, label, limits) tuple per line.
"""

from __future__ import annotations

from typing import Final

from bko.ontology.schema import Codes, FieldMapping


# fmt: off
_ROWS: Final[tuple[tuple[str, int, str, str, str], ...]] = (
)
# fmt: on

ROWS: Final[tuple[FieldMapping, ...]] = tuple(
    FieldMapping(
        when=(Codes(field, frozenset({code})),),
        concept=concept,
        label=label,
        source="research/automatic-mapping",
        limits=limits,
    )
    for field, code, concept, label, limits in _ROWS
)
"""The rows, sorted by (field, code, concept)."""
