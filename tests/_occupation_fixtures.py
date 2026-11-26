"""Offline Occupation Ontology text: a few OccO classes as ROBOT writes them in OBO.

The real data here is the chief executive stanza's `id`, `name`, `is_a` and `property_value`
lines, byte for byte from the release's conversion; the other stanzas keep the real ids, names,
parents and SOC ids, with their other lines left out.
"""

from __future__ import annotations

from itertools import starmap
from typing import Final


OCCO_HEADER: Final = "format-version: 1.2\ndata-version: releases/2025-10-06\nontology: occo\n"
"""The header ROBOT writes for the pinned release, its `data-version` included."""

CHIEF_EXECUTIVE: Final = (
    "[Term]\n"
    "id: OCCO:11101100\n"
    "name: chief executive\n"
    "is_a: OCCO:00000007 ! occupation holder with job zone 5\n"
    "is_a: OCCO:11101000 ! chief executive (broad)\n"
    'property_value: IAO:0000117 "Oliver He" xsd:string\n'
    'property_value: IAO:0000118 "Chief Executives" xsd:string\n'
    'property_value: OCCO:00000008 "11-1011.00" xsd:string\n'
)
"""The real chief executive stanza, its `relationship`, `def` and other annotation lines left
out: one literal `OCCO:00000008` beside literals of predicates no term keeps."""


def _stanza(curie: str, name: str, parents: tuple[str, ...], soc: str | None) -> str:
    lines = [f"[Term]\nid: {curie}\nname: {name}\n", *(f"is_a: {p}\n" for p in parents)]
    if soc is not None:
        lines.append(f'property_value: OCCO:00000008 "{soc}" xsd:string\n')
    return "".join(lines)


CLASSES: Final = (
    ("OCCO:00000001", "occupation holder", (), None),
    ("OCCO:00000007", "occupation holder with job zone 5", ("OCCO:00000001",), None),
    ("OCCO:00000055", "occupation holder with job zone 4", ("OCCO:00000001",), None),
    ("OCCO:11000000", "management occupation", ("OCCO:00000001",), "11-0000"),
    ("OCCO:11100000", "top executive", ("OCCO:11000000",), "11-1000"),
    ("OCCO:11101000", "chief executive (broad)", ("OCCO:11100000",), "11-1010"),
    ("OCCO:11300000", "operations specialties manager", ("OCCO:11000000",), "11-3000"),
    ("OCCO:11303000", "financial manager (broad)", ("OCCO:11300000",), "11-3030"),
    ("OCCO:11303100", "financial manager", ("OCCO:00000055", "OCCO:11303000"), "11-3031"),
    ("OCCO:11303101", "treasurers or controller", ("OCCO:00000055", "OCCO:11303000"), "11-3031.01"),
    ("OCCO:47200000", "construction trades worker", ("OCCO:00000001",), "47-2000"),
    (
        "OCCO:47204000",
        "carpet, floor, or tile installer or finisher",
        ("OCCO:47200000",),
        "47-2040",
    ),
    ("OCCO:15124000", "database or network administrator or architect", (), "15-1240"),
    ("OCCO:15200000", "mathematical science occupation", (), "15-2000"),
)
"""Each further class as `(curie, name, parents, SOC id)`, the release's own values."""

OCCO_TEXT: Final = "\n".join((OCCO_HEADER, CHIEF_EXECUTIVE, *starmap(_stanza, CLASSES)))
"""A small OccO release in OBO: the header, the chief executive stanza and `CLASSES`."""
