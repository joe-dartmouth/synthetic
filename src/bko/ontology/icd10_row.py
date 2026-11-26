"""NHAMCS's generated ICD-10-CM row type and the prefix of the xrefs it rests on.

`bko.ontology.icd10` generates the rows, committed as `bko.ontology.icd10cm_codes`. This module
imports only the standard library, so `bko.ontology.schema` can name the row type without loading
the generator. ADR `research/nhamcs-frame` records the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


PREFIX: Final = "ICD10CM:"
"""The prefix of MONDO's ICD-10-CM cross-references."""


@dataclass(frozen=True, slots=True)
class Icd10Code:
    """An ICD-10-CM id an NHAMCS token reads as and the one MONDO concept equated with it.

    Attributes:
        code: The id as MONDO's xrefs write it, such as `I50.9` or the category `A00`.
        concept: The MONDO curie.
        label: The concept's name in its pinned release, verbatim.
        holders: Every live concept in the pinned release that carries `xref` under `EXACT`,
            `concept` among them.
    """

    code: str
    concept: str
    label: str
    holders: frozenset[str]

    @property
    def xref(self) -> str:
        """The `ICD10CM:` xref id the concept carries."""
        return f"{PREFIX}{self.code}"

    @property
    def token(self) -> str:
        """The four-character token the ED file writes for `code`."""
        return self.code.replace(".", "").ljust(4, "-")
