"""Generate NHAMCS's ICD-10-CM diagnosis rows from a MONDO release.

`python -m bko.ontology codes --system icd10cm` runs `generate_icd10` over every diagnosis token of
the `nhamcs` records and writes `render_icd10`'s module text, committed as
`bko.ontology.icd10cm_codes`. A token reads as an ICD-10-CM id by restoring its implied decimal
(`icd10cm_id`): `I509` is `ICD10CM:I50.9` and a dash-filled `A00-` the category `ICD10CM:A00`;
blank and NCHS's own `ZZZ` tokens read as none. The id maps as an ICD-9 code does in
`bko.ontology.codes`: to the live concept carrying it under `source="MONDO:equivalentTo"`, else the
one such holder every other descends from by `is_a`, else nothing, with no curated rows. ADR
`research/nhamcs-frame` records the rule and the unmapped counts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final

from bko.datasets.icd10_tokens import BLANK, NOT_ICD, is_token, token_number
from bko.ontology.codes import _holders, _name, _spanning, _unmapped, render_rows
from bko.ontology.terms import Ontology


PREFIX: Final = "ICD10CM:"
"""The prefix of MONDO's ICD-10-CM cross-references."""

REASONS: Final = (
    "no live xref",
    "obsolete holder only",
    "relatedTo only",
    "unreviewed source only",
    "holders not nested",
    "not ICD-10-CM",
)
"""Why a token stays unmapped, in the order `generate_icd10` counts them: the first five as
`bko.ontology.codes.REASONS` reads them, then NCHS's own `ZZZ` tokens."""


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


def icd10cm_id(token: str) -> str | None:
    """The `ICD10CM:` id `token` reads as; None for a blank or NCHS-only token.

    Raises:
        ValueError: If `token` is not a diagnosis token.
    """
    if not is_token(token):
        msg = f"{token!r} is not an NHAMCS diagnosis token"
        raise ValueError(msg)
    if token == BLANK or token in NOT_ICD:
        return None
    head, tail = token[:3], token[3]
    return f"{PREFIX}{head}" if tail == "-" else f"{PREFIX}{head}.{tail}"


def generate_icd10(
    mondo: Ontology, tokens: Collection[str]
) -> tuple[list[Icd10Code], Counter[str]]:
    """Map each distinct token among `tokens` to one MONDO concept, or count why it stays unmapped.

    Args:
        mondo: The MONDO release.
        tokens: Diagnosis tokens as the file writes them; `BLANK` is skipped.

    Returns:
        One row per mapped token, in `token_number` order, and each of `REASONS` with how many
        distinct tokens it leaves unmapped.

    Raises:
        ValueError: If a token is not a diagnosis token, or a chosen concept has no name.
    """
    exact, weaker, dead = _holders(mondo, PREFIX)
    rows: list[Icd10Code] = []
    unmapped: Counter[str] = Counter(dict.fromkeys(REASONS, 0))
    for token in sorted({t for t in tokens if t != BLANK}, key=token_number):
        xref = icd10cm_id(token)
        if xref is None:
            unmapped["not ICD-10-CM"] += 1
            continue
        code = xref.removeprefix(PREFIX)
        holders = exact.get(code, ())
        if not holders:
            unmapped[_unmapped(code, weaker, dead)] += 1
            continue
        concept = _spanning(mondo, holders)
        if concept is None:
            unmapped["holders not nested"] += 1
            continue
        rows.append(Icd10Code(code, concept, _name(mondo, concept), frozenset(holders)))
    return rows, unmapped


_HEADER: Final = '''"""NHAMCS's ICD-10-CM diagnosis rows, generated from MONDO.

`python -m bko.ontology codes --system icd10cm` writes this module; regenerate it rather than
editing it. Each row maps the ICD-10-CM id one diagnosis token of the `nhamcs` records reads as to
the MONDO concept that carries its `ICD10CM:` xref under `source="MONDO:equivalentTo"` (ADR
`research/nhamcs-frame`). `_ROWS` holds one (code, concept, label) tuple per line, and `_ALSO` the
other live holders of such an xref for a code that has more than one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from bko.ontology.icd10 import Icd10Code


'''

_FOOTER: Final = '''
ICD10CM_CODES: Final[tuple[Icd10Code, ...]] = tuple(
    Icd10Code(code, concept, label, frozenset({concept, *_ALSO.get(code, ())}))
    for code, concept, label in _ROWS
)
"""Each mapped id, in token order, with its concept and every live exact holder of its xref."""
'''


def render_icd10(rows: Sequence[Icd10Code]) -> str:
    """The text of `bko.ontology.icd10cm_codes` holding `rows`, as `ruff format` leaves it.

    Raises:
        ValueError: If a row's code, concept or label holds a control character, naming its code.
    """
    return render_rows(rows, header=_HEADER, footer=_FOOTER, system="ICD-10-CM")
