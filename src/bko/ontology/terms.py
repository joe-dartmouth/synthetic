"""An ontology release as standard-library values: what every stage reads instead of pronto.

Every tuple is sorted, so iteration order never depends on the parser's threads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final


MATCH_PREDICATES: Final = frozenset(
    {
        "skos:exactMatch",
        "skos:closeMatch",
        "skos:broadMatch",
        "skos:narrowMatch",
        "skos:relatedMatch",
    }
)
"""The SKOS mapping properties a term's `property_value` lines may carry."""


@dataclass(frozen=True, slots=True)
class Term:
    """One term stanza.

    Attributes:
        curie: The term's id, such as `MONDO:0024644`.
        name: Its `name:` value; None when the stanza has none.
        parents: Its direct `is_a` targets.
        xrefs: Its xref ids; the release's qualifiers are not kept.
        matches: `(predicate, object)` for each `property_value` whose predicate is in
            `MATCH_PREDICATES`.
        obsolete: Whether the stanza says `is_obsolete: true`.
        replaced_by: Its `replaced_by` targets, which need not be terms of the release.
    """

    curie: str
    name: str | None
    parents: tuple[str, ...]
    xrefs: tuple[str, ...]
    matches: tuple[tuple[str, str], ...]
    obsolete: bool
    replaced_by: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Ontology:
    """One parsed release.

    Attributes:
        name: The pinned `Source`'s name, such as `mondo`.
        version: The header's `data-version`, verbatim.
        terms: Every term stanza by curie, in sorted order, obsolete and imported terms included.
    """

    name: str
    version: str
    terms: Mapping[str, Term]


def ancestors(ontology: Ontology, curie: str, *, prefix: str | None = None) -> frozenset[str]:
    """The classes `curie` reaches by following `is_a` upward in `ontology`.

    Args:
        ontology: The release.
        curie: The class to start from.
        prefix: When given, only parents with this id prefix are followed, so a branch that leaves
            the release's own classes ends there.

    Returns:
        Every class reached; a parent the release lacks is included and ends its branch.
    """
    seen: set[str] = set()
    pending = [curie]
    while pending:
        term = ontology.terms.get(pending.pop())
        for parent in term.parents if term is not None else ():
            if parent not in seen and (prefix is None or parent.startswith(f"{prefix}:")):
                seen.add(parent)
                pending.append(parent)
    return frozenset(seen)
