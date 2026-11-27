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

SOC_ID: Final = "OCCO:00000008"
"""The Occupation Ontology's literal annotation that holds a class's US SOC id, such as
`11-1011.00`."""


@dataclass(frozen=True, slots=True)
class Term:
    """One term stanza.

    Attributes:
        curie: The term's id, such as `MONDO:0024644`.
        name: Its `name:` value; None when the stanza has none.
        parents: Its direct `is_a` targets.
        xrefs: Its xref ids.
        matches: `(predicate, object)` for each `property_value` whose predicate is in
            `MATCH_PREDICATES`.
        obsolete: Whether the stanza says `is_obsolete: true`.
        replaced_by: Its `replaced_by` targets, which need not be terms of the release.
        xref_qualifiers: `(xref id, ((key, value), ...))` for each xref the stanza qualifies, such
            as `("ICD9:250", (("source", "MONDO:equivalentTo"),))`; an xref without qualifiers
            is absent.
        annotations: `(predicate, literal)` for each literal `property_value` whose predicate the
            loader keeps (`bko.ontology.obo.ANNOTATION_PREDICATES`), such as
            `("OCCO:00000008", "11-1011.00")`.
        synonyms: `(scope, text)` for each `synonym:` line, such as `("EXACT", "hypertension")`.
    """

    curie: str
    name: str | None
    parents: tuple[str, ...]
    xrefs: tuple[str, ...]
    matches: tuple[tuple[str, str], ...]
    obsolete: bool
    replaced_by: tuple[str, ...]
    xref_qualifiers: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    annotations: tuple[tuple[str, str], ...] = ()
    synonyms: tuple[tuple[str, str], ...] = ()

    def xref_sources(self, xref: str) -> tuple[str, ...]:
        """The `source` qualifier values the stanza gives `xref`, sorted; empty if none."""
        for held, qualifiers in self.xref_qualifiers:
            if held == xref:
                return tuple(value for key, value in qualifiers if key == "source")
        return ()

    def exact_synonyms(self) -> tuple[str, ...]:
        """The texts of the stanza's EXACT synonyms, sorted; empty if none."""
        return tuple(text for scope, text in self.synonyms if scope == "EXACT")

    def annotation(self, predicate: str) -> tuple[str, ...]:
        """The literals the stanza gives `predicate`, sorted; empty if none."""
        return tuple(value for key, value in self.annotations if key == predicate)


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
