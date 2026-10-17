"""Loaded ontologies as BKO fragments: the BKB each release contributes to a fusion.

Each class in scope is one component whose one state is `INDIVIDUAL`, so the I-node
`(curie, INDIVIDUAL)` reads "a is a member of curie", as in the example of Santos, Jurmain and
Ragazzi (`santos-2024`, section 7). A direct `is_a` edge whose two ends are in scope becomes the
S-node P(parent | child) = 1, the PTA a subsumption axiom is; a `skos:exactMatch` one release
declares between scoped classes of two releases becomes one such S-node in each direction. The
caller fixes the scope, so an edge to a class outside it is dropped, never shortened. ADR
`research/bko-fragment-encoding` records the encoding and its limits.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Final

from bko.bkb.structure import BKB
from bko.ontology.errors import ScopeError
from bko.ontology.terms import Ontology, ancestors


INDIVIDUAL: Final = "a"
"""The one state of every class component: the individual a fragment's statements are about."""

EXACT_MATCH: Final = "skos:exactMatch"
"""The one mapping predicate a bridge reads."""


def is_a_fragment(
    ontology: Ontology, scope: Collection[str], *, prefix: str, name: str | None = None
) -> BKB:
    """The direct `is_a` edges of `ontology` within `scope`, as a BKB fragment.

    Args:
        ontology: The loaded release.
        scope: The classes the fragment holds.
        prefix: The id prefix of the release's own classes, such as `MONDO`; MONDO carries copies
            of HP terms, and the prefix keeps another release's classes out.
        name: The fragment's name, which `fuse` takes as its source name; `ontology.name` when
            omitted.

    Returns:
        One I-node per scoped class, in sorted order, and one S-node P(parent | child) = 1 per
        direct `is_a` edge with both ends in scope, ordered by child and then parent.

    Raises:
        ScopeError: If `scope` is empty or names a class without the prefix, absent from the
            release or obsolete in it.
    """
    classes = _scoped(ontology, scope, prefix)
    fragment = BKB(ontology.name if name is None else name)
    for curie in classes:
        fragment.add_inode(curie, INDIVIDUAL)
    members = set(classes)
    for curie in classes:
        for parent in ontology.terms[curie].parents:
            if parent in members:
                fragment.add_snode(parent, INDIVIDUAL, 1.0, [(curie, INDIVIDUAL)])
    return fragment


def exact_match_fragment(
    subject: Ontology,
    subject_scope: Collection[str],
    target: Ontology,
    target_scope: Collection[str],
    *,
    subject_prefix: str,
    target_prefix: str,
    name: str = "bridge",
) -> BKB:
    """The `skos:exactMatch` links `subject` declares to `target`, within the scopes, as a fragment.

    Only `subject`'s declarations are read. Each link becomes P(target class | subject class) = 1
    and P(subject class | target class) = 1.

    Returns:
        One I-node per scoped class, `subject`'s and then `target`'s, each in sorted order, and two
        S-nodes per link, ordered by subject class and then target class.

    Raises:
        ScopeError: If the two prefixes are equal, or as `is_a_fragment` for either scope.
    """
    if subject_prefix == target_prefix:
        msg = f"a bridge links two releases; both scopes have the prefix {subject_prefix}"
        raise ScopeError(msg)
    left = _scoped(subject, subject_scope, subject_prefix)
    right = _scoped(target, target_scope, target_prefix)
    fragment = BKB(name)
    for curie in (*left, *right):
        fragment.add_inode(curie, INDIVIDUAL)
    members = set(right)
    for curie in left:
        for predicate, match in subject.terms[curie].matches:
            if predicate == EXACT_MATCH and match in members:
                fragment.add_snode(match, INDIVIDUAL, 1.0, [(curie, INDIVIDUAL)])
                fragment.add_snode(curie, INDIVIDUAL, 1.0, [(match, INDIVIDUAL)])
    return fragment


def scope(ontology: Ontology, concepts: Collection[str], *, prefix: str) -> frozenset[str]:
    """The classes a fragment of `ontology` holds for `concepts`, by the rule of the mapping table.

    Only the release's own classes are kept, and only through them: an ancestor outside the prefix,
    such as MONDO's BFO roots, ends its branch, so no class above it is cut off from the rest.

    Returns:
        The concepts with `prefix` and their ancestors with it; empty when no concept has it.
    """
    own = [curie for curie in concepts if curie.startswith(f"{prefix}:")]
    return frozenset(own).union(*(ancestors(ontology, curie, prefix=prefix) for curie in own))


def _scoped(ontology: Ontology, scope: Collection[str], prefix: str) -> list[str]:
    classes = sorted(set(scope))
    if not classes:
        msg = f"the scope for {ontology.name} {ontology.version} is empty"
        raise ScopeError(msg)
    for curie in classes:
        term = ontology.terms.get(curie)
        if not curie.startswith(f"{prefix}:"):
            problem = f"is not a {prefix} term"
        elif term is None:
            problem = "is absent"
        elif term.obsolete:
            problem = "is obsolete"
        else:
            continue
        msg = f"{curie} {problem} in {ontology.name} {ontology.version}"
        raise ScopeError(msg)
    return classes
