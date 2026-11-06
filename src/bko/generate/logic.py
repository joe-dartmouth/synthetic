"""Concept presence from logic alone: the assignments the `is_a` edges and nested pairs allow.

The logic-only draw reads no data prior (ADR `research/logic-only-ablation`). Its concepts form
components, joined wherever an `is_a` edge or a nested field pair says one concept's presence
implies another's. Each component draws one of its consistent assignments uniformly, the maximum
entropy choice given only those implications: a lone concept is present at 1/2, a pair at each of
its three allowed assignments at 1/3. A component above `MAX_COMPONENT` concepts refuses rather
than enumerate its assignments. `bko.generate.direct.sample` takes `Presence.LOGIC` to draw this
way; `Presence.SHARES` keeps its share draw.
"""

from __future__ import annotations

import enum
import functools
import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np


MAX_COMPONENT: Final = 12
"""The most concepts a component may hold: its 2 ** 12 assignments are enumerated."""


class Presence(enum.StrEnum):
    """How the direct draw decides each concept's presence."""

    SHARES = "shares"
    """At its share, read from the fused BKB's data priors (`bko.gaps.shares`)."""

    LOGIC = "logic"
    """Uniformly over its component's consistent assignments, with no prior."""


@dataclass(frozen=True, slots=True)
class Component:
    """Concepts joined by implications, drawn together.

    Attributes:
        concepts: The concepts, sorted.
        implies: `(concept, implied)` pairs within the component, sorted: a draw holding the first
            holds the second.
    """

    concepts: tuple[str, ...]
    implies: tuple[tuple[str, str], ...]


def components(
    concepts: Iterable[str], implies: Iterable[tuple[str, str]]
) -> tuple[Component, ...]:
    """The connected components of `concepts` under `implies`, ordered by their first concept.

    Args:
        concepts: The concepts drawn by logic.
        implies: `(concept, implied)` pairs; a pair naming a concept outside `concepts` is
            dropped.

    Raises:
        ValueError: If a component holds more than `MAX_COMPONENT` concepts.
    """
    members = sorted(set(concepts))
    parent = {concept: concept for concept in members}

    def root(concept: str) -> str:
        while parent[concept] != concept:
            concept = parent[concept]
        return concept

    edges = sorted({(a, b) for a, b in implies if a in parent and b in parent and a != b})
    for a, b in edges:
        parent[root(a)] = root(b)
    grouped: dict[str, list[str]] = {}
    for concept in members:
        grouped.setdefault(root(concept), []).append(concept)
    found: list[Component] = []
    for group in sorted(grouped.values()):
        if len(group) > MAX_COMPONENT:
            msg = (
                f"a component of {len(group)} concepts, {group[0]} first, is above "
                f"{MAX_COMPONENT}; its assignments are not enumerated"
            )
            raise ValueError(msg)
        inside = set(group)
        found.append(Component(tuple(group), tuple(e for e in edges if e[0] in inside)))
    return tuple(found)


def logic_assignment(component: Component, rng: np.random.Generator) -> frozenset[str]:
    """The concepts present in one assignment drawn uniformly from `component`'s consistent ones.

    One `rng.integers` call per component.
    """
    allowed = _consistent(component)
    return allowed[int(rng.integers(len(allowed)))]


@functools.cache
def _consistent(component: Component) -> tuple[frozenset[str], ...]:
    """Every assignment no implication contradicts, in `itertools.product` order."""
    allowed: list[frozenset[str]] = []
    for flags in itertools.product((False, True), repeat=len(component.concepts)):
        present = frozenset(c for c, flag in zip(component.concepts, flags, strict=True) if flag)
        if all(implied in present for a, implied in component.implies if a in present):
            allowed.append(present)
    return tuple(allowed)
