"""Ancestor-shrunk shares: a coded field's values as the leaves of an ontology tree.

A tree field is a column whose rows each map one set of its codes and partition them
(`tree_fields`). `code_tree` builds its tree from the mapped concepts' ancestors in their releases,
one parent per node, under a virtual top that also holds the `unmapped` leaf. `tree_shares` reads
each node's share as its parent's share times a conditional: the cell's split of the parent shrunk
toward the label group's. ADR `research/ancestor-shrunk-shares` records the rule; the ontologies
are passed in, so this module loads no release. `column_trees` holds each tree field's tree and
the leaf each of its codes shows, which the direct sampler (`bko.generate.direct`) draws from.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from bko.datasets.frames import FrameSpec
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology, ancestors


TOP: Final = "top"
"""The virtual root every tree hangs from."""

UNMAPPED: Final = "unmapped"
"""The leaf under the top that holds every value no row maps."""

SELF: Final = "/self"
"""The suffix of a mapped concept's own leaf when the concept is an ancestor of another."""

_MIN_ROWS: Final = 2


@dataclass(frozen=True, slots=True)
class TreePart:
    """One node's share and what it is read from.

    Attributes:
        node: The node.
        parent: Its parent.
        cell: The cell's records at the node.
        parent_cell: The cell's records at the parent.
        group: The label group's records at the node.
        parent_group: The label group's records at the parent.
        weight: The label group's weight, in records.
        conditional: The node's share of its parent's.
        share: The node's share.
    """

    node: str
    parent: str
    cell: int
    parent_cell: int
    group: int
    parent_group: int
    weight: int
    conditional: float
    share: float


@dataclass(frozen=True, slots=True)
class ColumnTree:
    """One tree field's tree and the leaf each mapped code shows.

    Attributes:
        column: The tree field.
        tree: Each node with its parent (`code_tree`).
        codes: Each mapped code's leaf; any other value shows `UNMAPPED`.
    """

    column: str
    tree: Mapping[str, str]
    codes: Mapping[int, str]

    def leaf(self, record: Mapping[str, float]) -> str:
        """The leaf `record`'s value of the column shows."""
        return self.leaf_of(int(record[self.column]))

    def leaf_of(self, code: int) -> str:
        """The leaf `code` shows in the column."""
        return self.codes.get(code, UNMAPPED)

    def counts(self, codes: Iterable[int]) -> Counter[str]:
        """How many of `codes` show each leaf."""
        return Counter(self.leaf_of(code) for code in codes)


def column_trees(
    rows: Sequence[FieldMapping], frame: FrameSpec, ontologies: Mapping[str, Ontology]
) -> dict[str, ColumnTree]:
    """Each tree field of `rows` on `frame` with its tree over the concepts its rows map.

    A concept that is an ancestor of another shows as its `<concept>/self` leaf.
    """
    trees = {}
    for column in tree_fields(rows, frame):
        mapped = [row for row in rows if row.when[0].field == column]
        tree = code_tree({row.concept for row in mapped}, ontologies)
        codes = {
            code: f"{row.concept}{SELF}" if f"{row.concept}{SELF}" in tree else row.concept
            for row in mapped
            if isinstance(row.when[0], Codes)
            for code in row.when[0].codes
        }
        trees[column] = ColumnTree(column, tree, codes)
    return trees


def tree_fields(rows: Sequence[FieldMapping], frame: FrameSpec) -> tuple[str, ...]:
    """The columns of `frame` whose rows partition some of their codes, in column order.

    A column is a tree field when it is not `age`, `sex` or the label, at least two rows read it,
    every row reading it is one `Codes` clause on it alone, and their code sets are pairwise
    disjoint.
    """
    reading: dict[str, list[FieldMapping]] = {}
    for row in rows:
        for clause in row.when:
            reading.setdefault(clause.field, []).append(row)
    found = []
    for column in frame.columns:
        mapped = reading.get(column, [])
        if column in {"age", "sex", frame.label} or len(mapped) < _MIN_ROWS:
            continue
        if not all(len(row.when) == 1 and isinstance(row.when[0], Codes) for row in mapped):
            continue
        sets = [row.when[0].codes for row in mapped if isinstance(row.when[0], Codes)]
        if sum(map(len, sets)) == len(frozenset().union(*sets)):
            found.append(column)
    return tuple(found)


def code_tree(concepts: Collection[str], ontologies: Mapping[str, Ontology]) -> Mapping[str, str]:
    """Each node of the tree over `concepts` with its parent, leaves included.

    The nodes are the concepts and their ancestors within each concept's release, those on some
    concept's path to `TOP`. A node keeps the parent with the most concepts at or below it, ties
    to the lowest CURIE; a node with no parent in the closure, a concept its release lacks
    included, hangs from `TOP`. A concept that is an ancestor of another gets the leaf
    `<concept>/self` beneath itself. `UNMAPPED` hangs from `TOP`.
    """
    mapped = sorted(set(concepts))
    above = {concept: _ancestors(concept, ontologies) for concept in mapped}
    below = Counter(node for concept in mapped for node in above[concept] | {concept})
    tree: dict[str, str] = {}
    for concept in mapped:
        node = concept
        while node not in tree:
            parents = [p for p in _parents(node, ontologies) if p in below]
            parent = min(parents, key=lambda p: (-below[p], p)) if parents else TOP
            tree[node] = parent
            if parent == TOP:
                break
            node = parent
    inner = set(tree.values())
    for concept in mapped:
        if concept in inner or any(concept in above[other] for other in mapped):
            tree[f"{concept}{SELF}"] = concept
    tree[UNMAPPED] = TOP
    return dict(sorted(tree.items()))


def leaves(tree: Mapping[str, str]) -> tuple[str, ...]:
    """The tree's leaves in depth-first order from `TOP`, children by id."""
    children = _children(tree)
    found: list[str] = []
    pending = [TOP]
    while pending:
        node = pending.pop()
        below = children.get(node, [])
        if not below and node != TOP:
            found.append(node)
        pending += reversed(below)
    return tuple(found)


def leaf_concept(leaf: str) -> str | None:
    """The concept a leaf stands for; None for `UNMAPPED`."""
    if leaf == UNMAPPED:
        return None
    return leaf.removesuffix(SELF)


def tree_shares(
    cell: Mapping[str, int], group: Mapping[str, int], tree: Mapping[str, str], weight: int
) -> tuple[dict[str, float], tuple[TreePart, ...]]:
    """Each node's share, `TOP`'s 1.0 included, and each node's part in depth-first order.

    `cell` and `group` count records per leaf; a leaf they omit counts 0, and a node counts its
    leaves' sum. A child `v` of `u` takes `(n_c(v) + w n_g(v) / n_g(u)) / (n_c(u) + w)` of `u`'s
    share, the group term 0 where `n_g(u)` is 0, computed exactly and rounded once.

    Raises:
        ValueError: If `weight` is below 1.
    """
    if weight < 1:
        msg = f"weight {weight} must be at least 1"
        raise ValueError(msg)
    children = _children(tree)
    n_c, n_g = _totals(cell, children), _totals(group, children)
    exact: dict[str, Fraction] = {TOP: Fraction(1)}
    parts: list[TreePart] = []
    for node in _preorder(children)[1:]:
        parent = tree[node]
        borrowed = Fraction(weight * n_g[node], n_g[parent]) if n_g[parent] else Fraction(0)
        conditional = (n_c[node] + borrowed) / (n_c[parent] + weight)
        exact[node] = exact[parent] * conditional
        parts.append(
            TreePart(
                node=node,
                parent=parent,
                cell=n_c[node],
                parent_cell=n_c[parent],
                group=n_g[node],
                parent_group=n_g[parent],
                weight=weight,
                conditional=float(conditional),
                share=float(exact[node]),
            )
        )
    return {node: float(share) for node, share in exact.items()}, tuple(parts)


def _ancestors(concept: str, ontologies: Mapping[str, Ontology]) -> frozenset[str]:
    prefix = concept.partition(":")[0]
    ontology = ontologies.get(prefix)
    return frozenset() if ontology is None else ancestors(ontology, concept, prefix=prefix)


def _parents(node: str, ontologies: Mapping[str, Ontology]) -> tuple[str, ...]:
    prefix = node.partition(":")[0]
    ontology = ontologies.get(prefix)
    term = None if ontology is None else ontology.terms.get(node)
    if term is None:
        return ()
    return tuple(parent for parent in term.parents if parent.startswith(f"{prefix}:"))


def _children(tree: Mapping[str, str]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for node, parent in sorted(tree.items()):
        children.setdefault(parent, []).append(node)
    return children


def _preorder(children: Mapping[str, list[str]]) -> list[str]:
    order: list[str] = []
    pending = [TOP]
    while pending:
        node = pending.pop()
        order.append(node)
        pending += reversed(children.get(node, []))
    return order


def _totals(counts: Mapping[str, int], children: Mapping[str, list[str]]) -> dict[str, int]:
    """Each node's count: a leaf's from `counts`, an inner node's its children's sum."""
    totals: dict[str, int] = {}
    for node in reversed(_preorder(children)):
        below = children.get(node)
        totals[node] = sum(totals[child] for child in below) if below else counts.get(node, 0)
    return totals
