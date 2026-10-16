"""The BKB structure: I-nodes, S-nodes and the mutual-exclusion check.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-16; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).

An I-node pairs a component with one of its states. An S-node gives its head I-node a probability
given its tail, a set of I-nodes on other components; an S-node with an empty tail is a prior.
Components and states are strings, except that a collapsed source I-node (`bko.bkb.fusion`) has
the set of its source names as its state. A BKB keeps I-nodes and S-nodes in the order they were
added, and an S-node's index is its position. Two BKBs are equal when they hold the same I-nodes
and the same S-nodes, counted with multiplicity, whatever their names, descriptions and order; a
BKB is mutable, so it is unhashable.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from bko.bkb.errors import StructureError


type State = str | frozenset[str]
"""A state: a string, or for a collapsed source I-node the set of its source names."""

type INode = tuple[str, State]
"""An instantiation node: a component and one of its states."""


@dataclass(frozen=True, slots=True)
class SNode:
    """A support node: the probability of its head given its tail.

    Attributes:
        head: The I-node the S-node supports.
        prob: The probability, in [0, 1].
        tail: The I-nodes it is conditioned on, one per component; empty for a prior.
    """

    head: INode
    prob: float
    tail: frozenset[INode]


class BKB:
    """A Bayesian Knowledge Base, built by adding I-nodes and then S-nodes over them.

    Attributes:
        name: What the BKB is called; `bko.bkb.fusion.fuse` uses it as the default source name.
        description: Free text, or None.
    """

    __hash__ = None

    def __init__(self, name: str, description: str | None = None) -> None:
        self.name = name
        self.description = description
        self._index: dict[INode, int] = {}
        self._states: dict[str, list[State]] = {}
        self._snodes: list[SNode] = []
        self._by_head: dict[INode, list[int]] = {}

    def __repr__(self) -> str:
        return f"BKB(name={self.name!r}, inodes={len(self._index)}, snodes={len(self._snodes)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BKB):
            return NotImplemented
        same_inodes = set(self.inodes) == set(other.inodes)
        return same_inodes and Counter(self.snodes) == Counter(other.snodes)

    def __contains__(self, inode: object) -> bool:
        return inode in self._index

    @property
    def inodes(self) -> tuple[INode, ...]:
        """The I-nodes, in the order they were added."""
        return tuple(self._index)

    @property
    def snodes(self) -> tuple[SNode, ...]:
        """The S-nodes, in the order they were added."""
        return tuple(self._snodes)

    def components(self) -> tuple[str, ...]:
        """The components, in the order their first I-node was added."""
        return tuple(self._states)

    def states(self, component: str) -> tuple[State, ...]:
        """The states of `component`, in the order they were added; empty if it has none."""
        return tuple(self._states.get(component, ()))

    def add_inode(self, component: str, state: State) -> None:
        """Add the I-node `(component, state)`; adding one the BKB holds does nothing.

        Raises:
            StructureError: If `component` is not a string, or `state` is neither a string nor a
                non-empty frozenset of strings.
        """
        _check_names(component, state)
        if (component, state) in self._index:
            return
        self._index[component, state] = len(self._index)
        self._states.setdefault(component, []).append(state)

    def add_snode(
        self, component: str, state: State, prob: float, tail: Iterable[INode] = ()
    ) -> int:
        """Add an S-node on `(component, state)` with probability `prob` given `tail`.

        Returns:
            The new S-node's index.

        Raises:
            StructureError: If the head or a tail I-node is not in the BKB, `prob` is not a number
                in [0, 1], or the tail names the head's component or names a component twice.
        """
        head: INode = (component, state)
        if head not in self._index:
            msg = f"the head {head!r} is not an I-node of {self.name!r}"
            raise StructureError(msg)
        if isinstance(prob, bool) or not isinstance(prob, int | float) or not 0.0 <= prob <= 1.0:
            msg = f"an S-node on {head!r} has probability {prob!r}, not a number in [0, 1]"
            raise StructureError(msg)
        members = frozenset(tail)
        self._check_tail(head, members)
        self._snodes.append(SNode(head, float(prob), members))
        index = len(self._snodes) - 1
        self._by_head.setdefault(head, []).append(index)
        return index

    def find_snodes(
        self,
        component: str,
        state: State,
        *,
        prob: float | None = None,
        tail_subset: Iterable[INode] = (),
    ) -> list[int]:
        """The indices of the S-nodes on `(component, state)`, in the order they were added.

        Args:
            component: The head's component.
            state: The head's state.
            prob: Keep only S-nodes with exactly this probability.
            tail_subset: Keep only S-nodes whose tail holds every one of these I-nodes.

        Returns:
            The matching indices; empty if none match.

        Raises:
            StructureError: If `(component, state)` is not an I-node of the BKB.
        """
        head: INode = (component, state)
        if head not in self._index:
            msg = f"{head!r} is not an I-node of {self.name!r}"
            raise StructureError(msg)
        wanted = frozenset(tail_subset)
        return [
            index
            for index in self._by_head.get(head, [])
            if (prob is None or self._snodes[index].prob == prob)
            and wanted <= self._snodes[index].tail
        ]

    def non_mutex_pair(self) -> tuple[int, int] | None:
        """The first two S-nodes on one head that do not exclude each other, or None.

        Two S-nodes on one head exclude each other when both tails are non-empty and some
        component is in both tails with different states, as in pybkb's `are_snodes_mutex`.
        """
        for indices in self._by_head.values():
            for first, second in itertools.combinations(indices, 2):
                if not _exclusive(self._snodes[first].tail, self._snodes[second].tail):
                    return first, second
        return None

    def is_mutex(self) -> bool:
        """Whether every two S-nodes on one head exclude each other."""
        return self.non_mutex_pair() is None

    def _check_tail(self, head: INode, tail: frozenset[INode]) -> None:
        for inode in tail:
            if inode not in self._index:
                msg = (
                    f"the tail I-node {inode!r} of an S-node on {head!r} "
                    f"is not an I-node of {self.name!r}"
                )
                raise StructureError(msg)
        components = [component for component, _ in tail]
        if head[0] in components or len(set(components)) != len(components):
            msg = (
                f"the tail of an S-node on {head!r} names its head's component or a component twice"
            )
            raise StructureError(msg)


def _check_names(component: object, state: object) -> None:
    if not isinstance(component, str):
        msg = f"the component {component!r} is not a string"
        raise StructureError(msg)
    if isinstance(state, str):
        return
    if isinstance(state, frozenset) and state and all(isinstance(name, str) for name in state):
        return
    msg = f"the state {state!r} of {component!r} is neither a string nor a set of source names"
    raise StructureError(msg)


def _exclusive(first: frozenset[INode], second: frozenset[INode]) -> bool:
    if not first or not second:
        return False
    states = dict(first)
    return any(states.get(component, state) != state for component, state in second)
