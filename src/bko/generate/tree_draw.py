"""Each tree field's leaves at a cell, as the direct sampler draws them and the trace reads them.

`TreeDraw` holds every tree field's tree (`bko.generate.tree.column_trees`) and, per label, each
field's label-group records by the leaf they show, read once per label. `TreeDraw.at` reads a
cell's records against its label group's into each field's `Placed`: the ancestor-shrunk shares at
`GROUP_WEIGHT`, each node's parts, and the leaves of positive share with their cumulative shares.
`leaf_at` places one uniform among those leaves. `bko.generate.direct.sample` draws through it and
`bko.evaluate.redraw.TreeShares` traces through it, so the trace reads the draw's own shares (ADR
`research/ancestor-shrunk-shares`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from bko.gaps.observe import Cell, cell_key
from bko.gaps.shares import GROUP_WEIGHT
from bko.generate.donors import _Source
from bko.generate.errors import SamplingError
from bko.generate.tree import ColumnTree, TreePart, leaves, tree_shares


@dataclass(frozen=True, slots=True, eq=False)
class Placed:
    """One tree field's shares at a cell and the records showing each leaf.

    Attributes:
        column: The tree field.
        leaves: The leaves of positive share, in depth-first order.
        cumulative: Their cumulative shares, in that order.
        parts: Each node's `TreePart` at the cell, by node.
        own: The cell's records by the leaf they show.
        group: The label group's records by the leaf they show.
    """

    column: str
    leaves: tuple[str, ...]
    cumulative: np.ndarray
    parts: Mapping[str, TreePart]
    own: Mapping[str, list[_Source]]
    group: Mapping[str, list[_Source]]


class TreeDraw:
    """Every tree field's shares at the cells of one draw, each label group's leaves read once."""

    def __init__(self, trees: Mapping[str, ColumnTree]) -> None:
        """`trees` are the tree fields' (`column_trees`), in column order."""
        self.trees = trees
        self._groups: dict[tuple[int, str], dict[str, list[_Source]]] = {}

    def at(self, cell: Cell, own: Sequence[_Source], group: Sequence[_Source]) -> dict[str, Placed]:
        """Each tree field's `Placed` at `cell`, whose records are `own`.

        `group` is the label group's records; the first call for a label reads them, and later
        calls for that label reuse that reading.

        Raises:
            SamplingError: If a record's value of a tree field is not finite (`ColumnTree.leaf`).
        """
        placed = {}
        for column, tree in self.trees.items():
            key = (int(cell[2]), column)
            if key not in self._groups:
                self._groups[key] = _by_leaf(tree, group)
            mine, theirs = _by_leaf(tree, own), self._groups[key]
            found, parts = tree_shares(_sizes(mine), _sizes(theirs), tree.tree, GROUP_WEIGHT)
            positive = tuple(leaf for leaf in leaves(tree.tree) if found[leaf] > 0)
            placed[column] = Placed(
                column,
                positive,
                np.cumsum([found[leaf] for leaf in positive]),
                {part.node: part for part in parts},
                mine,
                theirs,
            )
        return placed


def leaf_at(placed: Placed, uniform: float, cell: Cell) -> str:
    """The leaf `uniform` falls on by cumulative share, clipped to the last leaf.

    A uniform on a boundary falls on the leaf to its right.

    Raises:
        SamplingError: If no leaf has a positive share, as when the label group holds no record.
    """
    if not placed.leaves:
        msg = f"{cell_key(cell)} cannot draw {placed.column}: no leaf has a positive share"
        raise SamplingError(msg)
    position = int(np.searchsorted(placed.cumulative, uniform, side="right"))
    return placed.leaves[min(position, len(placed.leaves) - 1)]


def _by_leaf(tree: ColumnTree, records: Sequence[_Source]) -> dict[str, list[_Source]]:
    """`records` by the leaf each shows.

    Raises:
        SamplingError: As `ColumnTree.leaf`, naming the record's id.
    """
    shown: dict[str, list[_Source]] = {}
    for record in records:
        try:
            leaf = tree.leaf(record[1])
        except SamplingError as exc:
            msg = f"training record {record[0]}: {exc}"
            raise SamplingError(msg) from exc
        shown.setdefault(leaf, []).append(record)
    return shown


def _sizes(shown: Mapping[str, list[_Source]]) -> dict[str, int]:
    return {leaf: len(records) for leaf, records in shown.items()}
