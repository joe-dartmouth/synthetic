"""Source-weighted fusion of BKB fragments.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-16; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).

`fuse` follows pybkb's `fuse`. Each S-node of each fragment gains a source I-node in its tail,
whose component is `__Source__[<head component>]` and whose state is the fragment's source name;
with `collapse`, one fused S-node stands for every fragment holding that exact S-node, and its
source state is the set of their names. Each source I-node gets one prior S-node: its reliability
(with `collapse`, the sum over its names) divided by the total over the source I-nodes of its
component, which are the sources holding S-nodes on that head component. S-nodes merge only when
head, tail and probability are exactly equal. Totals are summed left to right in pybkb's order, so
priors match pybkb's to the last bit; `sum` compensates from Python 3.12 and would not.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Final

from bko.bkb.errors import FusionError
from bko.bkb.structure import BKB, INode


logger = logging.getLogger(__name__)

SOURCE_PREFIX: Final = "__Source__"
"""What every source component's name starts with."""

type _Key = tuple[INode, frozenset[INode], float]
type _Row = tuple[_Key, INode, float]


def source_component(component: str) -> str:
    """The source component `fuse` adds to the tails of S-nodes on `component`."""
    return f"{SOURCE_PREFIX}[{component}]"


def is_source(component: str) -> bool:
    """Whether `component` is a source component, as `fuse` names them."""
    return component.startswith(f"{SOURCE_PREFIX}[") and component.endswith("]")


def fuse(
    fragments: Sequence[BKB],
    reliabilities: Sequence[float],
    *,
    sources: Sequence[str] | None = None,
    collapse: bool = True,
    name: str = "fused",
) -> BKB:
    """Fuse BKB fragments into one BKB, weighting each fragment's S-nodes by its reliability.

    Args:
        fragments: The BKBs to fuse; their order fixes the fused BKB's order.
        reliabilities: One per fragment, each finite and positive.
        sources: One source name per fragment; each fragment's name by default.
        collapse: Merge the copies of an S-node that several fragments hold, as pybkb does by
            default.
        name: The fused BKB's name.

    Returns:
        A new BKB; the fragments are not changed.

    Raises:
        FusionError: If there are no fragments, the sequences differ in length, a reliability is
            not a finite positive number, two fragments share a source name, or a fragment holds
            a source component already.
    """
    names = _check(fragments, reliabilities, sources)
    supports: dict[_Key, list[tuple[float, str]]] = {}
    for fragment, reliability, source in zip(fragments, reliabilities, names, strict=True):
        for snode in fragment.snodes:
            key = (snode.head, snode.tail, snode.prob)
            supports.setdefault(key, []).append((reliability, source))
    rows = _collapsed(supports) if collapse else _per_source(supports)
    fused = BKB(name)
    for fragment in fragments:
        for component, state in fragment.inodes:
            fused.add_inode(component, state)
    _add(fused, rows)
    logger.info(
        "fused fragments=%d collapse=%s inodes=%d snodes=%d",
        len(fragments),
        collapse,
        len(fused.inodes),
        len(fused.snodes),
    )
    return fused


def _check(
    fragments: Sequence[BKB], reliabilities: Sequence[float], sources: Sequence[str] | None
) -> list[str]:
    if not fragments:
        msg = "there are no fragments to fuse"
        raise FusionError(msg)
    names = [fragment.name for fragment in fragments] if sources is None else list(sources)
    if not len(fragments) == len(reliabilities) == len(names):
        msg = (
            f"{len(fragments)} fragments, {len(reliabilities)} reliabilities "
            f"and {len(names)} source names"
        )
        raise FusionError(msg)
    for reliability in reliabilities:
        if (
            isinstance(reliability, bool)
            or not isinstance(reliability, int | float)
            or not math.isfinite(reliability)
            or reliability <= 0
        ):
            msg = f"the reliability {reliability!r} is not a finite positive number"
            raise FusionError(msg)
    if len(set(names)) != len(names):
        msg = f"two fragments share a source name: {names}"
        raise FusionError(msg)
    for fragment in fragments:
        held = [component for component in fragment.components() if is_source(component)]
        if held:
            msg = f"{fragment.name!r} holds the source component {held[0]!r}; fuse its fragments"
            raise FusionError(msg)
    return names


def _per_source(supports: dict[_Key, list[tuple[float, str]]]) -> list[_Row]:
    rows: list[_Row] = [
        (key, (source_component(key[0][0]), source), reliability)
        for key, entries in supports.items()
        for reliability, source in entries
    ]
    return rows


def _collapsed(supports: dict[_Key, list[tuple[float, str]]]) -> list[_Row]:
    rows: list[_Row] = []
    for key, entries in supports.items():
        weight = 0.0
        for reliability, _ in entries:
            weight += reliability  # Left to right, as pybkb sums.
        names = frozenset(source for _, source in entries)
        rows.append((key, (source_component(key[0][0]), names), weight))
    return rows


def _add(fused: BKB, rows: list[_Row]) -> None:
    for _, (component, state), _ in rows:
        fused.add_inode(component, state)
    weights: dict[INode, float] = {}
    for _, source, weight in rows:
        weights.setdefault(source, weight)
    totals: dict[str, float] = {}
    for (component, _), weight in weights.items():
        totals[component] = totals.get(component, 0.0) + weight
    primed: set[INode] = set()
    for (head, tail, prob), source, _ in rows:
        fused.add_snode(head[0], head[1], prob, [*tail, source])
        if source not in primed:
            fused.add_snode(source[0], source[1], weights[source] / totals[source[0]])
            primed.add(source)
