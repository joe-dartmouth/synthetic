"""What a BKB's S-nodes imply from an anchor, and the probability of one inference.

Santos, Jurmain and Ragazzi (`santos-2024`, sections 5.3 and 7.2) instantiate a BKO by adding, pass
after pass, every rule whose antecedents hold, and give an inference the product of the S-nodes in
it, the anchor taken as evidence. `instantiate` returns the passes, `reach` the I-nodes they reach
with the S-node credited for each, `derivation` the S-nodes, source priors included, of the
inference that reached one I-node, and `probability` their product. No posterior is computed:
`bko.bkb.reason` holds the only update (ADR `research/bkb-layer-source`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from bko.bkb.errors import QueryError
from bko.bkb.fusion import is_source
from bko.bkb.structure import BKB, INode, SNode


def instantiate(bkb: BKB, anchor: Iterable[INode]) -> tuple[tuple[int, ...], ...]:
    """The passes of instantiating `bkb` from `anchor`.

    Each pass adds every S-node that no earlier pass added, whose head is not on a source
    component, and whose tail outside source components holds only anchor I-nodes and heads of
    S-nodes earlier passes added; source I-nodes stand on their priors. Passes stop when one adds
    nothing.

    Returns:
        Each pass's S-node indices, ascending.

    Raises:
        QueryError: If the anchor is empty, or names an I-node `bkb` lacks or one on a source
            component.
    """
    held = set(_anchor(bkb, anchor))
    snodes = bkb.snodes
    added: set[int] = set()
    passes: list[tuple[int, ...]] = []
    while True:
        new = tuple(
            index
            for index, snode in enumerate(snodes)
            if index not in added and not is_source(snode.head[0]) and _supported(snode, held)
        )
        if not new:
            return tuple(passes)
        passes.append(new)
        added.update(new)
        held.update(snodes[index].head for index in new)


def reach(bkb: BKB, anchor: Iterable[INode]) -> dict[INode, int | None]:
    """Every I-node instantiating `bkb` from `anchor` reaches, with the S-node credited for it.

    Returns:
        Each anchor I-node mapped to None, in the order given, then each reached I-node mapped to
        the lowest-index S-node heading it in the first pass that does, in the order reached.

    Raises:
        QueryError: As `instantiate`.
    """
    start = _anchor(bkb, anchor)
    snodes = bkb.snodes
    reached: dict[INode, int | None] = {}
    for inode in start:
        reached[inode] = None
    for added in instantiate(bkb, start):
        for index in added:
            reached.setdefault(snodes[index].head, index)
    return reached


def derivation(bkb: BKB, reached: Mapping[INode, int | None], goal: INode) -> tuple[int, ...]:
    """The S-nodes of the inference that reached `goal`, given what `reach` returned.

    Following `reached` back from `goal` to the anchor, each credited S-node joins, with the prior
    S-node of each source I-node in its tail.

    Returns:
        Their indices, ascending.

    Raises:
        QueryError: If `goal` is not in `reached`, or a source I-node in a credited tail has no
            prior S-node or more than one.
    """
    if goal not in reached:
        msg = f"{goal!r} is not reached"
        raise QueryError(msg)
    snodes = bkb.snodes
    found: set[int] = set()
    pending = [goal]
    while pending:
        index = reached[pending.pop()]
        if index is None or index in found:
            continue
        found.add(index)
        for inode in snodes[index].tail:
            if is_source(inode[0]):
                found.add(_prior(bkb, snodes, inode))
            else:
                pending.append(inode)
    return tuple(sorted(found))


def probability(bkb: BKB, snodes: Iterable[int]) -> float:
    """The product of the probabilities of `snodes`, taken in ascending index order.

    Raises:
        QueryError: If an index is not an S-node of `bkb`.
    """
    held = bkb.snodes
    indices = sorted(set(snodes))
    for index in indices:
        if not 0 <= index < len(held):
            msg = f"{index!r} is not an S-node of {bkb.name!r}"
            raise QueryError(msg)
    return math.prod((held[index].prob for index in indices), start=1.0)


def _anchor(bkb: BKB, anchor: Iterable[INode]) -> list[INode]:
    start: list[INode] = []
    for inode in anchor:
        if inode not in bkb:
            msg = f"the anchor I-node {inode!r} is not an I-node of {bkb.name!r}"
            raise QueryError(msg)
        if is_source(inode[0]):
            msg = f"the anchor I-node {inode!r} is on a source component"
            raise QueryError(msg)
        if inode not in start:
            start.append(inode)
    if not start:
        msg = "the anchor is empty"
        raise QueryError(msg)
    return start


def _supported(snode: SNode, held: set[INode]) -> bool:
    return all(is_source(inode[0]) or inode in held for inode in snode.tail)


def _prior(bkb: BKB, snodes: Sequence[SNode], inode: INode) -> int:
    priors = [index for index in bkb.find_snodes(*inode) if not snodes[index].tail]
    if len(priors) != 1:
        msg = f"the source I-node {inode!r} has {len(priors)} prior S-nodes, not one"
        raise QueryError(msg)
    return priors[0]
