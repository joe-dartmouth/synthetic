"""Source-weighted fusion of BKB fragments.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-17; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).

`fuse` follows pybkb's `fuse` and, by default, the fusion of Santos et al. (`santos-2024`,
section 6.1). Each S-node of each fragment gains a source I-node in its tail, whose component is
`__Source__[<head component>]` and whose state is the fragment's source name. Each source I-node
gets one prior S-node: its weight divided by a total, which `Normalization` chooses. Under
`FRAGMENTS`, the default, the total is every fragment's reliability, so the priors on one source
component sum to less than 1 where a fragment holds no S-node on its head component. Under
pybkb's `COMPONENT` and `COMPONENT_COLLAPSED` it is the total over the source I-nodes of that
component, which are the sources holding S-nodes on that head component; `COMPONENT_COLLAPSED`
also makes one fused S-node stand for every fragment holding that exact S-node, with the set of
their names as its source state and the sum of their reliabilities as its weight. S-nodes merge
only when head, tail and probability are exactly equal. Every total is summed left to right:
under `FRAGMENTS` over the fragments, as Santos et al.'s algorithm sums, and under pybkb's modes
in pybkb's order, so their priors match pybkb's to the last bit; `sum` compensates from Python
3.12 and would not.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from bko.bkb.errors import FusionError
from bko.bkb.structure import BKB, INode, SNode


logger = logging.getLogger(__name__)

SOURCE_PREFIX: Final = "__Source__"
"""What every source component's name starts with."""

_TOLERANCE: Final = 1e-9
"""How far a source component's priors may sum from 1: each is a quotient, so rounded."""

type _Key = tuple[INode, frozenset[INode], float]
type _Row = tuple[_Key, INode, float]


def source_component(component: str) -> str:
    """The source component `fuse` adds to the tails of S-nodes on `component`."""
    return f"{SOURCE_PREFIX}[{component}]"


def is_source(component: str) -> bool:
    """Whether `component` is a source component, as `fuse` names them."""
    return component.startswith(f"{SOURCE_PREFIX}[") and component.endswith("]")


class Normalization(StrEnum):
    """How `fuse` weights sources: the total a source's weight is divided by, and collapse."""

    FRAGMENTS = "fragments"
    """Every fragment's reliability, with no collapse (Santos et al.); the default."""
    COMPONENT = "component"
    """The weights of the sources holding S-nodes on the source's component (pybkb)."""
    COMPONENT_COLLAPSED = "component-collapsed"
    """`COMPONENT`, with one fused S-node per S-node several fragments hold (pybkb's collapse)."""

    @property
    def collapse(self) -> bool:
        """Whether one fused S-node stands for every fragment holding that exact S-node."""
        return self is Normalization.COMPONENT_COLLAPSED


def fuse(
    fragments: Sequence[BKB],
    reliabilities: Sequence[float],
    *,
    sources: Sequence[str] | None = None,
    normalization: Normalization = Normalization.FRAGMENTS,
    name: str = "fused",
) -> BKB:
    """Fuse BKB fragments into one BKB, weighting each fragment's S-nodes by its reliability.

    Args:
        fragments: The BKBs to fuse; their order fixes the fused BKB's order.
        reliabilities: One per fragment, each finite and positive.
        sources: One source name per fragment; each fragment's name by default.
        normalization: How each source is weighted; Santos et al.'s by default.
        name: The fused BKB's name.

    Returns:
        A new BKB; the fragments are not changed.

    Raises:
        FusionError: If there are no fragments, the sequences differ in length, a reliability is
            not a finite positive number or is an integer that overflows a float, or the
            reliabilities overflow when summed, a source name is not a non-empty string or two
            fragments share one, a fragment holds a source component already, a fragment holds
            one S-node twice, or `normalization` is not a `Normalization`.
    """
    names = _check(fragments, reliabilities, sources)
    mode = _mode(normalization)
    supports: dict[_Key, list[tuple[float, str]]] = {}
    for fragment, reliability, source in zip(fragments, reliabilities, names, strict=True):
        for snode in fragment.snodes:
            key = (snode.head, snode.tail, snode.prob)
            supports.setdefault(key, []).append((reliability, source))
    rows = _collapsed(supports) if mode.collapse else _per_source(supports)
    fused = BKB(name)
    for fragment in fragments:
        for component, state in fragment.inodes:
            fused.add_inode(component, state)
    _add(fused, rows, _total(reliabilities) if mode is Normalization.FRAGMENTS else None)
    logger.info(
        "fused fragments=%d normalization=%s inodes=%d snodes=%d name=%s",
        len(fragments),
        mode.value,
        len(fused.inodes),
        len(fused.snodes),
        name,
    )
    return fused


def check_fused(
    fused: BKB,
    fragments: Sequence[BKB],
    *,
    sources: Sequence[str] | None = None,
    normalization: Normalization = Normalization.FRAGMENTS,
) -> None:
    """Check that `fused` is a valid fusion of `fragments`, as `fuse` builds one.

    Four things must hold. Every input S-node is present, with its source I-node added to its
    tail and its probability unchanged. Every fused S-node other than a source prior maps to an
    input S-node of each source its source state names. Each source component's priors sum to 1
    under pybkb's modes; under `FRAGMENTS` they sum to at most 1, and to 1 where every fragment
    holds an S-node on the head component, since the total is every fragment's reliability. Every
    probability is in [0, 1], which `BKB.add_snode` enforces as each S-node is added, so a `BKB`
    cannot fail it and this check does not repeat it. Mutual exclusion is not required (ADR
    `research/data-contradiction-handling`).

    Args:
        fused: The fused BKB.
        fragments: The fragments it was fused from.
        sources: One source name per fragment; each fragment's name by default, as `fuse` takes.
        normalization: The mode `fused` was fused under.

    Raises:
        FusionError: If any of the first three fails, naming the first S-node or component that
            breaks it, or if `sources` does not give one distinct name per fragment.
    """
    names = [fragment.name for fragment in fragments] if sources is None else list(sources)
    if len(names) != len(fragments) or len(set(names)) != len(names):
        msg = f"{len(fragments)} fragments need as many distinct source names, not {names!r}"
        raise FusionError(msg)
    inputs: dict[_Key, set[str]] = {}
    for fragment, source in zip(fragments, names, strict=True):
        for snode in fragment.snodes:
            inputs.setdefault((snode.head, snode.tail, snode.prob), set()).add(source)
    held: dict[_Key, set[str]] = {}
    priors: dict[str, list[float]] = {}
    for index, snode in enumerate(fused.snodes):
        if is_source(snode.head[0]) and not snode.tail:
            priors.setdefault(snode.head[0], []).append(snode.prob)
            continue
        key, named = _unsourced(snode)
        if not named or not named <= inputs.get(key, set()):
            msg = f"S-node {index} on {snode.head!r} maps to no input S-node of {sorted(named)}"
            raise FusionError(msg)
        held.setdefault(key, set()).update(named)
    for key, wanted in inputs.items():
        if missing := wanted - held.get(key, set()):
            msg = f"{fused.name!r} lacks S-node on {key[0]!r} at {key[2]!r} from {sorted(missing)}"
            raise FusionError(msg)
    for component, probs in priors.items():
        _check_priors(component, probs, len(fused.states(component)), len(fragments), normalization)


def _unsourced(snode: SNode) -> tuple[_Key, set[str]]:
    """`snode` without its source I-node, and the sources that I-node's state names."""
    own = source_component(snode.head[0])
    rest = frozenset(inode for inode in snode.tail if inode[0] != own)
    named: set[str] = set()
    for component, state in snode.tail:
        if component == own:
            named |= {state} if isinstance(state, str) else state
    return (snode.head, rest, snode.prob), named


def _check_priors(
    component: str, probs: list[float], held: int, fragments: int, normalization: Normalization
) -> None:
    total = math.fsum(probs)
    whole = normalization is not Normalization.FRAGMENTS or held == fragments
    if total > 1.0 + _TOLERANCE or (whole and total < 1.0 - _TOLERANCE):
        wanted = "1" if whole else "at most 1"
        msg = f"the priors on {component!r} sum to {total!r}, not {wanted}"
        raise FusionError(msg)


def _check(
    fragments: Sequence[BKB], reliabilities: Sequence[float], sources: Sequence[str] | None
) -> list[str]:
    if not fragments:
        msg = "there are no fragments to fuse"
        raise FusionError(msg)
    if isinstance(sources, str):
        msg = f"the source names {sources!r} are one string, not one name per fragment"
        raise FusionError(msg)
    names = [fragment.name for fragment in fragments] if sources is None else list(sources)
    if not len(fragments) == len(reliabilities) == len(names):
        msg = (
            f"{len(fragments)} fragments, {len(reliabilities)} reliabilities "
            f"and {len(names)} source names"
        )
        raise FusionError(msg)
    for reliability in reliabilities:
        _check_reliability(reliability)
    for source in names:
        if not isinstance(source, str) or not source:
            msg = f"the source name {source!r} is not a non-empty string"
            raise FusionError(msg)
    repeated = [source for source in names if names.count(source) > 1]
    if repeated:
        msg = f"two fragments share the source name {repeated[0]!r}"
        raise FusionError(msg)
    for fragment in fragments:
        _check_fragment(fragment)
    return names


def _check_reliability(reliability: object) -> None:
    try:
        valid = (
            isinstance(reliability, int | float)
            and not isinstance(reliability, bool)
            and math.isfinite(reliability)
            and reliability > 0
        )
    except OverflowError:
        # No repr: an int past 4300 digits cannot be printed.
        msg = "a reliability is an integer that overflows a float"
        raise FusionError(msg) from None
    if not valid:
        msg = f"the reliability {reliability!r} is not a finite positive number"
        raise FusionError(msg)


def _check_fragment(fragment: BKB) -> None:
    held = [component for component in fragment.components() if is_source(component)]
    if held:
        msg = f"{fragment.name!r} holds the source component {held[0]!r}; fuse its fragments"
        raise FusionError(msg)
    seen: set[_Key] = set()
    for index, snode in enumerate(fragment.snodes):
        key = (snode.head, snode.tail, snode.prob)
        if key in seen:
            msg = f"{fragment.name!r} holds S-node {index} on {snode.head!r} twice"
            raise FusionError(msg)
        seen.add(key)


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


def _mode(normalization: str) -> Normalization:
    try:
        return Normalization(normalization)
    except ValueError:
        known = ", ".join(mode.value for mode in Normalization)
        msg = f"the normalization {normalization!r} is not one of {known}"
        raise FusionError(msg) from None


def _total(reliabilities: Sequence[float]) -> float:
    total = 0.0
    for reliability in reliabilities:
        total += reliability  # Left to right, as Santos et al.'s algorithm sums.
    return total


def _add(fused: BKB, rows: list[_Row], total: float | None) -> None:
    for _, (component, state), _ in rows:
        fused.add_inode(component, state)
    weights: dict[INode, float] = {}
    for _, source, weight in rows:
        weights.setdefault(source, weight)
    totals: dict[str, float] = {}
    for (component, _), weight in weights.items():
        if total is None:
            totals[component] = totals.get(component, 0.0) + weight
        else:
            totals[component] = total
    for component, value in totals.items():
        if not math.isfinite(value):
            msg = f"the reliabilities of the sources of {component!r} overflow when summed"
            raise FusionError(msg)
    primed: set[INode] = set()
    for (head, tail, prob), source, _ in rows:
        fused.add_snode(head[0], head[1], prob, [*tail, source])
        if source not in primed:
            fused.add_snode(source[0], source[1], weights[source] / totals[source[0]])
            primed.add(source)
