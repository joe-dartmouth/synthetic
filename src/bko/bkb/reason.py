"""The log joint mass of each state of a target component, given complete evidence.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-16; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).

`log_joint_complete` is pybkb's `BKBReasoner.heuristic_update_complete_fused` under an honest
name. For each state t of the target it takes the assignment of the evidence with target = t, and
for each I-node of that assignment the S-nodes on it whose tail agrees with the assignment outside
source components; each S-node brings its probability times the prior of the source I-node in its
tail, if any. The result for t is the log of the product, over the assignment's I-nodes, of the
sum over each I-node's S-nodes: the summed probability of every choice of one S-node per I-node,
which is pybkb's sum over its Cartesian product, computed factorized and with a stable log-sum.
Like pybkb, it sums every choice, cyclic ones included. A state some I-node has no agreeing S-node
for is None, and one whose every choice has probability 0 is minus infinity. Nothing is normalized
across states: the values are joint masses, equal to P(target = t, evidence) when the BKB is
mutex, no choice is cyclic and its inferences' probabilities sum to 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from bko.bkb.errors import QueryError
from bko.bkb.fusion import is_source
from bko.bkb.structure import BKB, INode, SNode, State


def log_joint_complete(
    bkb: BKB, target: str, evidence: Mapping[str, str]
) -> dict[State, float | None]:
    """The log joint mass of each state of `target` with `evidence`, keyed by state.

    Args:
        bkb: The BKB, usually fused.
        target: A component of the BKB that is not a source component.
        evidence: A state for every other component the answer depends on.

    Returns:
        For each state of `target`, in the BKB's order: the natural log of its joint mass with
        the evidence, minus infinity when that mass is 0, or None when some I-node of the
        assignment has no S-node whose tail agrees with it. None takes precedence over the
        missing-component refusal below, whichever I-node either concerns.

    Raises:
        QueryError: If `target` has no I-nodes, is a source component or is in `evidence`; an
            evidence component or state is not a string, or its I-node is not in the BKB or is
            on a source component; the answer depends on a component `evidence` leaves out; or a
            source I-node does not head exactly one prior.
    """
    _check_query(bkb, target, evidence)
    snodes = bkb.snodes
    return {
        state: _log_mass(bkb, snodes, {**evidence, target: state}) for state in bkb.states(target)
    }


def _check_query(bkb: BKB, target: str, evidence: Mapping[str, str]) -> None:
    if is_source(target) or not bkb.states(target):
        msg = f"the target {target!r} is a source component or has no I-nodes in {bkb.name!r}"
        raise QueryError(msg)
    if target in evidence:
        msg = f"the target {target!r} is also in the evidence"
        raise QueryError(msg)
    for component, state in evidence.items():
        if not isinstance(component, str) or not isinstance(state, str):
            msg = f"the evidence {component!r}={state!r} is not a pair of strings"
            raise QueryError(msg)
        if is_source(component) or (component, state) not in bkb:
            msg = f"the evidence {component}={state!r} is on a source or not in {bkb.name!r}"
            raise QueryError(msg)


def _log_mass(bkb: BKB, snodes: tuple[SNode, ...], assignment: Mapping[str, State]) -> float | None:
    options: list[list[SNode]] = []
    undetermined: INode | None = None
    for inode in assignment.items():
        verdicts = [
            (snodes[index], _agrees(snodes[index].tail, assignment))
            for index in bkb.find_snodes(*inode)
        ]
        if all(verdict is False for _, verdict in verdicts):
            return None
        if undetermined is None and any(verdict is None for _, verdict in verdicts):
            undetermined = inode
        options.append([snode for snode, verdict in verdicts if verdict])
    if undetermined is not None:
        msg = f"an S-node on {undetermined!r} names a component the evidence leaves out"
        raise QueryError(msg)
    return math.fsum(
        _log_sum([_log_weight(bkb, snodes, snode) for snode in chosen]) for chosen in options
    )


def _agrees(tail: frozenset[INode], assignment: Mapping[str, State]) -> bool | None:
    # True: agrees outside sources; False: contradicts; None: agrees but names an unset component.
    verdict: bool | None = True
    for component, state in tail:
        if is_source(component):
            continue
        if component not in assignment:
            verdict = None
        elif assignment[component] != state:
            return False
    return verdict


def _log_weight(bkb: BKB, snodes: tuple[SNode, ...], snode: SNode) -> float:
    weight = _log(snode.prob)
    # A fixed order, so a tail with several source I-nodes sums the same under every hash seed.
    for member in sorted((m for m in snode.tail if is_source(m[0])), key=repr):
        weight += _log(_prior(bkb, snodes, member))
    return weight


def _prior(bkb: BKB, snodes: tuple[SNode, ...], source: INode) -> float:
    heads = bkb.find_snodes(*source)
    if len(heads) != 1 or snodes[heads[0]].tail:
        msg = f"the source I-node {source!r} heads {len(heads)} S-nodes, not one prior"
        raise QueryError(msg)
    return snodes[heads[0]].prob


def _log(prob: float) -> float:
    return math.log(prob) if prob > 0.0 else -math.inf


def _log_sum(values: list[float]) -> float:
    high = max(values)
    if high == -math.inf:
        return high
    return high + math.log(math.fsum(math.exp(value - high) for value in values))
