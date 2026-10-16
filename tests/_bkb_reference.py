"""pybkb's `heuristic_update_complete_fused`, transcribed in exact arithmetic as a test reference.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-16; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).

It reads a BKB document (`bko.bkb.serialize`) as plain data and imports nothing from `bko`, so it
checks `bko.bkb.reason` without sharing its code. It walks pybkb's Cartesian product of S-node
choices and sums them as fractions, which is exponential, so tests keep it to small queries.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Any


def _is_source(component: str) -> bool:
    return component.startswith("__Source__[") and component.endswith("]")


def reference_mass(
    document: dict[str, Any], target: str, evidence: dict[str, str]
) -> dict[str | frozenset[str], Fraction | None]:
    inodes = [
        (component, state if isinstance(state, str) else frozenset(state))
        for component, state in document["inodes"]
    ]
    snodes = [
        (inodes[s["head"]], Fraction(s["prob"]), [inodes[j] for j in s["tail"]])
        for s in document["snodes"]
    ]
    priors = {head: prob for head, prob, tail in snodes if _is_source(head[0]) and not tail}
    return {
        state: _mass(snodes, priors, {**evidence, target: state})
        for component, state in inodes
        if component == target
    }


def _mass(
    snodes: list[Any], priors: dict[Any, Fraction], assignment: dict[str, Any]
) -> Fraction | None:
    options = []
    for inode in assignment.items():
        agreeing = [
            snode
            for snode in snodes
            if snode[0] == inode and all(_is_source(c) or assignment[c] == v for c, v in snode[2])
        ]
        if not agreeing:
            return None
        options.append(agreeing)
    total = Fraction(0)
    for choice in itertools.product(*options):
        weight = Fraction(1)
        for _, prob, tail in choice:
            weight *= prob
            for member in tail:
                if _is_source(member[0]):
                    weight *= priors[member]
        total += weight
    return total
