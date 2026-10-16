"""Builders and the committed pybkb fixtures for the `bko.bkb` tests.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-16; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).

The JSON under `tests/pybkb/` was written by `tests/pybkb/export.py` from pybkb at the commit its
manifest names (`docs/DEVELOPMENT.md`, BKB fixtures). Files are read once per process and parsed
per call, so no two tests share a mutable BKB.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from bko.bkb.serialize import parse
from bko.bkb.structure import BKB, INode, State
from tests._corpus import REPO_ROOT


PYBKB_DIR = REPO_ROOT / "tests" / "pybkb"
"""Where the exporter's output is committed."""

NOT_DOCUMENTS = frozenset({"manifest", "queries", "run"})


@functools.cache
def fixture_text(name: str) -> str:
    return (PYBKB_DIR / f"{name}.json").read_text(encoding="utf-8")


def document_names() -> list[str]:
    return sorted(p.stem for p in PYBKB_DIR.glob("*.json") if p.stem not in NOT_DOCUMENTS)


def manifest() -> dict[str, Any]:
    return json.loads(fixture_text("manifest"))


PIRATE = ("fisherman", "illegal_dumping_ev", "illegal_fishing_ev", "pirate", "tsunami_ev")
"""pybkb's `test_pirate_fusion` fragments, in its order."""

RELIABILITIES = (0.5, 0.8, 0.2, 0.3, 0.9)
"""The exporter's non-uniform reliabilities for the pirate fragments."""


def exported(name: str) -> BKB:
    """The BKB the exporter wrote under `name`, parsed afresh."""
    return parse(fixture_text(name))


@dataclass(frozen=True)
class Query:
    bkb: str
    target: str
    evidence: Mapping[str, str]
    result: Mapping[str, float | None]


def exported_queries() -> list[Query]:
    return [
        Query(
            item["bkb"],
            item["target"],
            item["evidence"],
            {k: -math.inf if v == "-inf" else v for k, v in item["result"].items()},
        )
        for item in json.loads(fixture_text("queries"))
    ]


def build(name: str, snodes: Iterable[tuple[str, State, float, Iterable[INode]]]) -> BKB:
    """A BKB holding the I-nodes the S-nodes name, in order, then the S-nodes in order."""
    bkb = BKB(name)
    rows = [(component, state, prob, list(tail)) for component, state, prob, tail in snodes]
    for component, state, _, tail in rows:
        bkb.add_inode(component, state)
        for member in tail:
            bkb.add_inode(*member)
    for component, state, prob, tail in rows:
        bkb.add_snode(component, state, prob, tail)
    return bkb


def simple_build() -> BKB:
    """pybkb's `test_simple_build`, with its integer names as strings."""
    bkb = BKB("simple")
    for component, state in [("A", "1"), ("A", "2"), ("1", "1"), ("1", "B")]:
        bkb.add_inode(component, state)
    bkb.add_snode("A", "1", 0.003)
    bkb.add_snode("1", "1", 0.9, [("A", "1")])
    bkb.add_snode("1", "B", 0.56)
    bkb.add_snode("A", "2", 0.97, [("1", "1")])
    return bkb


def h2_fragments() -> tuple[BKB, BKB]:
    """Two fragments that share Y and X given Y=0, and differ on X given Y=1."""
    shared = [
        ("Y", "0", 0.5, []),
        ("Y", "1", 0.5, []),
        ("X", "0", 0.75, [("Y", "0")]),
        ("X", "1", 0.25, [("Y", "0")]),
    ]
    a = build("a", [*shared, ("X", "0", 0.25, [("Y", "1")]), ("X", "1", 0.75, [("Y", "1")])])
    b = build("b", [*shared, ("X", "0", 0.75, [("Y", "1")]), ("X", "1", 0.25, [("Y", "1")])])
    return a, b


def h1_fragments() -> tuple[BKB, BKB]:
    """Two fragments over Y and X given Y that share no S-node, with dyadic probabilities."""
    a = build(
        "a",
        [
            ("Y", "0", 0.5, []),
            ("Y", "1", 0.5, []),
            ("X", "0", 0.75, [("Y", "0")]),
            ("X", "1", 0.25, [("Y", "0")]),
            ("X", "0", 0.25, [("Y", "1")]),
            ("X", "1", 0.75, [("Y", "1")]),
        ],
    )
    b = build(
        "b",
        [
            ("Y", "0", 0.25, []),
            ("Y", "1", 0.75, []),
            ("X", "0", 0.5, [("Y", "0")]),
            ("X", "1", 0.5, [("Y", "0")]),
            ("X", "0", 0.5, [("Y", "1")]),
            ("X", "1", 0.5, [("Y", "1")]),
        ],
    )
    return a, b
