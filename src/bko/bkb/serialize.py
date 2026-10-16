"""A BKB as a JSON document: what the committed fixtures hold and `render` writes.

The document is one object with exactly the keys `name` (a string), `description` (a string or
null), `inodes` and `snodes`. `inodes` lists the I-nodes in the BKB's order, each
`[component, state]`, a collapsed source state written as the sorted array of its source names.
`snodes` lists the S-nodes in the BKB's order, each `{"head": i, "prob": p, "tail": [j, ...]}`,
where `i` and the strictly ascending `j` index `inodes`.

`render` writes sorted keys, a two-space indent, no NaN or infinity and a final newline, so a
document it wrote parses back to an equal BKB in the same order and renders to the same bytes.
`parse` refuses text that is not JSON, holds NaN, an infinite number or a repeated key, or breaks
the shape above or a rule of `bko.bkb.structure.BKB`.
"""

from __future__ import annotations

import json
import math
from typing import Final, TypeGuard

from bko.bkb.errors import DocumentFormatError, StructureError
from bko.bkb.structure import BKB, INode, State


KEYS: Final = frozenset({"description", "inodes", "name", "snodes"})
"""The document's keys."""

SNODE_KEYS: Final = frozenset({"head", "prob", "tail"})
"""Each S-node's keys."""


def render(bkb: BKB) -> str:
    """Write `bkb` as a JSON document, in the BKB's own order."""
    position = {inode: index for index, inode in enumerate(bkb.inodes)}
    document = {
        "description": bkb.description,
        "inodes": [[component, _state_value(state)] for component, state in bkb.inodes],
        "name": bkb.name,
        "snodes": [
            {"head": position[s.head], "prob": s.prob, "tail": sorted(position[m] for m in s.tail)}
            for s in bkb.snodes
        ],
    }
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse(text: str) -> BKB:
    """Read a JSON document back into a BKB.

    Raises:
        DocumentFormatError: If `text` is not a document of the module docstring's shape, or it
            breaks a rule of `bko.bkb.structure.BKB`.
    """
    document = _load(text)
    try:
        return _build(document)
    except StructureError as exc:
        msg = f"the document breaks a BKB rule: {exc}"
        raise DocumentFormatError(msg) from exc


def _state_value(state: State) -> str | list[str]:
    return state if isinstance(state, str) else sorted(state)


def _load(text: str) -> dict[str, object]:
    try:
        document = json.loads(
            text, object_pairs_hook=_unique, parse_constant=_constant, parse_float=_finite
        )
    except (ValueError, RecursionError) as exc:
        msg = f"the document is not strict JSON: {exc}"
        raise DocumentFormatError(msg) from exc
    if not isinstance(document, dict) or set(document) != KEYS:
        msg = f"the document is not an object with exactly the keys {sorted(KEYS)}"
        raise DocumentFormatError(msg)
    return document


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        msg = f"an object repeats a key: {keys}"
        raise ValueError(msg)
    return dict(pairs)


def _constant(name: str) -> float:
    msg = f"{name} is not a JSON number"
    raise ValueError(msg)


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        msg = f"{text} is not a finite number"
        raise ValueError(msg)
    return value


def _build(document: dict[str, object]) -> BKB:
    name, description = document["name"], document["description"]
    if not isinstance(name, str) or not (description is None or isinstance(description, str)):
        msg = "the name is not a string, or the description is neither a string nor null"
        raise DocumentFormatError(msg)
    bkb = BKB(name, description)
    items = _list(document["inodes"], "inodes")
    for item in items:
        bkb.add_inode(*_inode(item))
    order = bkb.inodes
    if len(order) != len(items):
        msg = "the document lists an I-node twice"
        raise DocumentFormatError(msg)
    for item in _list(document["snodes"], "snodes"):
        head, prob, tail = _snode(item, len(order))
        bkb.add_snode(order[head][0], order[head][1], prob, [order[j] for j in tail])
    return bkb


def _list(value: object, key: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"`{key}` is not an array"
        raise DocumentFormatError(msg)
    return value


def _inode(item: object) -> INode:
    if isinstance(item, list) and len(item) == 2:
        component, state = item
        if isinstance(component, str) and isinstance(state, str):
            return component, state
        if isinstance(component, str) and _is_source_names(state):
            return component, frozenset(state)
    msg = f"{item!r} is not an I-node: [component, state or sorted source names]"
    raise DocumentFormatError(msg)


def _snode(item: object, count: int) -> tuple[int, float, list[int]]:
    if isinstance(item, dict) and set(item) == SNODE_KEYS:
        head, prob, tail = item["head"], item["prob"], item["tail"]
        if _is_index(head, count) and _is_number(prob) and _is_tail(tail, count):
            return head, float(prob), tail
    msg = f"{item!r} is not an S-node: head, prob and an ascending tail of I-node indices"
    raise DocumentFormatError(msg)


def _is_source_names(value: object) -> TypeGuard[list[str]]:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(name, str) for name in value)
        and value == sorted(set(value))
    )


def _is_index(value: object, count: int) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < count


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_tail(value: object, count: int) -> TypeGuard[list[int]]:
    return (
        isinstance(value, list)
        and all(_is_index(j, count) for j in value)
        and value == sorted(set(value))
    )
