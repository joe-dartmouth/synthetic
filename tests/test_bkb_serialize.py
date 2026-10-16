"""The BKB document: lossless round trips, canonical fixtures, and the documents it refuses."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from bko.bkb.errors import DocumentFormatError, StructureError
from bko.bkb.serialize import parse, render
from bko.bkb.structure import BKB
from tests._bkb_fixtures import build, document_names, exported, fixture_text, simple_build


pytestmark = pytest.mark.filterwarnings("error")


def collapsed() -> BKB:
    return build(
        "collapsed",
        [
            ("Y", "0", 0.5, []),
            ("X", "0", 0.1 + 0.2, [("Y", "0"), ("__Source__[X]", frozenset({"b", "a"}))]),
            ("__Source__[X]", frozenset({"a", "b"}), 1.0, []),
        ],
    )


def no_snodes() -> BKB:
    bkb = BKB("bare")
    bkb.add_inode("A", "1")
    return bkb


EXAMPLES: dict[str, Callable[[], BKB]] = {
    "simple": simple_build,
    "collapsed": collapsed,
    "empty": lambda: BKB("empty"),
    "no S-nodes": no_snodes,
    "described": lambda: BKB("described", "a description"),
    "unicode": lambda: build("名前", [("Ünïcode [x]", "état", 0.9057551260540134, [])]),
    "tiny": lambda: build("tiny", [("A", "1", 5e-324, []), ("A", "2", 0.0, [])]),
}


@pytest.mark.parametrize("make", EXAMPLES.values(), ids=list(EXAMPLES))
def test_a_rendered_bkb_reads_back_equal_in_order_and_renders_the_same(make):
    bkb = make()
    back = parse(render(bkb))
    assert back == bkb
    assert (back.inodes, back.snodes) == (bkb.inodes, bkb.snodes)
    assert (back.name, back.description) == (bkb.name, bkb.description)
    assert render(back) == render(bkb)


@pytest.mark.parametrize("name", document_names())
def test_every_committed_document_is_canonical(name):
    text = fixture_text(name)
    assert render(parse(text)) == text


def test_the_json_fixture_holds_pybkbs_json_probabilities():
    snodes = exported("test_json").snodes
    assert {(s.head, s.prob) for s in snodes} >= {
        (("RV_1", "2"), 0.9057551260540134),
        (("RV_2", "0"), 0.961779626127865),
        (("RV_3", "0"), 0.3706564149505228),
    }
    tails = {s.tail for s in snodes if s.head == ("RV_3", "0")}
    assert frozenset({("RV_2", "0"), ("RV_1", "2")}) in tails


def test_render_writes_the_documented_shape():
    text = render(simple_build())
    assert json.loads(text) == {
        "description": None,
        "inodes": [["A", "1"], ["A", "2"], ["1", "1"], ["1", "B"]],
        "name": "simple",
        "snodes": [
            {"head": 0, "prob": 0.003, "tail": []},
            {"head": 2, "prob": 0.9, "tail": [0]},
            {"head": 3, "prob": 0.56, "tail": []},
            {"head": 1, "prob": 0.97, "tail": [2]},
        ],
    }
    assert text.endswith("}\n")


def document(**changes: Any) -> str:
    base = json.loads(render(simple_build()))
    base.update(changes)
    return json.dumps(base)


def with_snode(**changes: Any) -> str:
    base = json.loads(render(simple_build()))
    base["snodes"][1].update(changes)
    return json.dumps(base)


def without(key: str) -> str:
    base = json.loads(render(simple_build()))
    del base[key]
    return json.dumps(base)


MALFORMED: dict[str, Callable[[], str]] = {
    "not JSON": lambda: "{",
    "a NaN": lambda: render(simple_build()).replace("0.003", "NaN"),
    "an infinity": lambda: render(simple_build()).replace("0.003", "-Infinity"),
    "an overflowing number": lambda: render(simple_build()).replace("0.003", "1e400"),
    "a repeated key": lambda: render(simple_build()).replace(
        '"name": "simple"', '"name": "simple", "name": "again"'
    ),
    "a list": lambda: "[]",
    "a missing key": lambda: without("snodes"),
    "an extra key": lambda: document(version=1),
    "a numeric name": lambda: document(name=1),
    "a numeric description": lambda: document(description=2),
    "I-nodes that are not a list": lambda: document(inodes={}),
    "S-nodes that are not a list": lambda: document(snodes={}),
    "an I-node of three": lambda: document(inodes=[["A", "1", "x"]], snodes=[]),
    "an integer state": lambda: document(inodes=[["A", 1]], snodes=[]),
    "an unsorted source set": lambda: document(inodes=[["S", ["b", "a"]]], snodes=[]),
    "a repeated source name": lambda: document(inodes=[["S", ["a", "a"]]], snodes=[]),
    "an empty source set": lambda: document(inodes=[["S", []]], snodes=[]),
    "a repeated I-node": lambda: document(inodes=[["A", "1"], ["A", "1"]], snodes=[]),
    "an S-node that is not an object": lambda: document(snodes=[[0, 0.5, []]]),
    "an S-node with an extra key": lambda: with_snode(note="x"),
    "a boolean head": lambda: with_snode(head=True),
    "a head out of range": lambda: with_snode(head=4),
    "a negative tail index": lambda: with_snode(tail=[-1]),
    "an unsorted tail": lambda: with_snode(tail=[1, 0]),
    "a boolean probability": lambda: with_snode(prob=True),
    "a string probability": lambda: with_snode(prob="0.9"),
    "a probability above one": lambda: with_snode(prob=1.5),
    "a tail on the head's component": lambda: with_snode(tail=[3]),
}


@pytest.mark.parametrize("make", MALFORMED.values(), ids=list(MALFORMED))
def test_parse_refuses(make):
    with pytest.raises(DocumentFormatError):
        parse(make())


def test_a_broken_bkb_rule_is_reported_with_its_cause():
    with pytest.raises(DocumentFormatError) as caught:
        parse(with_snode(prob=1.5))
    assert isinstance(caught.value.__cause__, StructureError)
