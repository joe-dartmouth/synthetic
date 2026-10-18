"""Fusion: pybkb's fusion tests ported, its results matched, and the inputs it refuses.

Adapted from pybkb (https://github.com/di2ag/pybkb), by Chase Yakaboski, which is licensed under
the GNU General Public License, version 3. Modified for bko on 2026-10-17; ADR
`research/bkb-layer-source` names the pybkb commit and lists the changes. This file is released
under the GNU General Public License, version 3 only (`LICENSE`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import pytest

from bko.bkb.errors import FusionError
from bko.bkb.fusion import Normalization, fuse, is_source, source_component
from bko.bkb.serialize import render
from bko.bkb.structure import BKB, INode, SNode
from tests._bkb_fixtures import PIRATE, RELIABILITIES, build, exported, h2_fragments


pytestmark = pytest.mark.filterwarnings("error")

TEN = [str(index) for index in range(10)]
COMPONENT = Normalization.COMPONENT
COLLAPSED = Normalization.COMPONENT_COLLAPSED


def pirate() -> list[BKB]:
    return [exported(name) for name in PIRATE]


def priors(bkb: BKB) -> dict[INode, float]:
    return {s.head: s.prob for s in bkb.snodes if is_source(s.head[0])}


def test_pirate_fusion_matches_pybkbs_saved_result():
    fused = fuse(pirate(), [1] * 5, normalization=COMPONENT)
    assert fused.is_mutex()
    assert fused == exported("pirate_scenario_fused")


GOLDENS: dict[str, Callable[[], BKB]] = {
    "pirate_fused_r1": lambda: fuse(pirate(), [1] * 5, normalization=COMPONENT),
    "pirate_fused_mixed": lambda: fuse(pirate(), list(RELIABILITIES), normalization=COMPONENT),
    "pirate_fused_mixed_collapsed": lambda: fuse(
        pirate(), list(RELIABILITIES), normalization=COLLAPSED
    ),
    "goldfish_fused_collapsed": lambda: fuse(
        [exported("goldfish")] * 10, [1] * 10, sources=TEN, normalization=COLLAPSED
    ),
}


@pytest.mark.parametrize(("name", "make"), GOLDENS.items(), ids=list(GOLDENS))
def test_fuse_matches_pybkb_snode_for_snode(name, make):
    fused, golden = make(), exported(name)
    assert fused == golden
    assert fused.snodes == golden.snodes
    assert fused.is_mutex()


def test_goldfish_fusion_without_collapse_gives_each_source_its_own_copy():
    goldfish = exported("goldfish")
    fused = fuse([goldfish] * 10, [1] * 10, sources=TEN, normalization=COMPONENT)
    heads = {snode.head[0] for snode in goldfish.snodes}
    assert len(fused.snodes) == 10 * len(goldfish.snodes) + 10 * len(heads)
    assert set(priors(fused).values()) == {0.1}
    assert fused.is_mutex()


def test_unit_reliabilities_make_fragment_order_irrelevant():
    fragments = pirate()
    assert fuse(fragments, [1] * 5, normalization=COMPONENT) == fuse(
        fragments[::-1], [1] * 5, normalization=COMPONENT
    )


def test_each_source_gets_its_reliability_over_its_components_total():
    plain = fuse(list(h2_fragments()), [1, 3], normalization=COMPONENT)
    assert len(plain.snodes) == 16
    assert priors(plain) == {
        ("__Source__[Y]", "a"): 0.25,
        ("__Source__[X]", "a"): 0.25,
        ("__Source__[Y]", "b"): 0.75,
        ("__Source__[X]", "b"): 0.75,
    }
    assert priors(fuse(list(h2_fragments()), [2, 6], normalization=COMPONENT)) == priors(plain)


def test_collapse_merges_what_sources_share_and_weights_each_set():
    merged = fuse(list(h2_fragments()), [1, 3], normalization=COLLAPSED)
    both = frozenset({"a", "b"})
    assert len(merged.snodes) == 12
    assert priors(merged) == {
        ("__Source__[Y]", both): 1.0,
        ("__Source__[X]", both): 0.5,
        ("__Source__[X]", frozenset({"a"})): 0.125,
        ("__Source__[X]", frozenset({"b"})): 0.375,
    }
    shared = SNode(("X", "0"), 0.75, frozenset({("Y", "0"), ("__Source__[X]", both)}))
    assert shared in merged.snodes


def test_a_source_that_never_heads_a_component_is_left_out_of_its_total():
    a = h2_fragments()[0]
    c = build("c", [("Y", "0", 0.5, []), ("Y", "1", 0.5, [])])
    assert priors(fuse([a, c], [1, 1], normalization=COMPONENT)) == {
        ("__Source__[Y]", "a"): 0.5,
        ("__Source__[X]", "a"): 1.0,
        ("__Source__[Y]", "c"): 0.5,
    }


def test_sources_default_to_fragment_names_and_the_result_is_named_fused():
    fused = fuse(list(h2_fragments()), [1, 3])
    assert fused.name == "fused"
    assert set(fused.states(source_component("X"))) == {"a", "b"}


def test_fuse_leaves_its_fragments_unchanged():
    a, b = h2_fragments()
    before = (render(a), render(b))
    fuse([a, b], [1, 3])
    assert (render(a), render(b)) == before


def test_fuse_logs_one_summary_line(caplog):
    with caplog.at_level(logging.INFO, logger="bko.bkb.fusion"):
        fuse(list(h2_fragments()), [1, 3], normalization=COMPONENT)
    assert "fused fragments=2 normalization=component inodes=8 snodes=16" in caplog.text


def test_is_source_matches_only_the_names_fuse_writes():
    assert is_source("__Source__[X]")
    assert is_source(source_component("a component [with brackets]"))
    assert not is_source("my__Source__[x]")
    assert not is_source("__Source__X")
    assert not is_source("X")


NOT_A_NAME: Any = 1


def repeated_snode() -> BKB:
    return build("repeated", [("Y", "0", 0.5, []), ("Y", "0", 0.5, [])])


def fused_h2() -> BKB:
    return fuse(list(h2_fragments()), [1, 1])


REFUSED: dict[str, Callable[[], BKB]] = {
    "no fragments": lambda: fuse([], []),
    "too few reliabilities": lambda: fuse(list(h2_fragments()), [1]),
    "too many sources": lambda: fuse(list(h2_fragments()), [1, 1], sources=["a", "b", "c"]),
    "a zero reliability": lambda: fuse(list(h2_fragments()), [1, 0]),
    "a negative reliability": lambda: fuse(list(h2_fragments()), [1, -1]),
    "a NaN reliability": lambda: fuse(list(h2_fragments()), [1, math.nan]),
    "an infinite reliability": lambda: fuse(list(h2_fragments()), [1, math.inf]),
    "a boolean reliability": lambda: fuse(list(h2_fragments()), [1, True]),
    "an integer reliability beyond float range": lambda: fuse(list(h2_fragments()), [10**400, 1]),
    "a negative integer reliability beyond float range": lambda: fuse(
        list(h2_fragments()), [-(10**400), 1]
    ),
    "an integer reliability too long to print": lambda: fuse(list(h2_fragments()), [10**5000, 1]),
    "a repeated source name": lambda: fuse(list(h2_fragments()), [1, 1], sources=["a", "a"]),
    "a source name that is not a string": lambda: fuse(
        list(h2_fragments()), [1, 1], sources=["a", NOT_A_NAME]
    ),
    "an empty source name": lambda: fuse(list(h2_fragments()), [1, 1], sources=["", "b"]),
    "source names as one string": lambda: fuse(list(h2_fragments()), [1, 1], sources="ab"),
    "a fragment holding one S-node twice": lambda: fuse([repeated_snode()], [1]),
    "reliabilities that overflow when summed": lambda: fuse(
        list(h2_fragments()), [1e308, 1e308], normalization=COMPONENT
    ),
    "reliabilities that overflow when collapsed": lambda: fuse(
        list(h2_fragments()), [1e308, 1e308], normalization=COLLAPSED
    ),
    "fragments sharing a name": lambda: fuse([h2_fragments()[0]] * 2, [1, 1]),
    "a fused fragment": lambda: fuse([fused_h2()], [1]),
}


@pytest.mark.parametrize("make", REFUSED.values(), ids=list(REFUSED))
def test_fuse_refuses(make):
    with pytest.raises(FusionError):
        make()


def test_a_repeated_source_name_is_named():
    with pytest.raises(FusionError, match="the source name 'a'"):
        fuse(list(h2_fragments()), [1, 1], sources=["a", "a"])


def test_an_overflowing_reliability_is_named():
    with pytest.raises(FusionError, match="a reliability is an integer that overflows a float"):
        fuse(list(h2_fragments()), [1, 10**5000])
