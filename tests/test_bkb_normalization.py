"""`fuse`'s normalizations: every fragment's weight by default, pybkb's two modes on request.

And the order pybkb's modes sum a component in.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from typing import Any

import pytest

from bko.bkb.errors import FusionError
from bko.bkb.fusion import Normalization, fuse, is_source
from bko.bkb.reason import log_joint_complete
from bko.bkb.serialize import render
from bko.bkb.structure import BKB, INode
from tests._bkb_fixtures import build, h2_fragments
from tests._bkb_reference import reference_mass


pytestmark = pytest.mark.filterwarnings("error")

NOT_A_MODE: Any = "neither"
THIRD = 1 / 3


def priors(bkb: BKB) -> dict[INode, float]:
    return {s.head: s.prob for s in bkb.snodes if is_source(s.head[0])}


def lone() -> list[BKB]:
    """`a` holds S-nodes on Y and X; `c` holds them on Y only."""
    return [h2_fragments()[0], build("c", [("Y", "0", 0.5, []), ("Y", "1", 0.5, [])])]


def disjoint() -> list[BKB]:
    """Two fragments on different components, so no component's total sums both."""
    return [build("a", [("X", "0", 1.0, [])]), build("b", [("Y", "0", 1.0, [])])]


def test_only_the_collapsed_component_mode_collapses():
    assert [mode.collapse for mode in Normalization] == [False, False, True]


def test_a_source_absent_from_a_component_still_counts_in_its_total():
    assert priors(fuse(lone(), [1, 1])) == {
        ("__Source__[Y]", "a"): 0.5,
        ("__Source__[X]", "a"): 0.5,
        ("__Source__[Y]", "c"): 0.5,
    }
    assert priors(fuse(lone(), [1, 1], normalization=Normalization.COMPONENT)) == {
        ("__Source__[Y]", "a"): 0.5,
        ("__Source__[X]", "a"): 1.0,
        ("__Source__[Y]", "c"): 0.5,
    }


def test_each_prior_is_its_reliability_over_every_fragments():
    third = build("c", [("Z", "0", 1.0, [])])
    assert priors(fuse([*h2_fragments(), third], [1, 3, 4])) == {
        ("__Source__[Y]", "a"): 0.125,
        ("__Source__[X]", "a"): 0.125,
        ("__Source__[Y]", "b"): 0.375,
        ("__Source__[X]", "b"): 0.375,
        ("__Source__[Z]", "c"): 0.5,
    }


def test_three_unit_fragments_give_a_third_where_pybkb_gives_a_half():
    fragments = [*disjoint(), build("c", [("X", "0", 1.0, [])])]
    assert priors(fuse(fragments, [1, 1, 1])) == {
        ("__Source__[X]", "a"): THIRD,
        ("__Source__[Y]", "b"): THIRD,
        ("__Source__[X]", "c"): THIRD,
    }
    assert priors(fuse(fragments, [1, 1, 1], normalization=Normalization.COMPONENT)) == {
        ("__Source__[X]", "a"): 0.5,
        ("__Source__[Y]", "b"): 1.0,
        ("__Source__[X]", "c"): 0.5,
    }


def test_scaling_every_reliability_leaves_the_priors():
    assert priors(fuse(lone(), [2, 2])) == priors(fuse(lone(), [1, 1]))


def test_the_default_is_every_fragments_weight():
    explicit = fuse(lone(), [1, 3], normalization=Normalization.FRAGMENTS)
    assert fuse(lone(), [1, 3]) == explicit
    assert explicit != fuse(lone(), [1, 3], normalization=Normalization.COMPONENT)


def test_the_update_agrees_with_the_exact_reference_under_the_default():
    fused = fuse(lone(), [1, 1])
    exact = reference_mass(json.loads(render(fused)), "X", {"Y": "0"})
    got = log_joint_complete(fused, "X", {"Y": "0"})
    assert exact.keys() == got.keys()
    for state, mass in exact.items():
        assert mass is not None
        expected = math.log(mass.numerator) - math.log(mass.denominator)
        assert got[state] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fuse_logs_its_normalization(caplog):
    with caplog.at_level(logging.INFO, logger="bko.bkb.fusion"):
        fuse(list(h2_fragments()), [1, 3])
    assert "fused fragments=2 normalization=fragments inodes=8 snodes=16" in caplog.text


def test_pybkbs_totals_never_sum_fragments_apart():
    assert len(fuse(disjoint(), [1e308, 1e308], normalization=Normalization.COMPONENT).snodes) == 4


def interleaved() -> list[BKB]:
    """`a` and `c` share the S-node `X=x` at 0.5; `b`, between them, holds it at 0.25."""
    return [
        build("a", [("X", "x", 0.5, [])]),
        build("b", [("X", "x", 0.25, [])]),
        build("c", [("X", "x", 0.5, [])]),
    ]


def test_pybkbs_component_total_sums_sources_in_the_order_their_s_nodes_are_met():
    """pybkb sums a component's source priors in the order its S-node loop first meets them.

    pybkb's `build_fused_bkb` (lines 71-81 of `pybkb/fusion/__init__.py` at the commit ADR
    `research/bkb-layer-source` names) walks S-nodes by first appearance and appends each source's
    prior the first time it meets it; `normalize` (lines 122-132) sums them in that order. Here
    that is `a`, `c`, `b`; fragment order is `a`, `b`, `c`, one bit away. No pybkb fixture shares
    an S-node across interleaved fragments, so the expected order comes from reading that source.
    """
    weights = [0.1, 0.7, 0.3]
    in_s_node_order = (0.1 + 0.3) + 0.7
    in_fragment_order = (0.1 + 0.7) + 0.3
    assert in_s_node_order != in_fragment_order, "the control: this case tells the orders apart"
    assert priors(fuse(interleaved(), weights, normalization=Normalization.COMPONENT)) == {
        ("__Source__[X]", "a"): 0.1 / in_s_node_order,
        ("__Source__[X]", "c"): 0.3 / in_s_node_order,
        ("__Source__[X]", "b"): 0.7 / in_s_node_order,
    }
    assert priors(fuse(interleaved(), weights)) == {
        ("__Source__[X]", "a"): 0.1 / in_fragment_order,
        ("__Source__[X]", "b"): 0.7 / in_fragment_order,
        ("__Source__[X]", "c"): 0.3 / in_fragment_order,
    }


REFUSED: dict[str, tuple[Callable[[], BKB], str]] = {
    "an unknown normalization": (
        lambda: fuse(lone(), [1, 1], normalization=NOT_A_MODE),
        "the normalization 'neither' is not one of fragments, component, component-collapsed",
    ),
    "every fragment's weight overflowing where no component's does": (
        lambda: fuse(disjoint(), [1e308, 1e308]),
        "overflow when summed",
    ),
}


@pytest.mark.parametrize(("make", "match"), REFUSED.values(), ids=list(REFUSED))
def test_fuse_refuses(make, match):
    with pytest.raises(FusionError, match=match):
        make()


def test_a_mode_given_by_its_value_is_that_mode():
    assert fuse(lone(), [1, 3], normalization=Normalization("component")) == fuse(
        lone(), [1, 3], normalization=Normalization.COMPONENT
    )
