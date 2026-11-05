"""Fusion keeps both sources' S-nodes under each planted data contradiction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from bko.bkb.errors import FusionError
from bko.bkb.fusion import Normalization, check_fused, fuse, is_source, source_component
from bko.bkb.structure import BKB, SNode
from bko.datasets.uci import UCI_FRAME
from bko.gaps.detect import ontology_fragments
from bko.gaps.observe import Cell, dataset_fragment, observe_records, record_cells
from tests._contradiction_fixtures import KINDS, NAME, Planted, plant


def fragments(planted: Planted) -> dict[Cell, list[BKB]]:
    """Per cell: the ontology fragments, the cell's dataset fragment and its label group's."""
    cells = record_cells(planted.frame, planted.rows, name=NAME, edges=UCI_FRAME.edges)
    ontology = ontology_fragments(planted.ontologies, {row.concept for row in planted.rows})
    found: dict[Cell, list[BKB]] = {}
    for cell, records in cells.each.items():
        members = planted.frame.loc[planted.frame[UCI_FRAME.label].eq(cell[2])]
        group = observe_records(members, planted.rows, name=f"{NAME}/group")
        found[cell] = [*ontology, dataset_fragment(records.observed), dataset_fragment(group)]
    return found


def rebuilt(fused: BKB, keep: Callable[[int, SNode], bool], extra: list[SNode]) -> BKB:
    """A copy of `fused` holding the S-nodes `keep` admits, then `extra`."""
    copy = BKB(fused.name)
    for component, state in fused.inodes:
        copy.add_inode(component, state)
    kept = [snode for index, snode in enumerate(fused.snodes) if keep(index, snode)]
    for snode in [*kept, *extra]:
        copy.add_snode(snode.head[0], snode.head[1], snode.prob, snode.tail)
    return copy


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("normalization", list(Normalization))
def test_every_cells_fusion_passes_the_check(kind, normalization):
    for every in fragments(plant(kind)).values():
        fused = fuse(every, [1.0] * len(every), normalization=normalization)
        check_fused(fused, every, normalization=normalization)


@pytest.mark.parametrize("kind", KINDS)
def test_fusion_keeps_both_sources_s_nodes_with_unchanged_probabilities(kind):
    for every in fragments(plant(kind)).values():
        fused = fuse(every, [1.0] * len(every))
        held = {(s.head, s.tail, s.prob) for s in fused.snodes}
        for fragment in every:
            for snode in fragment.snodes:
                tail = snode.tail | {(source_component(snode.head[0]), fragment.name)}
                assert (snode.head, tail, snode.prob) in held


def _tampered(fused: BKB, how: str) -> BKB:
    first = next(i for i, s in enumerate(fused.snodes) if not is_source(s.head[0]))
    snode = fused.snodes[first]
    match how:
        case "dropped":
            return rebuilt(fused, lambda index, _: index != first, [])
        case "changed":
            changed = SNode(snode.head, snode.prob / 2, snode.tail)
            return rebuilt(fused, lambda index, _: index != first, [changed])
        case "unsourced":
            bare = frozenset(n for n in snode.tail if not is_source(n[0]))
            return rebuilt(fused, lambda *_: True, [SNode(snode.head, snode.prob, bare)])
        case _:
            prior = next(
                i
                for i, s in enumerate(fused.snodes)
                if is_source(s.head[0]) and len(fused.states(s.head[0])) > 1
            )
            head = fused.snodes[prior].head
            return rebuilt(fused, lambda index, _: index != prior, [SNode(head, 1.0, frozenset())])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("how", "message"),
    [
        ("dropped", "lacks S-node"),
        ("changed", "maps to no input S-node"),
        ("unsourced", "maps to no input S-node"),
        ("prior", r"the priors on .* sum to"),
    ],
)
def test_a_tampered_fusion_fails_the_check(kind, how, message):
    every = fragments(plant(kind))["<55", "male", 1]
    fused = fuse(every, [1.0] * len(every))
    with pytest.raises(FusionError, match=message):
        check_fused(_tampered(fused, how), every)
