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


def two_fragments() -> list[BKB]:
    """`a` and `b` both hold `X` at 0.5; only `b` holds `Y`, so `X`'s component has every source."""
    a, b = BKB("a"), BKB("b")
    for fragment in (a, b):
        fragment.add_inode("X", "yes")
        fragment.add_snode("X", "yes", 0.5)
    b.add_inode("Y", "yes")
    b.add_snode("Y", "yes", 0.3)
    return [a, b]


def resourced(fused: BKB, component: str, state: str | frozenset[str]) -> BKB:
    """`fused` with the source state of its S-node on `(component, "yes")` replaced by `state`."""
    first = next(i for i, s in enumerate(fused.snodes) if s.head == (component, "yes"))
    snode = fused.snodes[first]
    own = source_component(component)
    tail = frozenset(n for n in snode.tail if n[0] != own) | {(own, state)}
    copy = rebuilt(fused, lambda index, _: index != first, [])
    if (own, state) not in copy.inodes:
        copy.add_inode(own, state)
    copy.add_snode(component, "yes", snode.prob, tail)
    return copy


def halved(fused: BKB, component: str) -> BKB:
    """`fused` with the first prior on `component`'s source component halved."""
    own = source_component(component)
    prior = next(i for i, s in enumerate(fused.snodes) if s.head[0] == own and not s.tail)
    snode = fused.snodes[prior]
    return rebuilt(
        fused, lambda index, _: index != prior, [SNode(snode.head, snode.prob / 2, frozenset())]
    )


@pytest.mark.parametrize("normalization", list(Normalization))
def test_a_fusion_under_unequal_reliabilities_passes_the_check(normalization):
    for every in fragments(plant(KINDS[0])).values():
        reliabilities = [float(index + 1) for index in range(len(every))]
        fused = fuse(every, reliabilities, normalization=normalization)
        check_fused(fused, every, normalization=normalization)
    every = two_fragments()
    check_fused(
        fuse(every, [1.0, 3.0], normalization=normalization), every, normalization=normalization
    )


@pytest.mark.parametrize(
    ("normalization", "component"),
    [
        (Normalization.COMPONENT, "Y"),
        (Normalization.COMPONENT_COLLAPSED, "Y"),
        (Normalization.FRAGMENTS, "X"),
    ],
)
def test_priors_summing_short_of_1_fail_the_check_where_they_must_be_whole(
    normalization, component
):
    every = two_fragments()
    fused = fuse(every, [1.0, 3.0], normalization=normalization)
    with pytest.raises(FusionError, match=r"the priors on .* sum to .*, not 1$"):
        check_fused(halved(fused, component), every, normalization=normalization)


def test_priors_short_of_1_pass_under_fragments_where_a_fragment_lacks_the_component():
    every = two_fragments()
    fused = fuse(every, [1.0, 3.0])
    check_fused(halved(fused, "Y"), every)


def test_a_source_state_naming_a_source_without_the_input_fails_the_check():
    every = two_fragments()
    fused = fuse(every, [1.0, 1.0], normalization=Normalization.COMPONENT_COLLAPSED)
    tampered = resourced(fused, "Y", frozenset({"a", "b"}))
    with pytest.raises(FusionError, match="maps to no input S-node of \\['a', 'b'\\]"):
        check_fused(tampered, every, normalization=Normalization.COMPONENT_COLLAPSED)


def test_a_shared_s_node_missing_one_of_its_sources_fails_the_check():
    every = two_fragments()
    fused = fuse(every, [1.0, 1.0], normalization=Normalization.COMPONENT_COLLAPSED)
    tampered = resourced(fused, "X", "a")
    with pytest.raises(FusionError, match=r"lacks S-node on \('X', 'yes'\) at 0.5 from \['b'\]"):
        check_fused(tampered, every, normalization=Normalization.COMPONENT_COLLAPSED)


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        (["a"], "2 fragments and 1 source names"),
        (["a", "b", "c"], "2 fragments and 3 source names"),
        (["a", "a"], "two fragments share the source name 'a'"),
        ("ab", "are one string"),
        (["a", ""], "the source name '' is not a non-empty string"),
        (["a", 7], "the source name 7 is not a non-empty string"),
    ],
)
def test_source_names_fuse_refuses_fail_the_check(sources, message):
    every = two_fragments()
    fused = fuse(every, [1.0, 1.0])
    with pytest.raises(FusionError, match=message):
        check_fused(fused, every, sources=sources)


def unprimed(fused: BKB, component: str) -> BKB:
    """`fused` with every prior on `component`'s source component dropped."""
    own = source_component(component)
    return rebuilt(fused, lambda _, s: not (s.head[0] == own and not s.tail), [])


@pytest.mark.parametrize(
    ("normalization", "component"),
    [
        (Normalization.COMPONENT, "Y"),
        (Normalization.COMPONENT_COLLAPSED, "X"),
        (Normalization.FRAGMENTS, "X"),
    ],
)
def test_a_named_source_component_with_no_prior_fails_where_its_priors_must_be_whole(
    normalization, component
):
    every = two_fragments()
    fused = fuse(every, [1.0, 3.0], normalization=normalization)
    with pytest.raises(FusionError, match=r"the priors on .* sum to 0\.0, not 1$"):
        check_fused(unprimed(fused, component), every, normalization=normalization)


def test_a_named_source_component_with_no_prior_passes_under_fragments_where_one_lacks_it():
    every = two_fragments()
    check_fused(unprimed(fuse(every, [1.0, 3.0]), "Y"), every)


@pytest.mark.parametrize(
    ("normalization", "state"),
    [
        (Normalization.FRAGMENTS, "z"),
        (Normalization.COMPONENT_COLLAPSED, frozenset({"a", "z"})),
    ],
)
def test_a_prior_on_a_state_naming_no_known_source_fails_the_check(normalization, state):
    every = two_fragments()
    fused = fuse(every, [1.0, 1.0], normalization=normalization)
    own = source_component("X")
    tampered = rebuilt(fused, lambda *_: True, [])
    tampered.add_inode(own, state)
    tampered.add_snode(own, state, 0.0)
    with pytest.raises(FusionError, match=r"the prior on .* names other than the sources"):
        check_fused(tampered, every, normalization=normalization)
