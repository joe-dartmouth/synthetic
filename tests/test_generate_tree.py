"""The ancestor-shrunk shares of `bko.generate.tree` and `bko-tree`'s draw of a tree field.

The hand tree is `top -> {R, unmapped}`, `R -> {A, B}`, `A -> {a1, a2}`, `B -> {b}`; its shares
are worked by hand from the rule of ADR `research/ancestor-shrunk-shares` at `GROUP_WEIGHT` 20.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Final

import pytest

from bko.datasets.cells import LABEL
from bko.datasets.uci import UCI_FRAME
from bko.gaps.shares import GROUP_WEIGHT
from bko.generate.tree import (
    TOP,
    UNMAPPED,
    code_tree,
    leaf_concept,
    leaves,
    tree_fields,
    tree_shares,
)
from bko.ontology.obo import parse_obo
from bko.ontology.schema import AtLeast, Codes, FieldMapping
from bko.ontology.terms import Ontology


TREE: Final = {
    "R": TOP,
    UNMAPPED: TOP,
    "A": "R",
    "B": "R",
    "a1": "A",
    "a2": "A",
    "b": "B",
}
"""Child to parent: two branches under one root, beside the unmapped leaf."""

CELL: Final = {"a1": 4, "b": 2, UNMAPPED: 4}
GROUP: Final = {"a1": 10, "a2": 6, "b": 4, UNMAPPED: 20}

HAND: Final = {
    "a1": Fraction(11, 39),
    "a2": Fraction(5, 39),
    "b": Fraction(8, 65),
    UNMAPPED: Fraction(7, 15),
}
"""Each leaf's share: `r(R|top) = 16/30`, `r(A|R) = 20/26`, `r(a1|A) = 16.5/24`, `r(b|B) = 1`."""


def _row(column: str, *codes: int, concept: str = "MONDO:9000001") -> FieldMapping:
    return FieldMapping(
        when=(Codes(column, frozenset(codes)),),
        concept=concept,
        label="test",
        source="test",
        limits="test",
    )


def _diamond(tmp_path: Path) -> dict[str, Ontology]:
    """`MONDO:9000004` under both `MONDO:9000002` and `MONDO:9000003`, and `MONDO:9000005` under
    `MONDO:9000003`, all under `MONDO:9000001`."""
    stanzas = [
        ("MONDO:9000001", ()),
        ("MONDO:9000002", ("MONDO:9000001",)),
        ("MONDO:9000003", ("MONDO:9000001",)),
        ("MONDO:9000004", ("MONDO:9000002", "MONDO:9000003")),
        ("MONDO:9000005", ("MONDO:9000003",)),
    ]
    lines = ["format-version: 1.2", "data-version: releases/2026-01-01", "ontology: mondo", ""]
    for curie, parents in stanzas:
        lines += ["[Term]", f"id: {curie}", f"name: term {curie[-1]}"]
        lines += [f"is_a: {parent}" for parent in parents]
        lines.append("")
    path = tmp_path / "mondo.obo"
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"MONDO": parse_obo(path, name="mondo")}


def test_the_weight_is_the_one_the_hand_values_read():
    assert GROUP_WEIGHT == 20


def test_the_shares_are_the_hand_values():
    shares, _ = tree_shares(CELL, GROUP, TREE, GROUP_WEIGHT)
    assert {leaf: shares[leaf] for leaf in HAND} == {
        leaf: float(value) for leaf, value in HAND.items()
    }


def test_a_code_unseen_in_the_cell_gets_a_share_above_zero_and_below_its_parents():
    shares, parts = tree_shares(CELL, GROUP, TREE, GROUP_WEIGHT)
    assert CELL.get("a2", 0) == 0
    assert 0 < shares["a2"] < shares["A"]
    assert {node: shares[node] for node in ("a2", "A")} == {
        "a2": float(Fraction(5, 39)),
        "A": float(Fraction(16, 39)),
    }
    (a2,) = [part for part in parts if part.node == "a2"]
    assert (a2.parent, a2.cell, a2.parent_cell, a2.group, a2.parent_group) == ("A", 0, 4, 6, 16)
    assert (a2.weight, a2.conditional, a2.share) == (20, float(Fraction(5, 16)), shares["a2"])


@pytest.mark.parametrize("cell", [CELL, {}], ids=["cell", "empty-cell"])
def test_the_shares_of_a_field_sum_to_one(cell):
    shares, _ = tree_shares(cell, GROUP, TREE, GROUP_WEIGHT)
    assert leaves(TREE) == ("a1", "a2", "b", UNMAPPED)
    assert math.isclose(math.fsum(shares[leaf] for leaf in leaves(TREE)), 1.0, rel_tol=1e-15)
    assert {TOP: shares[TOP]} == {TOP: 1.0}


def test_an_empty_cell_gives_the_label_groups_shares():
    shares, _ = tree_shares({}, GROUP, TREE, GROUP_WEIGHT)
    assert {leaf: shares[leaf] for leaf in GROUP} == {
        leaf: count / sum(GROUP.values()) for leaf, count in GROUP.items()
    }


def test_a_chain_of_single_children_changes_no_share():
    chained = {**TREE, "A": "X", "X": "Y", "Y": "R"}
    plain, _ = tree_shares(CELL, GROUP, TREE, GROUP_WEIGHT)
    shares, parts = tree_shares(CELL, GROUP, chained, GROUP_WEIGHT)
    assert {leaf: shares[leaf] for leaf in HAND} == {leaf: plain[leaf] for leaf in HAND}
    assert {part.node: part.conditional for part in parts if part.node in {"X", "Y", "A"}} == {
        "Y": float(Fraction(10, 13)),
        "X": 1.0,
        "A": 1.0,
    }


def test_each_part_names_its_node_and_parent_in_depth_first_order():
    _, parts = tree_shares(CELL, GROUP, TREE, GROUP_WEIGHT)
    assert [(part.node, part.parent) for part in parts] == [
        ("R", TOP),
        ("A", "R"),
        ("a1", "A"),
        ("a2", "A"),
        ("B", "R"),
        ("b", "B"),
        (UNMAPPED, TOP),
    ]


def test_a_node_with_two_parents_keeps_the_one_with_more_mapped_leaves(tmp_path):
    ontologies = _diamond(tmp_path)
    tree = code_tree({"MONDO:9000004", "MONDO:9000005"}, ontologies)
    assert dict(tree) == {
        "MONDO:9000004": "MONDO:9000003",
        "MONDO:9000005": "MONDO:9000003",
        "MONDO:9000003": "MONDO:9000001",
        "MONDO:9000001": TOP,
        UNMAPPED: TOP,
    }
    tied = code_tree({"MONDO:9000004"}, ontologies)
    assert dict(tied) == {
        "MONDO:9000004": "MONDO:9000002",
        "MONDO:9000002": "MONDO:9000001",
        "MONDO:9000001": TOP,
        UNMAPPED: TOP,
    }


def test_a_mapped_concept_above_another_gets_its_own_leaf(tmp_path):
    tree = code_tree({"MONDO:9000003", "MONDO:9000005"}, _diamond(tmp_path))
    assert dict(tree) == {
        "MONDO:9000003/self": "MONDO:9000003",
        "MONDO:9000005": "MONDO:9000003",
        "MONDO:9000003": "MONDO:9000001",
        "MONDO:9000001": TOP,
        UNMAPPED: TOP,
    }
    assert leaves(tree) == ("MONDO:9000003/self", "MONDO:9000005", UNMAPPED)
    assert [leaf_concept(leaf) for leaf in leaves(tree)] == [
        "MONDO:9000003",
        "MONDO:9000005",
        None,
    ]


def test_a_concept_its_release_lacks_is_a_root():
    assert dict(code_tree({"MONDO:9000009", "HP:9000001"}, {})) == {
        "HP:9000001": TOP,
        "MONDO:9000009": TOP,
        UNMAPPED: TOP,
    }


def test_a_column_of_disjoint_single_clause_code_rows_is_a_tree_field():
    rows = (_row("cp", 1), _row("cp", 2, 3, concept="MONDO:9000002"), _row("fbs", 1))
    assert tree_fields(rows, UCI_FRAME) == ("cp",)


def test_a_field_whose_rows_overlap_is_not_a_tree_field():
    rows = (_row("cp", 1, 2), _row("cp", 2, concept="MONDO:9000002"))
    assert tree_fields(rows, UCI_FRAME) == ()


def test_a_field_with_a_single_row_is_not_a_tree_field():
    assert tree_fields((_row("cp", 1),), UCI_FRAME) == ()


def test_a_field_read_by_a_two_clause_or_threshold_row_is_not_a_tree_field():
    two = FieldMapping(
        when=(Codes("cp", frozenset({3})), Codes("fbs", frozenset({1}))),
        concept="MONDO:9000003",
        label="test",
        source="test",
        limits="test",
    )
    assert tree_fields((_row("cp", 1), _row("cp", 2), two), UCI_FRAME) == ()
    high = FieldMapping(
        when=(AtLeast("chol", 240.0),), concept="HP:9000001", label="t", source="t", limits="t"
    )
    low = FieldMapping(
        when=(AtLeast("chol", 300.0),), concept="HP:9000002", label="t", source="t", limits="t"
    )
    assert tree_fields((high, low), UCI_FRAME) == ()


def test_the_label_is_never_a_tree_field():
    assert tree_fields((_row(LABEL, 0), _row(LABEL, 1)), UCI_FRAME) == ()


def test_a_weight_below_one_is_refused():
    with pytest.raises(ValueError, match="weight 0"):
        tree_shares(CELL, GROUP, TREE, 0)
