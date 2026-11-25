"""The ancestor-shrunk shares of `bko.generate.tree` and `bko-tree`'s draw of a tree field.

The hand tree is `top -> {R, unmapped}`, `R -> {A, B}`, `A -> {a1, a2}`, `B -> {b}`; its shares
are worked by hand from the rule of ADR `research/ancestor-shrunk-shares` at `GROUP_WEIGHT` 20.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import LABEL, SEXES, cell_keys
from bko.datasets.uci import UCI_FRAME
from bko.gaps.observe import Cell
from bko.gaps.shares import GROUP_WEIGHT
from bko.generate.direct import Draw, _Plan, sample
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
from bko.generate.presence import Presence
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
from tests.test_generate_direct import (
    DRAW_DIGEST,
    NAME,
    ONTOLOGIES,
    YOUNG_WOMEN_0,
    YOUNG_WOMEN_1,
    draw,
    varied,
)


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


TREE_SPEC: Final = DrawSpec(tree=True)

CP_ROWS: Final = (
    _row("cp", 1, concept="HP:0001681"),
    _row("cp", 2, concept="HP:0100749"),
    _row("cp", 3, concept="HP:0004421"),
)
"""`cp` as a tree field: angina, chest pain above it, and a sibling; `cp` 4 maps no concept."""

CP_LEAVES: Final = {1: "HP:0001681", 2: "HP:0100749/self", 3: "HP:0004421"}


def _tree_draw(
    train: pd.DataFrame,
    quota: Mapping[Cell, int],
    *,
    seed: int = 0,
    rows: tuple[FieldMapping, ...] = CP_ROWS,
    spec: DrawSpec = TREE_SPEC,
) -> dict[Cell, tuple[Draw, ...]]:
    return sample(
        train,
        quota,
        rows=rows,
        ontologies=ONTOLOGIES,
        name=NAME,
        seed=seed,
        frame=UCI_FRAME,
        spec=spec,
    )


def _leaf(record: Mapping[str, object]) -> str:
    return CP_LEAVES.get(int(cast("int", record["cp"])), UNMAPPED)


def _tree_oracle(
    train: pd.DataFrame, quota: Mapping[Cell, int], seed: int
) -> dict[Cell, list[dict[str, object]]]:
    """An independent draw: one uniform per record placing `cp`'s leaf, then a uniform pick per
    column from the cell's records, else the label group's, `cp`'s among those showing the leaf.
    """
    rng = np.random.default_rng(seed)
    tree = code_tree({row.concept for row in CP_ROWS}, ONTOLOGIES)
    keys = cell_keys(train, edges=UCI_FRAME.edges)
    records: list[dict[str, object]] = [
        {str(k): v for k, v in row.items()}
        for row in train[list(UCI_FRAME.columns)].to_dict("records")
    ]
    cells = list(keys.itertuples(index=False, name=None))
    drawn: dict[Cell, list[dict[str, object]]] = {}
    for cell, count in quota.items():
        band, sex, label = cell
        inside = [r for r, key in zip(records, cells, strict=True) if key == cell]
        group = [r for r in records if r[LABEL] == label]
        banded = [
            r for r, key in zip(records, cells, strict=True) if (key[0], key[2]) == (band, label)
        ]
        shares, _ = tree_shares(
            Counter(map(_leaf, inside)), Counter(map(_leaf, group)), tree, GROUP_WEIGHT
        )
        placed = [leaf for leaf in leaves(tree) if shares[leaf] > 0]
        cumulative = np.cumsum([shares[leaf] for leaf in placed])
        made = []
        for _ in range(count):
            record: dict[str, object] = {}
            for column in UCI_FRAME.columns:
                if column == "sex":
                    record[column] = next(code for code, name in SEXES.items() if name == sex)
                    continue
                if column == LABEL:
                    record[column] = label
                    continue
                if column == "cp":
                    position = int(np.searchsorted(cumulative, rng.random(), side="right"))
                    leaf = placed[min(position, len(placed) - 1)]
                    pool = [r for r in inside if _leaf(r) == leaf] or [
                        r for r in group if _leaf(r) == leaf
                    ]
                else:
                    pool = inside or (banded if column == "age" else group)
                record[column] = pool[int(rng.integers(len(pool)))][column]
            made.append(record)
        drawn[cell] = made
    return drawn


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_a_tree_draw_spends_one_uniform_per_tree_column(seed):
    train = varied()
    quota = {YOUNG_WOMEN_1: 25, YOUNG_WOMEN_0: 25}
    drawn = _tree_draw(train, quota, seed=seed)
    found = {cell: [dict(d.record) for d in draws] for cell, draws in drawn.items()}
    assert found == _tree_oracle(train, quota, seed)


def test_a_tree_columns_value_comes_from_a_donor_showing_the_drawn_leaf():
    train = varied()
    for cell, draws in _tree_draw(train, {YOUNG_WOMEN_1: 60, YOUNG_WOMEN_0: 60}).items():
        inside = cell_keys(train, edges=UCI_FRAME.edges).eq(list(cell)).all(axis=1)
        for d in draws:
            source = d.sources["cp"]
            assert _leaf({"cp": train.at[source, "cp"]}) == _leaf(d.record)
            assert train.at[source, "cp"] == d.record["cp"]
            assert bool(inside.at[source]) is ("cp" not in d.fallback)
            concept = leaf_concept(_leaf(d.record))
            assert d.concepts == (frozenset() if concept is None else {concept})
            assert {concept} - {None} <= d.closed


def test_a_leaf_the_cell_lacks_is_copied_from_the_label_group():
    train = varied()
    lacking = train.loc[~((train["cp"] == 3) & (train["age"] < 55) & (train["sex"] == 0))]
    draws = _tree_draw(lacking, {YOUNG_WOMEN_1: 200, YOUNG_WOMEN_0: 200})
    fell = [d for each in draws.values() for d in each if d.record["cp"] == 3]
    assert fell
    assert all("cp" in d.fallback for d in fell)


def test_a_tree_field_leaves_the_concept_draw_and_the_field_groups():
    angina = FieldMapping(
        when=(Codes("exang", frozenset({1})),),
        concept="HP:0001681",
        label="test",
        source="test",
        limits="test",
    )
    rows = (*CP_ROWS, angina)
    plan = _Plan(rows, ONTOLOGIES, UCI_FRAME, TREE_SPEC)
    assert plan._order == ["HP:0001681"]
    assert set(plan._first) == {"exang"}
    assert "cp" not in plan._free
    assert set(plan.trees) == {"cp"}
    default = _Plan(rows, ONTOLOGIES, UCI_FRAME, DEFAULT_SPEC)
    assert default._order == ["HP:0001681", "HP:0004421", "HP:0100749"]
    assert set(default._first) == {"cp"}
    assert default.trees == {}


def test_the_tree_spec_logs_its_field_and_refuses_the_nomap_and_logic_draws():
    assert TREE_SPEC.params() == {"tree": True}
    assert TREE_SPEC.tokens() == " tree=true"
    with pytest.raises(ValueError, match="tree needs the mapping table"):
        DrawSpec(tree=True, mapped=False)
    with pytest.raises(ValueError, match="tree needs the mapping table"):
        DrawSpec(tree=True, presence=Presence.LOGIC)


def test_the_default_spec_still_draws_the_group_as_before():
    train = varied()
    draws = [
        [
            [dict(d.record), sorted(d.concepts), sorted(d.closed), sorted(d.fallback)]
            for cell in draw(train, {YOUNG_WOMEN_1: 30, YOUNG_WOMEN_0: 30}, seed=seed).values()
            for d in cell
        ]
        for seed in range(5)
    ]
    found = hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest()
    assert found == DRAW_DIGEST
