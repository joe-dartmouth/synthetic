"""Gap detection under each planted data contradiction: the exact listing per definition."""

from __future__ import annotations

from typing import Final

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps.definitions import Definition, gaps
from bko.gaps.detect import ontology_fragments
from bko.gaps.observe import cell_key, record_cells
from tests._contradiction_fixtures import (
    BROAD,
    INNER,
    KINDS,
    NAME,
    NARROW,
    OM1,
    OUTER,
    OVERLAP_A,
    OVERLAP_B,
    ROOT,
    YW1,
    plant,
)


ALIASES: Final = {
    "root": ROOT,
    "broad": BROAD,
    "narrow": NARROW,
    "outer": OUTER,
    "inner": INNER,
    "a": OVERLAP_A,
    "b": OVERLAP_B,
}
"""Each concept's short name in `EXPECTED`."""

EXPECTED: Final[dict[str, dict[Definition, dict[str, str]]]] = {
    "narrow-without-broad": {
        Definition.SUBGROUP: {
            "<55/female/0": "broad>root narrow>broad",
            "<55/male/0": "broad>root narrow>broad outer>root inner>root",
            ">=55/female/0": "broad>root narrow>broad outer>root inner>root",
            ">=55/female/1": "broad>root narrow>broad outer>root inner>root",
            ">=55/male/0": "broad>root narrow>broad",
            ">=55/male/1": "narrow>broad",
        },
        Definition.POPULATION: {},
        Definition.FREQUENCY: {
            ">=55/male/1": "narrow>broad",
        },
    },
    "child-share-above-parent": {
        Definition.SUBGROUP: {
            "<55/female/0": "broad>root narrow>broad",
            "<55/female/1": "inner>root",
            "<55/male/0": "broad>root narrow>broad outer>root inner>root",
            ">=55/female/0": "broad>root narrow>broad outer>root inner>root",
            ">=55/female/1": "broad>root narrow>broad outer>root inner>root",
            ">=55/male/0": "broad>root narrow>broad",
            ">=55/male/1": "narrow>broad",
        },
        Definition.POPULATION: {},
        Definition.FREQUENCY: {
            ">=55/male/1": "narrow>broad",
        },
    },
    "empty-cell": {
        Definition.SUBGROUP: {
            "<55/female/1": "broad>root narrow>broad outer>root inner>root",
            "<55/male/0": "broad>root narrow>broad outer>root inner>root",
            ">=55/female/0": "broad>root narrow>broad outer>root inner>root",
            ">=55/female/1": "broad>root narrow>broad outer>root inner>root",
            ">=55/male/0": "broad>root narrow>broad",
            ">=55/male/1": "narrow>broad",
        },
        Definition.POPULATION: {},
        Definition.FREQUENCY: {
            ">=55/male/0": "broad>root narrow>broad",
            ">=55/male/1": "narrow>broad",
        },
    },
    "all-show": {
        Definition.SUBGROUP: {
            "<55/male/0": "broad>root narrow>broad outer>root inner>root",
            ">=55/female/0": "broad>root narrow>broad outer>root inner>root",
            ">=55/female/1": "broad>root narrow>broad outer>root inner>root",
        },
        Definition.POPULATION: {},
        Definition.FREQUENCY: {},
    },
    "code-overlap": {
        Definition.SUBGROUP: {
            "<55/male/0": "broad>root narrow>broad outer>root inner>root a>root b>root",
            ">=55/female/0": "broad>root narrow>broad outer>root inner>root a>root b>root",
            ">=55/female/1": "broad>root narrow>broad outer>root inner>root a>root b>root",
        },
        Definition.POPULATION: {},
        Definition.FREQUENCY: {},
    },
}
"""Per kind and definition, each cell's ungrounded statements as `child>parent`, in order."""


def listing(kind: str, definition: Definition) -> dict[str, str]:
    """The gaps of `kind`'s split under `definition`, written as `EXPECTED` writes them."""
    planted = plant(kind)
    cells = record_cells(planted.frame, planted.rows, name=NAME, edges=UCI_FRAME.edges)
    fragments = ontology_fragments(planted.ontologies, {row.concept for row in planted.rows})
    short = {curie: alias for alias, curie in ALIASES.items()}
    found: dict[str, list[str]] = {}
    for gap in gaps(definition, fragments, cells):
        assert gap.statement.source == "hpo"
        pair = f"{short[gap.statement.antecedent]}>{short[gap.statement.consequent]}"
        found.setdefault(cell_key(gap.cell), []).append(pair)
    return {key: " ".join(pairs) for key, pairs in found.items()}


@pytest.mark.parametrize("definition", list(Definition))
@pytest.mark.parametrize("kind", KINDS)
def test_each_kinds_gaps_are_listed_exactly(kind, definition):
    assert listing(kind, definition) == EXPECTED[kind][definition]


def test_a_held_child_grounds_its_parent_though_no_record_shows_the_parent():
    planted = plant("narrow-without-broad")
    young = planted.frame.loc[planted.frame["age"].lt(55) & planted.frame["sex"].eq(0)]
    shown = young.loc[young["target"].eq(1)]
    assert shown["cp"].eq(1).all()
    assert shown["exang"].eq(0).all()
    subgroup = listing("narrow-without-broad", Definition.SUBGROUP)
    assert cell_key(YW1) not in subgroup, "narrow grounds narrow>broad, and broad then broad>root"
    assert subgroup[cell_key(OM1)] == "narrow>broad", "broad alone grounds nothing below it"
