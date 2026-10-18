"""BKO gaps: the ontology statements instantiation from a dataset's held concepts never adds."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from bko.bkb.fusion import Normalization, fuse
from bko.bkb.structure import BKB
from bko.datasets.cells import cell_counts
from bko.gaps.detect import (
    CellGap,
    Statement,
    count_gaps,
    detect,
    grounded,
    ontology_fragments,
    represent,
    statements,
    ungrounded,
    ungrounded_from,
)
from bko.gaps.observe import Observed, dataset_fragment
from bko.ontology.fragments import is_a_fragment
from bko.ontology.terms import Ontology, Term
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import records
from tests._gaps_fixtures import LADDER_HPO, LADDER_MONDO, parse_ladders
from tests._ontology_fixtures import write_obo


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CONCEPTS = frozenset({"HP:9000003", "MONDO:9000004"})
HPO_CHAIN = (
    Statement("hpo", "HP:9000002", "HP:9000001"),
    Statement("hpo", "HP:9000003", "HP:9000002"),
)
MONDO_CHAIN = (
    Statement("mondo", "MONDO:9000002", "MONDO:9000001"),
    Statement("mondo", "MONDO:9000004", "MONDO:9000002"),
)


@pytest.fixture
def ladders(tmp_path) -> dict[str, Ontology]:
    return parse_ladders(tmp_path)


def data(*held: str) -> BKB:
    holders = MappingProxyType(dict.fromkeys(sorted(held), 1))
    return dataset_fragment(Observed(name="uci", records=2, holders=holders))


def observed(*held: str) -> Observed:
    holders = dict.fromkeys(sorted(CONCEPTS), 0) | dict.fromkeys(held, 1)
    return Observed(name="uci", records=2, holders=holders)


def test_ungrounded_from_holds_a_concept_no_record_shows(ladders):
    fragments = ontology_fragments(ladders, CONCEPTS)
    assert ungrounded_from(fragments, observed(), {"MONDO:9000004"}) == HPO_CHAIN
    assert ungrounded_from(fragments, observed(), set()) == (*HPO_CHAIN, *MONDO_CHAIN)


def test_ungrounded_from_leaves_out_the_prior_of_a_shown_concept_it_does_not_hold(ladders):
    fragments = ontology_fragments(ladders, CONCEPTS)
    shown = observed(*CONCEPTS)
    assert ungrounded_from(fragments, shown, CONCEPTS) == ()
    assert ungrounded_from(fragments, shown, {"MONDO:9000004"}) == HPO_CHAIN


def test_ungrounded_from_the_shown_concepts_is_detect(ladders):
    fragments = ontology_fragments(ladders, CONCEPTS)
    for held in ((), ("MONDO:9000004",), tuple(sorted(CONCEPTS))):
        data_set = observed(*held)
        expected = detect(fragments, dataset_fragment(data_set))
        assert ungrounded_from(fragments, data_set, data_set.held) == expected


def test_each_fusion_takes_its_dataset_fragments_name(ladders):
    assert represent(ontology_fragments(ladders, CONCEPTS), data()).name == "uci"


def test_each_release_holds_the_concepts_it_owns_and_their_ancestors(ladders):
    fragments = ontology_fragments(ladders, CONCEPTS)
    assert [f.name for f in fragments] == ["hpo", "mondo"]
    assert [[c for c, _ in f.inodes] for f in fragments] == [
        ["HP:9000001", "HP:9000002", "HP:9000003"],
        ["MONDO:9000001", "MONDO:9000002", "MONDO:9000004"],
    ]


def test_a_release_no_concept_belongs_to_contributes_no_fragment(ladders):
    assert [f.name for f in ontology_fragments(ladders, {"MONDO:9000003"})] == ["mondo"]


def test_a_dataset_that_holds_every_mapped_concept_leaves_no_gap(ladders):
    assert detect(ontology_fragments(ladders, CONCEPTS), data(*CONCEPTS)) == ()


def test_an_unheld_concept_leaves_its_chain_ungrounded(ladders):
    assert detect(ontology_fragments(ladders, CONCEPTS), data("MONDO:9000004")) == HPO_CHAIN


def test_an_unheld_leaf_beside_a_held_one_leaves_only_its_own_statement(ladders):
    fragments = ontology_fragments(ladders, {"MONDO:9000003", "MONDO:9000004"})
    expected = (Statement("mondo", "MONDO:9000003", "MONDO:9000002"),)
    assert detect(fragments, data("MONDO:9000004")) == expected


def test_a_held_parent_grounds_only_the_statements_above_it(ladders):
    fragments = ontology_fragments(ladders, {"HP:9000002", "HP:9000003"})
    expected = (Statement("hpo", "HP:9000003", "HP:9000002"),)
    assert detect(fragments, data("HP:9000002")) == expected


def test_a_dataset_that_holds_nothing_leaves_every_statement_ungrounded(ladders):
    fragments = ontology_fragments(ladders, CONCEPTS)
    assert detect(fragments, data()) == (*HPO_CHAIN, *MONDO_CHAIN)
    assert statements(represent(fragments, data())) == (*HPO_CHAIN, *MONDO_CHAIN)
    assert grounded(fragments, data()) == ()


def test_grounded_is_every_statement_detect_does_not_list(ladders):
    fragments = ontology_fragments(ladders, CONCEPTS)
    assert grounded(fragments, data("MONDO:9000004")) == MONDO_CHAIN


@pytest.mark.parametrize("normalization", list(Normalization))
def test_the_gaps_do_not_depend_on_weights_or_normalization(ladders, normalization):
    fragments = [*ontology_fragments(ladders, CONCEPTS), data("MONDO:9000004")]
    fused = fuse(fragments, [0.5, 3.0, 238.0], normalization=normalization)
    found = [(s.antecedent, s.consequent) for s in ungrounded(fused, ["MONDO:9000004"])]
    assert found == [(s.antecedent, s.consequent) for s in HPO_CHAIN]


def test_every_held_subset_matches_an_independent_closure():
    parents = {
        "MONDO:1": (),
        "MONDO:2": ("MONDO:1",),
        "MONDO:3": ("MONDO:1",),
        "MONDO:4": ("MONDO:2", "MONDO:3"),
        "MONDO:5": ("MONDO:4",),
    }
    terms = {
        curie: Term(
            curie=curie,
            name=None,
            parents=up,
            xrefs=(),
            matches=(),
            obsolete=False,
            replaced_by=(),
        )
        for curie, up in parents.items()
    }
    fragment = is_a_fragment(
        Ontology(name="mondo", version="test", terms=terms), set(parents), prefix="MONDO"
    )
    edges = sorted((child, parent) for child, up in parents.items() for parent in up)
    for size in range(len(parents) + 1):
        for held in itertools.combinations(sorted(parents), size):
            reached = set(held)
            while new := {parent for child, parent in edges if child in reached} - reached:
                reached |= new
            expected = tuple(Statement("mondo", c, p) for c, p in edges if c not in reached)
            assert detect([fragment], data(*held)) == expected, held


def test_count_gaps_are_the_underrepresented_cells_in_grid_order():
    counts = cell_counts(records({(40, 0, 1): 19, (40, 1, 1): 20, (60, 1, 0): 25}))
    gaps = count_gaps(counts)
    assert gaps == (
        CellGap("<55", "female", 0, 0),
        CellGap("<55", "female", 1, 19),
        CellGap("<55", "male", 0, 0),
        CellGap(">=55", "female", 0, 0),
        CellGap(">=55", "female", 1, 0),
        CellGap(">=55", "male", 1, 0),
    )
    assert gaps[1].cell == ("<55", "female", 1)


DRIVER = """
import sys
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from bko.gaps.definitions import Definition, gaps
from bko.gaps.detect import detect, ontology_fragments
from bko.gaps.observe import Observed, dataset_fragment, record_cells
from bko.ontology.obo import parse_obo
from tests._gaps_fixtures import ANGINA, STENOSIS

directory = Path(sys.argv[1])
ladders = {
    "HP": parse_obo(directory / "hp.obo", name="hpo"),
    "MONDO": parse_obo(directory / "mondo.obo", name="mondo"),
}
concepts = {"HP:9000003", "MONDO:9000003", "MONDO:9000004"}
for held in ((), ("MONDO:9000004",), ("HP:9000003", "MONDO:9000003")):
    holders = MappingProxyType(dict.fromkeys(held, 1))
    observed = Observed(name="uci", records=2, holders=holders)
    gaps_found = detect(ontology_fragments(ladders, concepts), dataset_fragment(observed))
    sys.stdout.write(repr(gaps_found) + "\\n")
frame = pd.DataFrame(
    {"age": [40, 40, 60, 60, 60], "sex": [0, 0, 1, 1, 1], "target": [0, 1, 1, 1, 0],
     "cp": [1, 4, 1, 1, 4]}
)
cells = record_cells(frame, (ANGINA, STENOSIS), name="uci")
fragments = ontology_fragments(ladders, {"HP:9000003", "MONDO:9000004"})
for definition in Definition:
    sys.stdout.write(repr(gaps(definition, fragments, cells)) + "\\n")
"""


def run_driver(seed: str, directory: Path) -> str:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    env["PYTHONHASHSEED"] = seed
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", DRIVER, str(directory)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_the_gaps_do_not_depend_on_the_hash_seed(tmp_path):
    write_obo(tmp_path, LADDER_HPO, "hp.obo")
    write_obo(tmp_path, LADDER_MONDO, "mondo.obo")
    outputs = {run_driver(seed, tmp_path) for seed in ("0", "1", "4294967295")}
    assert len(outputs) == 1
