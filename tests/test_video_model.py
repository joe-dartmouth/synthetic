"""`video.model`: the demo video's running example, held to `bko.bkb` and `bko.gaps`."""

from __future__ import annotations

import pytest

from bko.bkb.fusion import check_fused
from bko.datasets.cells import UNDERREPRESENTED_BELOW
from bko.gaps.detect import ungrounded
from bko.gaps.shares import GROUP_WEIGHT, SharePart
from video import model


pytestmark = pytest.mark.filterwarnings("error")

ST_DEPRESSION, ABNORMAL_ST, ABNORMAL_EKG = "HP:0012250", "HP:0012249", "HP:0003115"


def test_the_cells_sum_to_the_training_split():
    assert sum(model.CELL_RECORDS.values()) == 238
    assert model.CELL_RECORDS[model.CELL] == 2


def test_the_count_gap_line_is_the_projects_threshold():
    assert model.COUNT_GAP_BELOW == UNDERREPRESENTED_BELOW


def test_each_is_a_edge_is_an_s_node_of_probability_one():
    snodes = model.ontology_fragment().snodes
    assert len(snodes) == 5
    assert {(s.prob, s.tail, s.head) for s in snodes} == {
        (1.0, frozenset({(child, "a")}), (parent, "a")) for child, parent in model.EDGES
    }


def test_the_dataset_fragment_holds_a_prior_per_shown_concept_only():
    snodes = model.dataset_fragment().snodes
    assert all(not s.tail for s in snodes)
    assert {s.head[0]: s.prob for s in snodes} == {ST_DEPRESSION: 1.0, ABNORMAL_EKG: 0.5}


def test_fusion_gives_each_of_seven_sources_a_half():
    check_fused(model.fused(), [model.ontology_fragment(), model.dataset_fragment()])
    assert len(model.source_priors()) == 7
    assert set(model.source_priors().values()) == {0.5}


def test_fusion_keeps_both_s_nodes_on_abnormal_ekg():
    probs = [s.prob for s in model.fused().snodes if s.head == (ABNORMAL_EKG, "a")]
    assert sorted(probs) == [0.5, 1.0]


def test_the_passes_follow_the_st_chain():
    assert model.passes() == (((ST_DEPRESSION, ABNORMAL_ST),), ((ABNORMAL_ST, ABNORMAL_EKG),))


def test_the_gaps_are_what_gap_detection_lists():
    detected = ungrounded(model.fused(), [ABNORMAL_EKG, ST_DEPRESSION])
    assert model.gaps() == tuple((s.antecedent, s.consequent) for s in detected)
    assert model.gaps() == (
        ("HP:0001717", "HP:0003207"),
        ("HP:0003207", "HP:0004934"),
        ("HP:0100749", "HP:0012531"),
    )


def test_the_calcification_share_is_the_samplers_formula():
    assert model.share("HP:0001717") == pytest.approx(0.6446280991735537)
    assert model.GROUP_WEIGHT == GROUP_WEIGHT
    assert SharePart(2, 0.0, 78 / 110).share == model.share("HP:0001717")
