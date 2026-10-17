"""The dataset fragment: one prior per mapped concept its records show, at its share of them."""

from __future__ import annotations

from types import MappingProxyType

import pandas as pd
import pytest

from bko.bkb.structure import SNode
from bko.gaps.errors import GapError
from bko.gaps.observe import Observed, dataset_fragment, observe_records
from bko.ontology.fragments import INDIVIDUAL
from bko.ontology.schema import AtLeast, FieldMapping
from tests._gaps_fixtures import ANGINA, STENOSIS


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CHOL = FieldMapping(
    when=(AtLeast("chol", 240.0),),
    concept="HP:9000003",
    label="Angina pectoris",
    source="test",
    limits="test",
)
"""A second row on ANGINA's concept, so a record either row maps shows it once."""


def frame(*rows: tuple[int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["cp", "target", "chol"], dtype="int64")


def test_a_record_two_rows_map_to_one_concept_counts_once():
    observed = observe_records(
        frame((1, 0, 250), (1, 0, 100), (4, 0, 250), (4, 1, 100)),
        [ANGINA, CHOL, STENOSIS],
        name="uci",
    )
    assert (observed.name, observed.records) == ("uci", 4)
    assert observed.holders == {"HP:9000003": 3, "MONDO:9000004": 1}


def test_the_at_least_limit_is_inclusive():
    observed = observe_records(frame((4, 0, 239), (4, 0, 240)), [CHOL], name="uci")
    assert observed.holders == {"HP:9000003": 1}


def test_a_concept_no_record_shows_has_no_prior_and_one_record_is_enough():
    observed = observe_records(frame((1, 0, 100), (4, 0, 100)), [ANGINA, STENOSIS], name="uci")
    assert observed.held == ("HP:9000003",)
    assert dataset_fragment(observed).snodes == (
        SNode(head=("HP:9000003", INDIVIDUAL), prob=0.5, tail=frozenset()),
    )


def test_each_prior_is_its_concept_share_of_the_records():
    holders = MappingProxyType({"A:1": 1, "B:1": 3, "C:1": 0})
    fragment = dataset_fragment(Observed(name="uci", records=3, holders=holders))
    assert fragment.name == "uci"
    assert [(s.head[0], s.prob) for s in fragment.snodes] == [("A:1", 1 / 3), ("B:1", 1.0)]


def test_an_empty_dataset_is_refused():
    with pytest.raises(GapError, match="uci holds no record"):
        observe_records(frame(), [ANGINA], name="uci")


def test_a_row_on_a_field_the_records_lack_is_refused():
    thal = FieldMapping(
        when=(AtLeast("thal", 6.0),), concept="HP:1", label="x", source="t", limits="t"
    )
    with pytest.raises(GapError, match="the rows read fields uci lacks: thal"):
        observe_records(frame((1, 0, 100)), [thal], name="uci")
