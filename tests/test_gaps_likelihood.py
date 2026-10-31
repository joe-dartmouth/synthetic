"""`bko.gaps.likelihood`: a record's log-probability under the fused BKB's concept draw."""

from __future__ import annotations

import math

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps.likelihood import Likelihood
from bko.ontology.mapping import FIELD_MAPPINGS
from tests._validate_fixtures import planted


SHARES = {
    "HP:0001681": 0.3,
    "HP:0100749": 0.6,
    "HP:0004421": 0.25,
    "HP:0003124": 0.5,
    "HP:0003074": 0.1,
    "HP:0003115": 0.4,
    "HP:0001712": 0.1,
    "HP:0012250": 0.2,
    "HP:0001717": 0.35,
    "HP:0500018": 0.45,
    "MONDO:0006715": 0.9,
    "HP:0005145": 0.9,
}
"""Hand-set shares for every UCI concept, the nested pair's outer above its inner."""

CONCEPTS = (
    "HP:0001681",
    "HP:0100749",
    "HP:0004421",
    "HP:0003124",
    "HP:0003074",
    "HP:0012250",
    "HP:0001717",
    "HP:0500018",
)
"""The UCI concepts drawn alone: neither label-only nor in the nested pair."""


def score(record, shares=SHARES) -> float:
    return Likelihood(shares, FIELD_MAPPINGS, UCI_FRAME)(record)


def shown(record, concept: str) -> bool:
    return any(row.applies(record) for row in FIELD_MAPPINGS if row.concept == concept)


def expected(record) -> float:
    """The sampler's draw probability by hand: Bernoulli terms, then the nested pair."""
    total = sum(math.log(SHARES[c] if shown(record, c) else 1 - SHARES[c]) for c in CONCEPTS)
    inner, outer = "HP:0001712", "HP:0003115"
    total += math.log(SHARES[inner] if shown(record, inner) else 1 - SHARES[inner])
    if not shown(record, inner):
        conditional = (SHARES[outer] - SHARES[inner]) / (1 - SHARES[inner])
        total += math.log(conditional if shown(record, outer) else 1 - conditional)
    return total


@pytest.mark.parametrize(
    "record",
    [planted(), planted(restecg=0), planted(restecg=1), planted(restecg=2, chol=320, fbs=1)],
    ids=["planted", "restecg-0", "restecg-1", "restecg-2"],
)
def test_the_score_sums_log_shares_of_present_and_absent_concepts(record):
    assert score(record) == pytest.approx(expected(record))


def test_flipping_one_concept_moves_the_score_by_its_term():
    low, high = planted(fbs=0), planted(fbs=1)
    assert not shown(low, "HP:0003074")
    assert shown(high, "HP:0003074")
    share = SHARES["HP:0003074"]
    assert score(high) - score(low) == pytest.approx(math.log(share) - math.log(1 - share))


def test_label_only_and_unmapped_concepts_contribute_nothing():
    record = planted()
    changed = {**SHARES, "MONDO:0006715": 0.01, "HP:0005145": 0.99, "HP:9999999": 0.5}
    assert score(record, changed) == score(record)


def test_the_nested_pair_scores_the_conditional_draw():
    inner_shown, outer_only = planted(restecg=2), planted(restecg=1)
    assert shown(inner_shown, "HP:0001712")
    assert shown(inner_shown, "HP:0003115")
    assert shown(outer_only, "HP:0003115")
    assert not shown(outer_only, "HP:0001712")
    rest = (
        score(planted(restecg=0))
        - math.log(1 - SHARES["HP:0001712"])
        - math.log(1 - (SHARES["HP:0003115"] - SHARES["HP:0001712"]) / (1 - SHARES["HP:0001712"]))
    )
    assert score(inner_shown) - rest == pytest.approx(math.log(SHARES["HP:0001712"]))
    conditional = (SHARES["HP:0003115"] - SHARES["HP:0001712"]) / (1 - SHARES["HP:0001712"])
    assert score(outer_only) - rest == pytest.approx(
        math.log(1 - SHARES["HP:0001712"]) + math.log(conditional)
    )


def test_a_two_row_concept_is_their_or():
    """`HP:0001681` reads `cp` 1 or `exang` 1: either row shows it."""
    rows = [row for row in FIELD_MAPPINGS if row.concept == "HP:0001681"]
    assert len(rows) == 2
    by_cp, by_exang, neither = (
        planted(cp=1, exang=0),
        planted(cp=4, exang=1),
        planted(cp=4, exang=0),
    )
    share = SHARES["HP:0001681"]
    others = score(neither) - math.log(1 - share)
    assert score(by_exang) - others == pytest.approx(math.log(share))
    cp_term = (
        math.log(1 - SHARES["HP:0100749"])
        if not shown(by_cp, "HP:0100749")
        else math.log(SHARES["HP:0100749"])
    )
    neither_cp = (
        math.log(1 - SHARES["HP:0100749"])
        if not shown(neither, "HP:0100749")
        else math.log(SHARES["HP:0100749"])
    )
    assert score(by_cp) - (others - neither_cp + cp_term) == pytest.approx(math.log(share))


def test_a_contradicting_share_scores_minus_infinity():
    record = planted(fbs=1)
    found = score(record, {**SHARES, "HP:0003074": 0.0})
    assert math.isinf(found)
    assert found < 0


def test_an_absent_concept_at_share_1_scores_minus_infinity():
    record = planted(fbs=0)
    assert score(record, {**SHARES, "HP:0003074": 1.0}) == -math.inf


def test_an_inner_share_of_1_makes_its_outer_certain():
    shares = {**SHARES, "HP:0001712": 1.0}
    inner = planted(restecg=2)
    assert shown(inner, "HP:0001712")
    assert shown(inner, "HP:0003115")
    others = sum(math.log(SHARES[c] if shown(inner, c) else 1 - SHARES[c]) for c in CONCEPTS)
    assert score(inner, shares) == pytest.approx(others)
    assert score(planted(restecg=0), shares) == -math.inf


def test_one_record_scores_its_hand_computed_value():
    record = planted(cp=4, exang=0, chol=200, fbs=0, restecg=0, trestbps=120, thalach=150)
    present = {c for c in (*CONCEPTS, "HP:0001712", "HP:0003115") if shown(record, c)}
    assert "HP:0001712" not in present
    assert "HP:0003115" not in present
    hand = sum(math.log(SHARES[c] if c in present else 1 - SHARES[c]) for c in CONCEPTS)
    hand += math.log(0.9) + math.log(1 - (0.4 - 0.1) / 0.9)
    assert score(record) == pytest.approx(hand)
