"""The logic-only draw: presence from the `is_a` edges and nested pairs alone, no data prior."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections import Counter
from typing import Final

import numpy as np
import pytest

import bko.generate.direct as direct_module
import bko.generate.logic as logic_module
from bko.datasets.uci import UCI_FRAME
from bko.generate.direct import logic_components, sample
from bko.generate.logic import MAX_COMPONENT, Component, components, logic_assignment
from bko.generate.presence import Presence
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.terms import Ontology
from tests._validate_fixtures import planted
from tests.test_generate_direct import NAME, ONTOLOGIES, YOUNG_WOMEN_0, YOUNG_WOMEN_1, frame


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

DRAWS: Final = 30_000
TOLERANCE: Final = 0.015

ANGINA: Final = "HP:0001681"
CHEST_PAIN: Final = "HP:0100749"
LVH: Final = "HP:0001712"
EKG: Final = "HP:0003115"
CORONARY: Final = "HP:0005145"


def spread():
    """Records whose mapped fields vary apart from the label, so every drawn pattern has one."""
    return frame(
        [
            planted(
                age=(35 + i % 19) if i % 3 else 60 + i % 10,
                sex=i % 2,
                target=(i // 2) % 2,
                cp=1 + (i // 3) % 4,
                exang=(i // 5) % 2,
                trestbps=110 + (i // 7) % 40,
                chol=180 + 3 * ((i // 11) % 30),
                fbs=(i // 13) % 2,
                restecg=(i // 17) % 3,
                thalach=100 + i % 50,
                oldpeak=round(0.2 * ((i // 19) % 10), 1),
                slope=1 + (i // 23) % 3,
                ca=(i // 29) % 4,
                thal=(3, 6, 7)[(i // 31) % 3],
            )
            for i in range(800)
        ]
    )


def shares_of(component: Component, seed: int = 0) -> Counter[frozenset[str]]:
    rng = np.random.default_rng(seed)
    return Counter(logic_assignment(component, rng) for _ in range(DRAWS))


def logic_draw(seed: int = 0, quota: int = 200, ontologies=ONTOLOGIES, tag=None):
    return sample(
        spread(),
        {YOUNG_WOMEN_1: quota, YOUNG_WOMEN_0: quota},
        rows=FIELD_MAPPINGS,
        ontologies=ontologies,
        name=NAME,
        seed=seed,
        frame=UCI_FRAME,
        tag=tag,
        presence=Presence.LOGIC,
    )


def under_lvh() -> dict[str, Ontology]:
    """`ONTOLOGIES` with the label-only coronary stenosis moved under the nested inner LVH."""
    hpo = ONTOLOGIES["HP"]
    terms = dict(hpo.terms)
    terms[CORONARY] = dataclasses.replace(terms[CORONARY], parents=(LVH,))
    return {**ONTOLOGIES, "HP": dataclasses.replace(hpo, terms=terms)}


def test_the_logic_draw_reads_no_share(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("the logic draw read a share")

    monkeypatch.setattr(direct_module, "shares", refuse)
    drawn = logic_draw(quota=20)
    assert sum(len(draws) for draws in drawn.values()) == 40


def test_a_pair_draws_each_allowed_assignment_at_a_third():
    (pair,) = components(["A:1", "A:2"], [("A:1", "A:2")])
    counts = shares_of(pair)
    assert set(counts) == {frozenset(), frozenset({"A:2"}), frozenset({"A:1", "A:2"})}
    for times in counts.values():
        assert times / DRAWS == pytest.approx(1 / 3, abs=TOLERANCE)


def test_an_unrelated_concept_draws_at_a_half():
    lone, other = components(["A:1", "B:1"], [])
    for component in (lone, other):
        counts = shares_of(component)
        (concept,) = component.concepts
        assert counts[frozenset({concept})] / DRAWS == pytest.approx(0.5, abs=TOLERANCE)


def test_a_chain_of_three_draws_each_of_its_four_assignments_at_a_quarter():
    (chain,) = components(["A:1", "A:2", "A:3"], [("A:1", "A:2"), ("A:2", "A:3"), ("A:1", "A:3")])
    counts = shares_of(chain)
    assert set(counts) == {
        frozenset(),
        frozenset({"A:3"}),
        frozenset({"A:2", "A:3"}),
        frozenset({"A:1", "A:2", "A:3"}),
    }
    for times in counts.values():
        assert times / DRAWS == pytest.approx(1 / 4, abs=TOLERANCE)


def test_concepts_joined_through_shared_ends_form_one_component_of_eight_assignments():
    (joined,) = components(["A", "B", "C", "D"], [("A", "C"), ("A", "D"), ("B", "C")])
    assert joined.concepts == ("A", "B", "C", "D")
    assert joined.implies == (("A", "C"), ("A", "D"), ("B", "C"))
    counts = shares_of(joined)
    allowed = {
        frozenset(present)
        for flags in itertools.product((False, True), repeat=4)
        for present in [{c for c, flag in zip("ABCD", flags, strict=True) if flag}]
        if ("A" not in present or {"C", "D"} <= present) and ("B" not in present or "C" in present)
    }
    assert len(allowed) == 8
    assert set(counts) == allowed
    for times in counts.values():
        assert times / DRAWS == pytest.approx(1 / 8, abs=TOLERANCE)


def test_a_component_above_twelve_concepts_refuses():
    concepts = [f"A:{index:02d}" for index in range(MAX_COMPONENT + 1)]
    edges = list(itertools.pairwise(concepts))
    assert len(components(concepts[:-1], edges[:-1])) == 1
    with pytest.raises(ValueError, match=f"{MAX_COMPONENT + 1} concepts"):
        components(concepts, edges)


def test_a_drawn_child_forces_its_parents_field():
    drawn = logic_draw()
    draws = [d for cell in drawn.values() for d in cell]
    angina = [d for d in draws if ANGINA in d.concepts]
    assert angina, "no draw held angina"
    assert all(CHEST_PAIN in d.concepts for d in angina)
    assert {d.record["cp"] for d in angina} <= {1, 2, 3}
    assert any(CHEST_PAIN in d.concepts and ANGINA not in d.concepts for d in draws)


def test_the_logic_draw_is_seeded_and_follows_the_label():
    first, again = logic_draw(seed=3, quota=30), logic_draw(seed=3, quota=30)
    assert first == again
    stenosis = {"MONDO:0006715", "HP:0005145"}
    assert all(stenosis <= d.concepts for d in first[YOUNG_WOMEN_1])
    assert all(not stenosis & d.concepts for d in first[YOUNG_WOMEN_0])


def test_the_logic_draw_builds_components_from_is_a_and_nesting():
    drawn = logic_draw(quota=400)
    draws = [d for cell in drawn.values() for d in cell]
    lvh, ekg = "HP:0001712", "HP:0003115"
    assert all(ekg in d.concepts for d in draws if lvh in d.concepts)
    with_lvh = sum(lvh in d.concepts for d in draws) / len(draws)
    assert with_lvh == pytest.approx(1 / 3, abs=0.05)


def test_an_is_a_child_is_present_at_a_third_and_its_parent_at_two_thirds():
    draws = [d for cell in logic_draw(quota=400).values() for d in cell]
    with_angina = sum(ANGINA in d.concepts for d in draws) / len(draws)
    with_chest_pain = sum(CHEST_PAIN in d.concepts for d in draws) / len(draws)
    assert with_angina == pytest.approx(1 / 3, abs=0.05)
    assert with_chest_pain == pytest.approx(2 / 3, abs=0.05)


def test_a_label_only_concept_under_a_nested_inner_brings_the_outer_too():
    drawn = logic_draw(quota=200, ontologies=under_lvh())
    assert all({CORONARY, LVH, EKG} <= d.concepts for d in drawn[YOUNG_WOMEN_1])
    assert all(d.record["restecg"] == 2 for d in drawn[YOUNG_WOMEN_1])
    assert any(LVH not in d.concepts for d in drawn[YOUNG_WOMEN_0])


def test_a_refused_component_names_the_dataset_frame_holdout_and_fill(monkeypatch):
    monkeypatch.setattr(logic_module, "MAX_COMPONENT", 1)
    with pytest.raises(
        ValueError,
        match=rf"^{NAME} frame={UCI_FRAME.name} holdout=3 fill=gap cannot draw: a component of 2 ",
    ):
        logic_draw(quota=1, tag=(3, "gap"))
    with pytest.raises(ValueError, match="a component of 2 concepts"):
        logic_components(FIELD_MAPPINGS, ONTOLOGIES, UCI_FRAME)


def test_logic_components_are_the_ones_the_draw_enumerates():
    found = logic_components(FIELD_MAPPINGS, ONTOLOGIES, UCI_FRAME)
    pairs = {component.concepts for component in found if len(component.concepts) > 1}
    assert pairs == {(ANGINA, CHEST_PAIN), (LVH, EKG)}


LOGIC_DIGEST: Final = "439a3baef92dd3cb2147d47b0e53f6ff04164b0bfc592db1cada1c9912e53bde"
"""The sha256 of seeds 0-4's logic-only draws on `spread()`; it guards `bko-logic`'s draw."""


def test_the_logic_draw_is_pinned_by_digest():
    draws = [
        [
            [dict(d.record), sorted(d.concepts), sorted(d.closed), sorted(d.fallback)]
            for cell in logic_draw(seed=seed, quota=30).values()
            for d in cell
        ]
        for seed in range(5)
    ]
    found = hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest()
    assert found == LOGIC_DIGEST
