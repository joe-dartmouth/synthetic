"""The logic-only draw: presence from the `is_a` edges and nested pairs alone, no data prior."""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Final

import numpy as np
import pytest

import bko.generate.direct as direct_module
from bko.datasets.uci import UCI_FRAME
from bko.generate.direct import sample
from bko.generate.logic import MAX_COMPONENT, Component, Presence, components, logic_assignment
from bko.ontology.mapping import FIELD_MAPPINGS
from tests._validate_fixtures import planted
from tests.test_generate_direct import NAME, ONTOLOGIES, YOUNG_WOMEN_0, YOUNG_WOMEN_1, frame


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

DRAWS: Final = 30_000
TOLERANCE: Final = 0.015

ANGINA: Final = "HP:0001681"
CHEST_PAIN: Final = "HP:0100749"


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


def logic_draw(seed: int = 0, quota: int = 200):
    return sample(
        spread(),
        {YOUNG_WOMEN_1: quota, YOUNG_WOMEN_0: quota},
        rows=FIELD_MAPPINGS,
        ontologies=ONTOLOGIES,
        name=NAME,
        seed=seed,
        frame=UCI_FRAME,
        presence=Presence.LOGIC,
    )


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
