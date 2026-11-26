"""The knowledge-only draw of `bko-kb`: records from K alone, reading no training record.

ADR `research/matched-knowledge-controls`, Decision 9, fixes the draw each test reads.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
from collections.abc import Sequence
from typing import Final

import numpy as np
import pytest

from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.generate.direct import Draw, sample
from bko.generate.draw_spec import DrawSpec
from bko.generate.errors import SamplingError
from bko.generate.knowledge_draw import kb_sample
from bko.generate.logic import components, logic_assignment
from bko.generate.presence import Presence
from bko.ontology.schema import AtLeast, Codes, FieldMapping
from tests._kb_fixtures import (
    BROAD,
    FRAME,
    JOINT,
    MIDDLE_0,
    NARROW,
    OLD_1,
    ONTOLOGIES,
    QUOTA,
    ROWS,
    YOUNG_1,
    kb_row,
    kb_train,
)


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

KB: Final = DrawSpec(presence=Presence.LOGIC, donors=False)


def _kb(
    *,
    rows: Sequence[FieldMapping] = ROWS,
    frame: FrameSpec = FRAME,
    age_range: tuple[int, int] = (25, 80),
    seed: int = 3,
) -> list[Draw]:
    drawn = kb_sample(
        QUOTA, rows=rows, ontologies=ONTOLOGIES, frame=frame, seed=seed, age_range=age_range
    )
    return [d for draws in drawn.values() for d in draws]


def test_the_draw_takes_no_training_frame():
    parameters = inspect.signature(kb_sample).parameters
    assert set(parameters) == {
        "quota",
        "rows",
        "ontologies",
        "frame",
        "seed",
        "age_range",
        "tag",
    }
    assert all(name not in parameters for name in ("train", "records", "data"))


def test_two_training_frames_of_one_age_range_give_one_fill():
    one = kb_train([25, *range(30, 80, 3), 80], shift=1)
    two = kb_train([80, *range(41, 79, 5), 25], shift=2)
    assert not one.reset_index(drop=True).equals(two.reset_index(drop=True))
    drawn = [
        sample(
            train,
            QUOTA,
            rows=ROWS,
            ontologies=ONTOLOGIES,
            name="toy",
            seed=3,
            frame=FRAME,
            spec=KB,
        )
        for train in (one, two)
    ]
    assert drawn[0] == drawn[1]
    assert drawn[0] == kb_sample(
        QUOTA, rows=ROWS, ontologies=ONTOLOGIES, frame=FRAME, seed=3, age_range=(25, 80)
    )


def test_every_value_is_a_codebook_code_showing_the_pattern():
    draws = _kb()
    for draw in draws:
        record = draw.record
        assert all(record[f] in FRAME.codebook[f] for f in FRAME.categorical)
        assert record["sex"] == (1 if draw.cell[1] == "male" else 0)
        assert record["target"] == draw.cell[2]
        for row in ROWS:
            assert row.applies(record) is (row.concept in draw.concepts)
        assert draw.sources == {}
        assert draw.fallback == frozenset()
        assert draw.closed >= draw.concepts
    patterns = {frozenset(d.concepts) for d in draws}
    assert {
        frozenset(),
        frozenset({BROAD.concept}),
        frozenset({BROAD.concept, NARROW.concept}),
    } <= {p - {JOINT.concept} for p in patterns}
    assert all(NARROW.concept not in p or BROAD.concept in p for p in patterns)


def test_age_is_inside_the_band_and_the_age_range():
    draws = _kb(age_range=(33, 64))
    bands = {YOUNG_1: (33, 39), MIDDLE_0: (40, 59), OLD_1: (60, 64)}
    for cell, (low, high) in bands.items():
        ages = {d.record["age"] for d in draws if d.cell == cell}
        assert ages <= set(range(low, high + 1))
        assert len(ages) > 1
    assert all(isinstance(d.record["age"], int) for d in draws)


def test_swapping_one_components_rows_leaves_other_components_equal():
    swapped = dataclasses.replace(
        JOINT, when=(Codes("b", frozenset({0})), Codes("c", frozenset({1})))
    )
    before, after = _kb(), _kb(rows=(BROAD, NARROW, swapped))
    kept = ("age", "sex", "a", "target")
    assert [{f: d.record[f] for f in kept} for d in before] == [
        {f: d.record[f] for f in kept} for d in after
    ]
    pair = {BROAD.concept, NARROW.concept}
    assert [d.concepts & pair for d in before] == [d.concepts & pair for d in after]
    assert [(d.record["b"], d.record["c"]) for d in before] != [
        (d.record["b"], d.record["c"]) for d in after
    ]
    for draw in after:
        shows = draw.record["b"] == 0 and draw.record["c"] == 1
        assert shows is (swapped.concept in draw.concepts)


def test_a_frame_with_a_free_field_is_refused():
    free = dataclasses.replace(
        FRAME,
        name="kb-free",
        features=(*FRAME.features, "d"),
        categorical=(*FRAME.categorical, "d"),
        codebook={**FRAME.codebook, "d": {0: "no", 1: "yes"}},
        descriptions={**FRAME.descriptions, "d": "an unmapped flag"},
    )
    with pytest.raises(ValueError, match=r"\bd\b"):
        _kb(frame=free)


def test_a_kb_refusal_names_the_holdout_and_the_fill():
    free = dataclasses.replace(
        FRAME,
        name="kb-free",
        features=(*FRAME.features, "d"),
        categorical=(*FRAME.categorical, "d"),
        codebook={**FRAME.codebook, "d": {0: "no", 1: "yes"}},
        descriptions={**FRAME.descriptions, "d": "an unmapped flag"},
    )
    train = kb_train([25, *range(30, 80, 3), 80], shift=1).assign(d=0)
    with pytest.raises(ValueError, match="toy frame=kb-free holdout=3 fill=gap cannot draw: "):
        sample(
            train,
            QUOTA,
            rows=ROWS,
            ontologies=ONTOLOGIES,
            name="toy",
            seed=3,
            frame=free,
            spec=KB,
            tag=(3, "gap"),
        )


def test_a_frame_with_a_numeric_clause_is_refused():
    numeric = dataclasses.replace(
        FRAME,
        name="kb-numeric",
        features=(*FRAME.features, "level"),
        numeric=(*FRAME.numeric, "level"),
        ranges=(*FRAME.ranges, RangeConstraint("level", "u", 0, 9, "toy", "toy")),
        descriptions={**FRAME.descriptions, "level": "a level"},
    )
    rows = (*ROWS, kb_row("HP:0000011", AtLeast("level", 5)))
    with pytest.raises(ValueError, match=r"\blevel\b"):
        _kb(rows=rows, frame=numeric)


def test_a_pattern_no_codebook_tuple_shows_is_refused():
    unshowable = kb_row("HP:0000012", Codes("b", frozenset({7})), Codes("c", frozenset({2})))
    with pytest.raises(SamplingError, match="no codebook tuple of"):
        _kb(rows=(BROAD, NARROW, unshowable))


def test_a_band_with_no_age_inside_the_range_is_refused():
    with pytest.raises(SamplingError, match="its band holds no age inside"):
        _kb(age_range=(70, 80))


def test_the_draw_follows_its_seed():
    assert _kb(seed=3) == _kb(seed=3)
    assert _kb(seed=3) != _kb(seed=4)


def test_components_and_columns_draw_from_distinct_seed_prefixes():
    draws = [d for d in _kb() if d.cell == YOUNG_1]
    index = FRAME.columns.index("age")
    expected = [
        int(np.random.default_rng([3, 0, record, 1, index]).integers(25, 40))
        for record in range(QUOTA[YOUNG_1])
    ]
    assert [d.record["age"] for d in draws] == expected
    pair = {BROAD.concept, NARROW.concept}
    concepts = (JOINT.concept, *pair)
    found = components(concepts, [(NARROW.concept, BROAD.concept)])
    k, component = next((k, c) for k, c in enumerate(found) if BROAD.concept in c.concepts)
    expected_pairs = [
        logic_assignment(component, np.random.default_rng([3, 0, record, 0, k]))
        for record in range(QUOTA[YOUNG_1])
    ]
    assert [d.concepts & pair for d in draws] == expected_pairs


def test_the_kb_spec_needs_presence_logic():
    assert KB.params() == {"presence": "logic", "donors": False}
    assert KB.tokens() == " donors=false"
    for presence in (Presence.SHARES, Presence.POOLED):
        with pytest.raises(ValueError, match="donors false needs presence logic"):
            DrawSpec(presence=presence, donors=False)
    with pytest.raises(ValueError, match="donors false needs presence logic"):
        DrawSpec(mapped=False, donors=False)


def test_every_codebook_tuple_showing_a_pattern_is_drawn():
    draws = _kb()
    joint = {(d.record["b"], d.record["c"]) for d in draws if JOINT.concept in d.concepts}
    absent = {(d.record["b"], d.record["c"]) for d in draws if JOINT.concept not in d.concepts}
    assert joint == {(1, 2), (1, 3)}
    assert absent == set(itertools.product((0, 1), range(4))) - joint
