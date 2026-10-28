"""The direct sampler: records drawn from the fused BKB's concept shares, with no API call."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Final

import pandas as pd
import pytest

from bko.bkb.fusion import fuse, source_component
from bko.datasets.cells import LABEL
from bko.datasets.uci import UCI_FRAME
from bko.gaps.detect import ontology_fragments
from bko.gaps.observe import Cell, dataset_fragment, observe_records, record_cells
from bko.generate.direct import (
    GROUP_WEIGHT,
    Draw,
    FieldGroup,
    draw_line,
    field_groups,
    label_only,
    nested,
    sample,
    shares,
)
from bko.generate.errors import SamplingError
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology, Term
from bko.validate.checks import validate
from tests._validate_fixtures import planted


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

YOUNG_WOMEN_1: Final[Cell] = ("<55", "female", 1)
YOUNG_WOMEN_0: Final[Cell] = ("<55", "female", 0)
NAME: Final = "uci-test"


def _term(curie: str, *parents: str) -> Term:
    return Term(curie, curie, parents, (), (), False, ())


def _ontology(name: str, terms: Sequence[Term]) -> Ontology:
    return Ontology(name, "test", {term.curie: term for term in sorted(terms, key=str)})


ONTOLOGIES: Final[Mapping[str, Ontology]] = {
    "HP": _ontology(
        "hpo",
        [
            _term("HP:0000001"),
            _term("HP:0000118", "HP:0000001"),
            _term("HP:0100749", "HP:0000118"),
            _term("HP:0001681", "HP:0100749"),
            *(
                _term(curie, "HP:0000118")
                for curie in (
                    "HP:0004421",
                    "HP:0003124",
                    "HP:0003074",
                    "HP:0003115",
                    "HP:0001712",
                    "HP:0012250",
                    "HP:0001717",
                    "HP:0500018",
                    "HP:0005145",
                )
            ),
        ],
    ),
    "MONDO": _ontology(
        "mondo",
        [_term("MONDO:0005010"), _term("MONDO:0006715", "MONDO:0005010")],
    ),
}
"""The mapping table's concepts under invented parents: angina under chest pain, so a drawn
angina's closure holds a concept with a row of its own."""

SICK: Final = {
    "cp": 4,
    "exang": 1,
    "trestbps": 150,
    "chol": 250,
    "fbs": 1,
    "restecg": 2,
    "oldpeak": 2.0,
    "slope": 2,
    "ca": 1,
    "thal": 7,
}
"""Field values showing every mapped concept but chest pain, which `cp` 4 leaves absent."""


def frame(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Records as a training split: UCI's columns and dtypes, indexed from 1."""
    data = pd.DataFrame(list(records), columns=list(UCI_FRAME.columns))
    for column in UCI_FRAME.columns:
        dtype = "float64" if column in UCI_FRAME.fractional else "int64"
        data[column] = data[column].astype(dtype)
    data.index = pd.RangeIndex(1, len(data) + 1, name="record_id")
    return data


def draw(train: pd.DataFrame, quota: Mapping[Cell, int], seed: int = 0) -> dict[Cell, tuple]:
    return sample(train, quota, rows=FIELD_MAPPINGS, ontologies=ONTOLOGIES, name=NAME, seed=seed)


def test_a_draw_on_shares_of_0_and_1_is_the_record_written_by_hand():
    train = frame(
        [
            planted(age=40, sex=0, target=1, thalach=120, **SICK),
            planted(age=63, sex=1, target=1, thalach=130, **{**SICK, "trestbps": 140}),
            planted(age=63, sex=1, target=0, cp=3, exang=0, restecg=0),
        ]
    )
    (first, second) = draw(train, {YOUNG_WOMEN_1: 2})[YOUNG_WOMEN_1]
    assert dict(first.record) == {
        "age": 40,
        "sex": 0,
        "cp": 4,
        "trestbps": 150,
        "chol": 250,
        "fbs": 1,
        "restecg": 2,
        "thalach": 120,
        "exang": 1,
        "oldpeak": 2.0,
        "slope": 2,
        "ca": 1,
        "thal": 7,
        "target": 1,
    }
    assert list(first.record) == list(UCI_FRAME.columns)
    assert first.concepts == {
        "MONDO:0006715",
        "HP:0005145",
        "HP:0001681",
        "HP:0004421",
        "HP:0003124",
        "HP:0003074",
        "HP:0003115",
        "HP:0001712",
        "HP:0012250",
        "HP:0001717",
        "HP:0500018",
    }
    assert first.fallback == frozenset()
    assert second == first


def test_closure_is_recorded_and_never_forces_an_ancestors_row():
    train = frame([planted(age=40, sex=0, target=1, **SICK)])
    (only,) = draw(train, {YOUNG_WOMEN_1: 1})[YOUNG_WOMEN_1]
    assert "HP:0100749" not in only.concepts
    assert only.closed == only.concepts | {"HP:0100749", "HP:0000118", "HP:0000001"} | {
        "MONDO:0005010"
    }
    assert only.record["cp"] == 4, "chest pain's row would have forced cp 1 to 3"


def test_restecg_keeps_both_marginals_and_its_nesting():
    restecg = [2, 2, 2, 1, 1, 1, 0, 0, 0, 0]
    train = frame([planted(age=40, sex=0, target=1, restecg=code) for code in restecg])
    draws = draw(train, {YOUNG_WOMEN_1: 4000})[YOUNG_WOMEN_1]
    lvh = [d for d in draws if "HP:0001712" in d.concepts]
    ekg = [d for d in draws if "HP:0003115" in d.concepts]
    assert {d.record["restecg"] for d in lvh} == {2}
    assert {d.record["restecg"] for d in ekg} == {1, 2}
    assert all("HP:0003115" in d.concepts for d in lvh)
    assert len(lvh) / len(draws) == pytest.approx(0.3, abs=0.03)
    assert len(ekg) / len(draws) == pytest.approx(0.6, abs=0.03)


def test_label_only_rows_follow_the_cells_label():
    train = frame(
        [planted(age=40, sex=0, target=label, **SICK) for label in (0, 1) for _ in range(3)]
    )
    drawn = draw(train, {YOUNG_WOMEN_0: 5, YOUNG_WOMEN_1: 5})
    stenosis = {"MONDO:0006715", "HP:0005145"}
    assert all(not stenosis & d.concepts for d in drawn[YOUNG_WOMEN_0])
    assert all(stenosis <= d.concepts for d in drawn[YOUNG_WOMEN_1])
    assert {d.record[LABEL] for d in drawn[YOUNG_WOMEN_0]} == {0}


def varied() -> pd.DataFrame:
    """Records of both labels in every young cell and the older men, values spread over fields."""
    return frame(
        [
            planted(
                age=(35 + index % 19) if index % 3 else 60 + index % 10,
                sex=index % 2,
                target=(index // 2) % 2,
                cp=1 + index % 4,
                exang=(index // 4) % 2,
                trestbps=110 + 2 * index,
                chol=180 + 5 * index,
                fbs=(index // 5) % 2,
                restecg=index % 3,
                thalach=100 + index,
                oldpeak=round(0.1 * index, 1),
                slope=1 + index % 3,
                ca=index % 4,
                thal=(3, 6, 7)[index % 3],
            )
            for index in range(48)
        ]
    )


def test_every_drawn_record_passes_validation_and_lies_inside_the_training_range():
    train = varied()
    cells = record_cells(train, FIELD_MAPPINGS, name=NAME)
    quota = {cell: 40 for cell, records in cells.each.items() if records.observed.records}
    drawn = draw(train, quota)
    records = [dict(d.record) for draws in drawn.values() for d in draws]
    assert len(validate(records, frame=UCI_FRAME).accepted) == len(records) == 40 * len(quota)
    for field in UCI_FRAME.numeric:
        values = [record[field] for record in records]
        assert train[field].min() <= min(values)
        assert max(values) <= train[field].max()
    for cell, draws in drawn.items():
        assert all(d.cell == cell for d in draws)
        assert {(d.record["sex"], d.record[LABEL]) for d in draws} == {
            (0 if cell[1] == "female" else 1, cell[2])
        }


def test_the_same_seed_draws_the_same_records_and_another_seed_does_not():
    train = varied()
    quota = {YOUNG_WOMEN_1: 30, YOUNG_WOMEN_0: 30}
    assert draw(train, quota, seed=7) == draw(train, quota, seed=7)
    assert draw(train, quota, seed=7) != draw(train, quota, seed=8)


def test_a_field_the_cell_lacks_the_drawn_side_of_falls_back_and_is_logged(caplog):
    caplog.set_level(logging.INFO)
    train = frame(
        [planted(age=40, sex=0, target=1, trestbps=120) for _ in range(5)]
        + [planted(age=63, sex=1, target=1, trestbps=150) for _ in range(15)]
    )
    draws = draw(train, {YOUNG_WOMEN_1: 60})[YOUNG_WOMEN_1]
    high = [d for d in draws if "HP:0004421" in d.concepts]
    assert high, "a share above 0 drew no elevated pressure in 60 records"
    assert all(d.fallback == {"trestbps"} and d.record["trestbps"] == 150 for d in high)
    assert all(not d.fallback and d.record["trestbps"] == 120 for d in draws if d not in high)
    assert f"fallback key=<55/female/1 field=trestbps draws={len(high)}" in caplog.messages


def test_a_draw_no_record_matches_refuses_the_attempt():
    train = frame(
        [planted(age=40, sex=0, target=1, cp=1, exang=0) for _ in range(5)]
        + [planted(age=40, sex=0, target=1, cp=4, exang=0) for _ in range(5)]
    )
    with pytest.raises(SamplingError, match=r"<55/female/1 cannot draw cp\+exang"):
        draw(train, {YOUNG_WOMEN_1: 200})


def test_shares_are_the_fused_bkbs_priors_weighted_by_the_cells_records():
    train = varied()
    cells = record_cells(train, FIELD_MAPPINGS, name=NAME)
    concepts = {row.concept for row in FIELD_MAPPINGS}
    for cell, records in cells.each.items():
        members = train.loc[train[LABEL].eq(cell[2])]
        group = observe_records(members, FIELD_MAPPINGS, name=f"{NAME}/group")
        mine, theirs = dataset_fragment(records.observed), dataset_fragment(group)
        fused = fuse([*ontology_fragments(ONTOLOGIES, concepts), mine, theirs], [1.0] * 4)

        def prior(concept: str, source: str, fused=fused) -> float:
            tail = frozenset({(source_component(concept), source)})
            found = [s.prob for s in fused.snodes if s.head[0] == concept and s.tail == tail]
            return found[0] if found else 0.0

        n = records.observed.records
        expected = {
            concept: (n * prior(concept, mine.name) + GROUP_WEIGHT * prior(concept, theirs.name))
            / (n + GROUP_WEIGHT)
            for concept in sorted(concepts)
        }
        assert shares(records, group) == expected
    assert GROUP_WEIGHT == 20


def test_the_mapping_table_reads_as_eight_field_groups_one_nesting_and_two_label_rows():
    assert field_groups(FIELD_MAPPINGS) == (
        FieldGroup(("cp", "exang"), ("HP:0001681", "HP:0100749")),
        FieldGroup(("trestbps",), ("HP:0004421",)),
        FieldGroup(("chol",), ("HP:0003124",)),
        FieldGroup(("fbs",), ("HP:0003074",)),
        FieldGroup(("restecg",), ("HP:0001712", "HP:0003115")),
        FieldGroup(("oldpeak", "slope"), ("HP:0012250",)),
        FieldGroup(("ca",), ("HP:0001717",)),
        FieldGroup(("thal",), ("HP:0500018",)),
    )
    assert nested(FIELD_MAPPINGS) == {"HP:0001712": "HP:0003115"}
    assert label_only(FIELD_MAPPINGS) == {"MONDO:0006715", "HP:0005145"}


def _restecg(concept: str, *codes: int) -> FieldMapping:
    return FieldMapping(
        when=(Codes("restecg", frozenset(codes)),),
        concept=concept,
        label=concept,
        source="test",
        limits="test",
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            (_restecg("HP:1", 2), _restecg("HP:2", 1, 2), _restecg("HP:3", 0, 1, 2)),
            "HP:1 lies inside 2 concepts",
        ),
        ((_restecg("HP:1", 1), _restecg("HP:2", 2), _restecg("HP:3", 1, 2)), "HP:3 holds 2"),
    ],
    ids=["chain", "two-inners"],
)
def test_nesting_deeper_or_wider_than_one_pair_is_refused(rows, message):
    with pytest.raises(ValueError, match=message):
        nested(rows)
    with pytest.raises(ValueError, match=message):
        sample(frame([planted()]), {}, rows=rows, ontologies={}, name=NAME, seed=0)


def test_an_empty_cells_age_comes_from_its_label_group_inside_its_band():
    train = frame(
        [planted(age=40 + i, sex=1, target=1, thalach=120 + i) for i in range(3)]
        + [planted(age=60 + i, sex=1, target=1, thalach=150 + i) for i in range(12)]
    )
    draws = draw(train, {YOUNG_WOMEN_1: 60})[YOUNG_WOMEN_1]
    assert {d.record["age"] for d in draws} == {40, 41, 42}
    assert all({"age", "thalach"} <= d.fallback for d in draws)
    assert {d.record["thalach"] for d in draws} > {120, 121, 122}, "thalach reads every band"


def test_an_empty_cell_with_no_label_record_in_its_band_refuses():
    train = frame([planted(age=60 + i, sex=1, target=1) for i in range(5)])
    with pytest.raises(SamplingError, match="<55/female/1 cannot draw age"):
        draw(train, {YOUNG_WOMEN_1: 1})


def test_a_cell_with_records_reads_age_and_thalach_from_them_with_no_fallback():
    train = frame(
        [planted(age=40, sex=0, target=1, thalach=111)]
        + [planted(age=63, sex=1, target=1, thalach=150) for _ in range(5)]
    )
    draws = draw(train, {YOUNG_WOMEN_1: 20})[YOUNG_WOMEN_1]
    assert {(d.record["age"], d.record["thalach"]) for d in draws} == {(40, 111)}
    assert not any({"age", "thalach"} & d.fallback for d in draws)


def test_the_draw_log_line_holds_every_field_of_the_draw():
    record = {"age": 41, "target": 1}
    found = Draw(
        YOUNG_WOMEN_1,
        record,
        frozenset({"HP:2", "HP:1"}),
        frozenset({"HP:2", "HP:1", "HP:0"}),
        frozenset({"exang", "cp"}),
    )
    assert draw_line(3, 7, found) == (
        '{"call": 3, "cell": "<55/female/1", "closed": ["HP:0", "HP:1", "HP:2"], '
        '"concepts": ["HP:1", "HP:2"], "fallback": ["cp", "exang"], "kind": "draw", '
        '"position": 7}\n'
    )


def test_seed_0_draws_each_cell_in_quota_order_and_each_concept_in_its_order():
    drawn = draw(varied(), {YOUNG_WOMEN_1: 3, YOUNG_WOMEN_0: 3}, seed=0)
    assert list(drawn) == [YOUNG_WOMEN_1, YOUNG_WOMEN_0]
    first, second = drawn[YOUNG_WOMEN_1][0], drawn[YOUNG_WOMEN_0][0]
    assert dict(first.record) == {
        "age": 42,
        "sex": 0,
        "cp": 4,
        "trestbps": 114,
        "chol": 190,
        "fbs": 1,
        "restecg": 2,
        "thalach": 102,
        "exang": 0,
        "oldpeak": 2.2,
        "slope": 2,
        "ca": 2,
        "thal": 7,
        "target": 1,
    }
    assert first.concepts == {
        "HP:0001712",
        "HP:0001717",
        "HP:0003074",
        "HP:0003115",
        "HP:0005145",
        "HP:0012250",
        "HP:0500018",
        "MONDO:0006715",
    }
    assert first.fallback == {"cp", "exang"}
    assert dict(second.record) == {
        "age": 44,
        "sex": 0,
        "cp": 1,
        "trestbps": 142,
        "chol": 380,
        "fbs": 1,
        "restecg": 0,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 0.8,
        "slope": 3,
        "ca": 0,
        "thal": 7,
        "target": 0,
    }
    assert second.concepts == {
        "HP:0001681",
        "HP:0003074",
        "HP:0003124",
        "HP:0004421",
        "HP:0100749",
        "HP:0500018",
    }
