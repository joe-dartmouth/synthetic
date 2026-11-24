"""The no-mapping draw: `bko-direct`'s sampler at a `DrawSpec` with `mapped=False`."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Final

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import LABEL, SEXES, cell_keys
from bko.datasets.uci import UCI_FRAME
from bko.gaps.observe import Cell
from bko.generate.direct import Draw, sample
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
from bko.generate.presence import Presence
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import FieldMapping
from tests._validate_fixtures import planted
from tests.test_generate_direct import (
    NAME,
    ONTOLOGIES,
    YOUNG_WOMEN_0,
    YOUNG_WOMEN_1,
    frame,
    varied,
)


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

NOMAP: Final = DrawSpec(mapped=False)
CELL_COLUMNS: Final = frozenset({"sex", LABEL})
"""The columns that follow the cell, read from no record."""


def nomap(
    train: pd.DataFrame,
    quota: Mapping[Cell, int],
    *,
    seed: int = 0,
    rows: tuple[FieldMapping, ...] = FIELD_MAPPINGS,
    tag: tuple[int, str] | None = None,
) -> dict[Cell, tuple[Draw, ...]]:
    return sample(
        train,
        quota,
        rows=rows,
        ontologies=ONTOLOGIES,
        name=NAME,
        seed=seed,
        frame=UCI_FRAME,
        tag=tag,
        spec=NOMAP,
    )


def oracle(
    train: pd.DataFrame, quota: Mapping[Cell, int], seed: int
) -> dict[Cell, list[dict[str, object]]]:
    """An independent uniform pick per column: the cell's records, else the label group's.

    `age` falls back to the label group's records in the cell's band; `sex` and the label follow
    the cell and draw nothing.
    """
    rng = np.random.default_rng(seed)
    keys = cell_keys(train, edges=UCI_FRAME.edges)
    records: list[dict[str, object]] = [
        {str(k): v for k, v in row.items()}
        for row in train[list(UCI_FRAME.columns)].to_dict("records")
    ]
    drawn: dict[Cell, list[dict[str, object]]] = {}
    for cell, count in quota.items():
        band, sex, label = cell
        inside = [
            r
            for r, (b, s, t) in zip(records, keys.itertuples(index=False), strict=True)
            if (b, s, t) == cell
        ]
        group = [r for r in records if r[LABEL] == label]
        banded = [
            r
            for r, (b, _, t) in zip(records, keys.itertuples(index=False), strict=True)
            if t == label and b == band
        ]
        cells = []
        for _ in range(count):
            record: dict[str, object] = {}
            for column in UCI_FRAME.columns:
                if column == "sex":
                    record[column] = next(code for code, name in SEXES.items() if name == sex)
                elif column == LABEL:
                    record[column] = label
                else:
                    pool = inside or (banded if column == "age" else group)
                    record[column] = pool[int(rng.integers(len(pool)))][column]
            cells.append(record)
        drawn[cell] = cells
    return drawn


def records_of(drawn: Mapping[Cell, tuple[Draw, ...]]) -> dict[Cell, list[dict[str, object]]]:
    return {cell: [dict(d.record) for d in draws] for cell, draws in drawn.items()}


def test_the_default_spec_logs_nothing():
    assert DrawSpec() == DEFAULT_SPEC
    assert DEFAULT_SPEC.params() == {}
    assert DEFAULT_SPEC.tokens() == ""


def test_a_logic_spec_logs_presence_alone():
    spec = DrawSpec(presence=Presence.LOGIC)
    assert spec.params() == {"presence": "logic"}
    assert spec.tokens() == ""


def test_a_nomap_spec_logs_mapped_false():
    assert NOMAP.params() == {"mapped": False}
    assert NOMAP.tokens() == " mapped=false"


def test_a_logic_presence_without_the_table_is_refused():
    with pytest.raises(ValueError, match="no meaning without the mapping table"):
        DrawSpec(presence=Presence.LOGIC, mapped=False)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_a_nomap_draw_is_a_uniform_pick_per_column_from_the_cell_pool(seed):
    train = varied()
    quota = {YOUNG_WOMEN_1: 25}
    assert records_of(nomap(train, quota, seed=seed)) == oracle(train, quota, seed)


def young_men_only() -> pd.DataFrame:
    """Target 1 records of young and older men alone, so every young women cell is empty."""
    return frame(
        [planted(age=40 + i, sex=1, target=1, thalach=120 + i, chol=200 + i) for i in range(4)]
        + [planted(age=60 + i, sex=1, target=1, thalach=150 + i, chol=300 + i) for i in range(9)]
    )


def test_a_nomap_draw_reads_the_label_group_when_the_cell_is_empty():
    train = young_men_only()
    quota = {YOUNG_WOMEN_1: 40}
    drawn = nomap(train, quota, seed=3)
    assert records_of(drawn) == oracle(train, quota, 3)
    copied = set(UCI_FRAME.columns) - CELL_COLUMNS
    assert all(d.fallback == copied for d in drawn[YOUNG_WOMEN_1])
    assert {d.record["age"] for d in drawn[YOUNG_WOMEN_1]} <= {40, 41, 42, 43}
    assert {d.record["thalach"] for d in drawn[YOUNG_WOMEN_1]} > {120, 121, 122, 123}


def test_a_nomap_draw_names_no_concept_and_a_source_for_every_copied_column():
    train = varied()
    drawn = nomap(train, {YOUNG_WOMEN_1: 20, YOUNG_WOMEN_0: 20}, seed=2)
    for draw in (d for draws in drawn.values() for d in draws):
        assert draw.concepts == draw.closed == frozenset()
        assert set(draw.sources) == set(UCI_FRAME.columns) - CELL_COLUMNS
        for column, source in draw.sources.items():
            assert train.loc[source, column] == draw.record[column]


def test_a_nomap_draw_ignores_the_mapping_rows():
    train = varied()
    quota = {YOUNG_WOMEN_1: 20, YOUNG_WOMEN_0: 20}
    with_rows = nomap(train, quota, seed=4)
    without = nomap(train, quota, seed=4, rows=())
    assert with_rows == without


def test_two_cells_draw_from_one_stream_in_quota_order():
    train = varied()
    forward = {YOUNG_WOMEN_1: 10, YOUNG_WOMEN_0: 10}
    backward = {YOUNG_WOMEN_0: 10, YOUNG_WOMEN_1: 10}
    assert records_of(nomap(train, forward, seed=5)) == oracle(train, forward, 5)
    assert records_of(nomap(train, backward, seed=5)) == oracle(train, backward, 5)
    assert records_of(nomap(train, forward, seed=5)) != records_of(nomap(train, backward, seed=5))


def test_a_nomap_line_carries_mapped_false_before_presence(caplog):
    caplog.set_level(logging.INFO, logger="bko.generate.direct")
    nomap(young_men_only(), {YOUNG_WOMEN_1: 5}, tag=(2, "gap"))
    lines = [r.getMessage() for r in caplog.records if r.name == "bko.generate.direct"]
    direct = [line for line in lines if line.startswith("direct holdout=2 fill=gap ")]
    fallback = [line for line in lines if line.startswith("fallback holdout=2 fill=gap ")]
    assert len(direct) == 1
    assert fallback
    assert all(line.endswith(" mapped=false presence=shares") for line in direct + fallback)
