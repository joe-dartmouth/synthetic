"""`bko-direct`'s sampler under each planted data contradiction."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps.errors import GapError
from bko.gaps.observe import Cell
from bko.gaps.shares import label_only
from bko.generate.direct import Draw, sample
from bko.generate.errors import SamplingError
from bko.ontology.schema import shown
from tests._contradiction_fixtures import (
    BROAD,
    KINDS,
    NAME,
    NARROW,
    OM1,
    YW0,
    YW1,
    Planted,
    plant,
)


def draw(
    planted: Planted, quota: Mapping[Cell, int], train: pd.DataFrame | None = None
) -> dict[Cell, tuple[Draw, ...]]:
    return sample(
        planted.frame if train is None else train,
        quota,
        rows=planted.rows,
        ontologies=planted.ontologies,
        name=NAME,
        seed=0,
        frame=UCI_FRAME,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_fills_its_quota_and_every_record_shows_exactly_its_draw(kind):
    planted = plant(kind)
    drawn = draw(planted, {YW1: 300, YW0: 300, OM1: 300})
    following = label_only(planted.rows, frame=UCI_FRAME)
    for cell, draws in drawn.items():
        assert len(draws) == 300
        for d in draws:
            assert shown(dict(d.record), planted.rows) - following == d.concepts - following, cell
            assert d.concepts <= d.closed


def test_a_drawn_child_never_forces_its_is_a_parent():
    draws = draw(plant("narrow-without-broad"), {YW1: 300})[YW1]
    alone = [d for d in draws if NARROW in d.concepts and BROAD not in d.concepts]
    assert alone, "the cell's records show the child without its parent"
    assert all(BROAD in d.closed and d.record["exang"] == 0 for d in alone)


def test_an_unmatched_draw_falls_back_to_the_label_groups_records():
    draws = draw(plant("narrow-without-broad"), {YW1: 300})[YW1]
    broad = [d for d in draws if BROAD in d.concepts]
    assert broad, "the label group's share drew the parent the cell never shows"
    assert all("exang" in d.fallback and d.record["exang"] == 1 for d in broad)
    assert all("exang" not in d.fallback for d in draws if BROAD not in d.concepts)


def test_a_draw_neither_the_cell_nor_its_group_matches_raises():
    planted = plant("code-overlap")
    train = planted.frame.assign(ca=[(1, 3)[i % 2] for i in range(len(planted.frame))])
    with pytest.raises(
        SamplingError, match=r"^<55/female/1 cannot draw ca: no training record matches the draw$"
    ):
        draw(planted, {YW1: 300}, train)


def test_an_empty_label_group_raises_gap_error_once_before_any_draw(caplog):
    caplog.set_level(logging.INFO)
    planted = plant("empty-cell")
    train = planted.frame.loc[planted.frame[UCI_FRAME.label].eq(0)]
    with pytest.raises(GapError, match=r"^uci-test/target=1 holds no record"):
        draw(planted, {YW0: 5, YW1: 5, OM1: 5}, train)
    assert not [m for m in caplog.messages if m.startswith("direct ")]


def test_an_empty_cell_reads_its_label_group_inside_its_band():
    draws = draw(plant("empty-cell"), {YW1: 50})[YW1]
    assert len(draws) == 50
    assert all(d.record["age"] < 55 and "age" in d.fallback for d in draws)


def test_a_child_share_above_its_two_row_parents_draws_in_one_pass():
    planted = plant("child-share-above-parent")
    assert len([row for row in planted.rows if row.concept == BROAD]) == 2
    first = draw(planted, {YW1: 300})[YW1]
    assert len(first) == 300
    assert first == draw(planted, {YW1: 300})[YW1]
    broad = [d for d in first if BROAD in d.concepts]
    assert broad
    assert all(d.record["exang"] == 1 or d.record["thal"] == 7 for d in broad)
