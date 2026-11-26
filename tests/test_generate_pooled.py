"""The pooled direct draws: `bko-nomap`'s record with a chosen set of fields pooled.

ADR `research/matched-knowledge-controls` (Decision 3) fixes the pooled rule each test reads.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Final, cast

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import LABEL
from bko.datasets.uci import UCI_FRAME
from bko.gaps.observe import Cell
from bko.gaps.shares import GROUP_WEIGHT
from bko.generate.direct import Draw, sample
from bko.generate.donors import _pick, _pick_pooled, _Source
from bko.generate.draw_spec import DrawSpec, Pool
from bko.generate.errors import SamplingError
from bko.generate.fields import field_groups
from bko.generate.pooling import POOL_RANDOM_STREAM, pooled_units
from bko.generate.presence import Presence
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import FieldMapping
from tests.test_generate_direct import NAME, ONTOLOGIES, YOUNG_WOMEN_0, YOUNG_WOMEN_1, varied


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

DONOR: Final = DrawSpec(presence=Presence.POOLED)
POOL_RANDOM: Final = DrawSpec(presence=Presence.POOLED, pool=Pool.RANDOM)
POOL_ALL: Final = DrawSpec(presence=Presence.POOLED, pool=Pool.ALL)
NOMAP: Final = DrawSpec(mapped=False)
QUOTA: Final[Mapping[Cell, int]] = {YOUNG_WOMEN_1: 30, YOUNG_WOMEN_0: 30}
FEATURES: Final = tuple(c for c in UCI_FRAME.columns if c not in {"age", "sex", LABEL})
"""UCI's features less `age` and `sex`: what `Pool.RANDOM` draws from and `Pool.ALL` pools."""


def draw(
    spec: DrawSpec,
    *,
    seed: int = 0,
    rows: Sequence[FieldMapping] = FIELD_MAPPINGS,
    quota: Mapping[Cell, int] = QUOTA,
    train: pd.DataFrame | None = None,
) -> list[Draw]:
    """`quota`'s draws on `varied()` at `spec`, every cell's in quota order."""
    drawn = sample(
        varied() if train is None else train,
        quota,
        rows=rows,
        ontologies=ONTOLOGIES,
        name=NAME,
        seed=seed,
        frame=UCI_FRAME,
        tag=(0, "gap"),
        spec=spec,
    )
    return [d for draws in drawn.values() for d in draws]


class _Fixed:
    """A generator stand-in: `random` gives `uniform`, `integers(n)` gives `n - 1` and is kept."""

    def __init__(self, uniform: float) -> None:
        self.uniform = uniform
        self.sizes: list[int] = []

    def random(self) -> float:
        return self.uniform

    def integers(self, size: int) -> int:
        self.sizes.append(size)
        return size - 1


def _chol(own: list[_Source], other: list[_Source], fixed: _Fixed) -> tuple[_Source, bool]:
    """`_pick_pooled` of `chol` at weight 20, `fixed` standing in for the generator."""
    rng = cast("np.random.Generator", fixed)
    return _pick_pooled(YOUNG_WOMEN_1, ("chol",), own, other, rng, weight=20)


def _sources(ids: range) -> list[_Source]:
    return [(i, {"chol": float(i)}) for i in ids]


def test_a_pooled_spec_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="pool random needs presence pooled"):
        DrawSpec(pool=Pool.RANDOM)
    with pytest.raises(ValueError, match="min_donors needs the mapping table and presence shares"):
        DrawSpec(presence=Presence.POOLED, min_donors=2)
    with pytest.raises(ValueError, match="presence pooled has no meaning without the mapping"):
        DrawSpec(presence=Presence.POOLED, mapped=False)


def test_a_pooled_spec_logs_its_presence_and_its_pool():
    assert DONOR.params() == {"presence": "pooled"}
    assert DONOR.tokens() == ""
    assert POOL_RANDOM.params() == {"presence": "pooled", "pool": "random"}
    assert POOL_RANDOM.tokens() == " pool=random"
    assert POOL_ALL.tokens() == " pool=all"
    assert DrawSpec().values()["pool"] == "mapped"


@pytest.mark.parametrize("spec", [DONOR, POOL_RANDOM, POOL_ALL])
def test_a_pooled_draw_holds_no_concept(spec):
    drawn = draw(spec)
    assert drawn
    assert all(d.concepts == d.closed == frozenset() for d in drawn)


def test_a_mapped_group_is_copied_whole_from_one_donor():
    train = varied()
    assert pooled_units(FIELD_MAPPINGS, UCI_FRAME, Pool.MAPPED, seed=0) == tuple(
        group.fields for group in field_groups(FIELD_MAPPINGS, frame=UCI_FRAME)
    )
    for found in draw(DONOR, train=train):
        assert set(found.sources) == set(UCI_FRAME.columns) - {"sex", LABEL}
        assert found.sources["cp"] == found.sources["exang"]
        assert found.sources["oldpeak"] == found.sources["slope"]
        for field, record_id in found.sources.items():
            assert train.loc[record_id, field] == found.record[field]


@pytest.mark.parametrize(
    ("spec", "rows"),
    [
        (DONOR, tuple(r for r in FIELD_MAPPINGS if r.when[0].field in {"trestbps", "chol"})),
        (POOL_ALL, FIELD_MAPPINGS),
    ],
)
def test_an_unpooled_unit_is_picked_as_nomap_picks_it(spec, rows, monkeypatch):
    """With the pooled pick standing in as nomap's own-first pick, the draw is nomap's.

    Every unit of these specs is one field, so each consumes the generator as nomap's does; equal
    source ids at a seed show the unpooled units' pools, pick and order are nomap's.
    """

    def own_first(cell, fields, own, other, rng, *, weight):
        assert weight == GROUP_WEIGHT
        return _pick(cell, fields, own, other, rng)

    nomap = draw(NOMAP, seed=3, rows=rows)
    monkeypatch.setattr("bko.generate.direct._pick_pooled", own_first)
    pooled = draw(spec, seed=3, rows=rows)
    assert [dict(d.sources) for d in pooled] == [dict(d.sources) for d in nomap]
    assert [dict(d.record) for d in pooled] == [dict(d.record) for d in nomap]


def test_the_pooled_pick_takes_the_group_weight_and_the_whole_label_group(monkeypatch):
    own, other = _sources(range(19)), _sources(range(100, 160))
    below = _Fixed(19 / 39 - 1e-12)
    assert _chol(own, other, below) == (own[18], False)
    assert below.sizes == [19]
    at = _Fixed(19 / 39)
    assert _chol(own, other, at) == (other[59], True)
    assert at.sizes == [60]
    assert _chol([], other, _Fixed(0.0))[1]
    with pytest.raises(SamplingError, match="cannot draw chol"):
        _chol([], [], _Fixed(0.0))

    train = varied()
    seen: list[tuple[Cell, set[int], set[int], int]] = []

    def spy(cell, fields, own, other, rng, *, weight):
        seen.append((cell, {i for i, _ in own}, {i for i, _ in other}, weight))
        return _pick_pooled(cell, fields, own, other, rng, weight=weight)

    monkeypatch.setattr("bko.generate.direct._pick_pooled", spy)
    draw(POOL_ALL, train=train)
    assert seen
    for cell, own_ids, other_ids, weight in seen:
        assert weight == GROUP_WEIGHT
        assert other_ids == set(train.index[train[LABEL].eq(cell[2])])
        assert own_ids < other_ids


@pytest.mark.parametrize("seed", range(6))
def test_pool_random_draws_as_many_fields_as_the_groups_hold(seed):
    held = sum(len(g.fields) for g in field_groups(FIELD_MAPPINGS, frame=UCI_FRAME))
    units = pooled_units(FIELD_MAPPINGS, UCI_FRAME, Pool.RANDOM, seed=seed)
    fields = [field for unit in units for field in unit]
    assert all(len(unit) == 1 for unit in units)
    assert len(fields) == len(set(fields)) == held
    assert set(fields) <= set(FEATURES)
    assert fields == [c for c in UCI_FRAME.columns if c in fields]


def test_pool_random_fields_follow_the_seed_and_are_logged(caplog):
    assert POOL_RANDOM_STREAM == 19
    one = tuple(r for r in FIELD_MAPPINGS if r.when[0].field == "chol")
    chosen = {pooled_units(one, UCI_FRAME, Pool.RANDOM, seed=s) for s in range(8)}
    assert len(chosen) > 1
    picked = np.random.default_rng([5, POOL_RANDOM_STREAM]).choice(
        len(FEATURES), size=1, replace=False
    )
    assert pooled_units(one, UCI_FRAME, Pool.RANDOM, seed=5) == ((FEATURES[int(picked[0])],),)

    caplog.set_level(logging.INFO, logger="bko.generate.pooling")
    drawn = draw(POOL_RANDOM, seed=5)
    assert drawn == draw(POOL_RANDOM, seed=5)
    units = pooled_units(FIELD_MAPPINGS, UCI_FRAME, Pool.RANDOM, seed=5)
    fields = ",".join(field for unit in units for field in unit)
    logged = [m for m in caplog.messages if m.startswith("pool-random ")]
    assert logged == [f"pool-random holdout=0 fill=gap fields={fields}"] * 2


def test_pool_all_pools_every_feature_but_age_and_sex():
    units = tuple((field,) for field in FEATURES)
    assert pooled_units(FIELD_MAPPINGS, UCI_FRAME, Pool.ALL, seed=0) == units
    assert pooled_units((), UCI_FRAME, Pool.ALL, seed=4) == units
    rows = tuple(r for r in FIELD_MAPPINGS if r.when[0].field != "thal")
    assert draw(POOL_ALL, rows=rows) == draw(POOL_ALL)
