"""The donor floor and the reuse cap: `DrawSpec.min_donors` and `DrawSpec.reuse_cap` on a draw.

ADR `research/sampler-donor-limits` fixes the rule each test reads.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from typing import Final

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import LABEL
from bko.datasets.uci import UCI_FRAME
from bko.gaps.observe import Cell
from bko.generate.direct import Draw, sample
from bko.generate.donors import _pick_limited, _Tally
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
from bko.generate.errors import SamplingError
from bko.generate.presence import Presence
from bko.ontology.mapping import FIELD_MAPPINGS
from tests._validate_fixtures import planted
from tests.test_generate_direct import NAME, ONTOLOGIES, YOUNG_WOMEN_0, YOUNG_WOMEN_1, frame, varied


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CAPPED: Final = DrawSpec(min_donors=2, reuse_cap=1)
"""`bko-capped`'s spec: a floor of 2 cell donors and a cap of 1 per unit."""

ELEVATED: Final = "HP:0004421"
"""Elevated blood pressure, which `trestbps` 150 shows and 120 does not."""


def draw(
    train: pd.DataFrame, quota: Mapping[Cell, int], spec: DrawSpec, seed: int = 0
) -> list[Draw]:
    """`quota`'s draws at `spec`, every cell's in quota order."""
    drawn = sample(
        train,
        quota,
        rows=FIELD_MAPPINGS,
        ontologies=ONTOLOGIES,
        name=NAME,
        seed=seed,
        frame=UCI_FRAME,
        tag=(0, "gap"),
        spec=spec,
    )
    return [d for draws in drawn.values() for d in draws]


def most_reused(draws: list[Draw]) -> int:
    """The most draws one training record gives one field to: a unit's fields share a source."""
    per_field = Counter((field, source) for d in draws for field, source in d.sources.items())
    return max(per_field.values())


def singleton(pressured: int) -> pd.DataFrame:
    """Five young women of label 1, `pressured` of them at `trestbps` 150, and 60 young men of
    label 1, half at 150: the men are the women's label group in their age band."""
    women = [planted(age=40, sex=0, target=1, trestbps=150) for _ in range(pressured)]
    women += [planted(age=40, sex=0, target=1, trestbps=120) for _ in range(5 - pressured)]
    men = [planted(age=41, sex=1, target=1, trestbps=150 - 30 * (i % 2)) for i in range(60)]
    return frame(women + men)


def test_the_default_spec_logs_nothing():
    assert DEFAULT_SPEC.params() == {}
    assert DEFAULT_SPEC.tokens() == ""


def test_a_capped_spec_logs_both_limits():
    assert CAPPED.params() == {"min_donors": 2, "reuse_cap": 1}
    assert CAPPED.tokens() == " min_donors=2 reuse_cap=1"


@pytest.mark.parametrize(
    ("changes", "message"),
    [({"min_donors": 0}, "min_donors 0"), ({"reuse_cap": 0}, "reuse_cap 0")],
)
def test_a_limit_below_1_is_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        DrawSpec(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"mapped": False, "min_donors": 2},
        {"mapped": False, "reuse_cap": 1},
        {"presence": Presence.LOGIC, "min_donors": 2},
        {"presence": Presence.LOGIC, "reuse_cap": 1},
    ],
)
def test_a_limit_is_refused_without_the_table_or_beside_logic(changes):
    with pytest.raises(ValueError, match="needs the mapping table and presence shares"):
        DrawSpec(**changes)


def test_draws_never_exceed_the_reuse_cap():
    train = varied()
    quota = {YOUNG_WOMEN_1: 6, YOUNG_WOMEN_0: 6}
    assert most_reused(draw(train, quota, DEFAULT_SPEC)) > 1
    for seed in range(5):
        assert most_reused(draw(train, quota, CAPPED, seed=seed)) == 1


def test_a_singleton_pattern_is_never_copied_from_the_cell():
    train = singleton(pressured=1)
    plain = draw(train, {YOUNG_WOMEN_1: 20}, DEFAULT_SPEC)
    high = [d for d in plain if ELEVATED in d.concepts]
    assert high, "a share above 0 drew no elevated pressure in 20 records"
    assert {d.sources["trestbps"] for d in high} == {1}
    capped = [d for d in draw(train, {YOUNG_WOMEN_1: 20}, CAPPED) if ELEVATED in d.concepts]
    assert capped
    for found in capped:
        assert found.sources["trestbps"] > 5
        assert "trestbps" in found.fallback
        assert found.record["trestbps"] == 150


def test_a_pattern_no_cell_record_shows_is_no_floor_redirect(caplog):
    caplog.set_level(logging.INFO)
    drawn = draw(singleton(pressured=0), {YOUNG_WOMEN_1: 20}, CAPPED)
    assert any(ELEVATED in d.concepts for d in drawn)
    (line,) = [m for m in caplog.messages if m.startswith("limits ")]
    assert "floor_redirects=0 " in line


def test_a_pattern_at_the_floor_is_read_from_the_cell():
    train = singleton(pressured=2)
    draws = draw(train, {YOUNG_WOMEN_1: 20}, DrawSpec(min_donors=2))
    high = [d for d in draws if ELEVATED in d.concepts]
    assert high
    assert {d.sources["trestbps"] for d in high} <= {1, 2}
    assert not any("trestbps" in d.fallback for d in high)


SOURCES: Final = [(1, {"x": 1.0}), (2, {"x": 2.0})]
OTHERS: Final = [(1, {"x": 1.0}), (2, {"x": 2.0}), (7, {"x": 7.0})]


def _picked(
    used: Counter[tuple[tuple[str, ...], int]],
    *,
    floor: int = 1,
    other: list[tuple[int, dict[str, float]]] = OTHERS,
    tally: _Tally | None = None,
) -> tuple[int, bool]:
    (source, _), fell = _pick_limited(
        YOUNG_WOMEN_1,
        ("x",),
        SOURCES,
        other,
        np.random.default_rng(0),
        floor=floor,
        cap=1,
        used=used,
        tally=tally,
    )
    return source, fell


def test_a_donor_at_the_cap_is_skipped():
    used: Counter[tuple[tuple[str, ...], int]] = Counter({(("x",), 1): 1})
    assert _picked(Counter(used)) == (2, False)
    used[("y",), 2] = 1
    assert _picked(Counter(used)) == (2, False), "a cap counts per unit"
    used[("x",), 2] = 1
    assert _picked(used) == (7, True)
    assert used[("x",), 7] == 1


def test_below_the_floor_the_cells_own_donors_are_barred_from_the_label_group():
    assert _picked(Counter(), floor=3) == (7, True)


def test_below_the_floor_with_no_label_group_donor_but_the_cells_raises_sampling_error():
    with pytest.raises(SamplingError, match="no matching donor is left at min_donors=3"):
        _picked(Counter(), floor=3, other=SOURCES)


def test_the_tally_counts_a_redirect_only_for_a_cell_with_a_matching_record():
    tally = _Tally()
    _pick_limited(
        YOUNG_WOMEN_1,
        ("x",),
        [],
        OTHERS,
        np.random.default_rng(0),
        floor=2,
        cap=1,
        used=Counter(),
        tally=tally,
    )
    assert (tally.redirects, tally.skips) == (0, 0)
    _picked(Counter(), floor=3, tally=tally)
    assert (tally.redirects, tally.skips) == (1, 0)


def test_the_tally_counts_each_pick_whose_first_pool_held_a_capped_donor():
    tally = _Tally()
    used: Counter[tuple[tuple[str, ...], int]] = Counter()
    picked = [_picked(used, tally=tally) for _ in range(3)]
    assert sorted(source for source, _ in picked) == [1, 2, 7]
    assert tally.skips == 2, "the second and third picks skip a capped cell donor"
    others = [*OTHERS, (8, {"x": 8.0})]
    assert _picked(used, floor=3, other=others, tally=tally) == (8, True)
    assert (tally.redirects, tally.skips) == (1, 3), "a floor-fail pick skips capped donor 7"


def test_exhausted_pools_raise_sampling_error():
    used: Counter[tuple[tuple[str, ...], int]] = Counter(
        {(("x",), source): 1 for source in (1, 2, 7)}
    )
    with pytest.raises(SamplingError, match=r"<55/female/1 cannot draw x: .* reuse_cap=1"):
        _picked(used)
    train = singleton(pressured=1)
    message = "<55/female/1 cannot draw trestbps: no matching donor is left at min_donors=2"
    with pytest.raises(SamplingError, match=f"^{message} reuse_cap=1$"):
        draw(train, {YOUNG_WOMEN_1: 66}, CAPPED)


def test_the_capped_draw_is_deterministic_per_seed():
    train = varied()
    quota = {YOUNG_WOMEN_1: 6, YOUNG_WOMEN_0: 6}
    assert draw(train, quota, CAPPED, seed=7) == draw(train, quota, CAPPED, seed=7)
    assert draw(train, quota, CAPPED, seed=7) != draw(train, quota, CAPPED, seed=8)


def test_each_cells_limits_line_carries_its_own_counts(caplog):
    caplog.set_level(logging.INFO)
    rows = []
    for label in (0, 1):
        rows += [planted(age=40, sex=0, target=label, trestbps=150)]
        rows += [planted(age=40, sex=0, target=label, trestbps=120) for _ in range(4)]
        rows += [
            planted(age=41, sex=1, target=label, trestbps=150 - 30 * (i % 2)) for i in range(60)
        ]
    drawn = draw(frame(rows), {YOUNG_WOMEN_1: 20, YOUNG_WOMEN_0: 20}, CAPPED, seed=1)
    lines = [m for m in caplog.messages if m.startswith("limits ")]
    redirects = [
        sum(ELEVATED in d.concepts for d in drawn if d.cell == cell)
        for cell in (YOUNG_WOMEN_1, YOUNG_WOMEN_0)
    ]
    assert redirects[0] != redirects[1]
    assert [line.partition(" cap_skips=")[0] for line in lines] == [
        f"limits holdout=0 fill=gap key=<55/female/1 floor_redirects={redirects[0]}",
        f"limits holdout=0 fill=gap key=<55/female/0 floor_redirects={redirects[1]}",
    ]


def test_the_limits_line_counts_redirects_and_skips(caplog):
    caplog.set_level(logging.INFO)
    train = singleton(pressured=1)
    drawn = draw(train, {YOUNG_WOMEN_1: 20}, CAPPED)
    (line,) = [m for m in caplog.messages if m.startswith("limits ")]
    redirects = sum(ELEVATED in d.concepts for d in drawn)
    assert line.startswith(
        f"limits holdout=0 fill=gap key=<55/female/1 floor_redirects={redirects} cap_skips="
    )
    assert int(line.rpartition("cap_skips=")[2]) > 0
    caplog.clear()
    draw(train, {YOUNG_WOMEN_1: 20}, DEFAULT_SPEC)
    assert not any(m.startswith("limits ") for m in caplog.messages)
    assert {d.record[LABEL] for d in drawn} == {1}
