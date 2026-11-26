"""The micro draw: `DrawSpec.microaggregate` averages a unit's numeric fields over k donors.

ADR `research/sampler-donor-limits` fixes the rule each test reads.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps.observe import Cell
from bko.generate.direct import Draw, sample
from bko.generate.donors import _Limits, _Source
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
from bko.generate.errors import SamplingError
from bko.generate.micro import Means
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import AtLeast
from tests._validate_fixtures import planted
from tests.test_generate_capped import CAPPED, ELEVATED
from tests.test_generate_direct import NAME, ONTOLOGIES, YOUNG_WOMEN_1, frame, varied


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

MICRO: Final = DrawSpec(min_donors=2, reuse_cap=1, microaggregate=3)
"""The micro spec: `bko-capped`'s limits, numeric fields averaged over 3 donors."""

AVERAGED: Final = frozenset({"trestbps", "chol", "thalach", "oldpeak"})
"""UCI's numeric fields but `age`: the fields a micro draw averages."""

CLAUSES: Final = (AtLeast("trestbps", 130), AtLeast("chol", 240), AtLeast("oldpeak", 1.0))
"""UCI's `AtLeast` clauses."""


def draw(
    train: pd.DataFrame, quota: Mapping[Cell, int], spec: DrawSpec = MICRO, seed: int = 0
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


def sources(field: str, values: Sequence[float], start: int = 1) -> list[_Source]:
    """One record per value, holding `field` and `slope`, with ids from `start`."""
    return [(start + i, {field: value, "slope": 2}) for i, value in enumerate(values)]


def mean(
    own: Sequence[_Source],
    other: Sequence[_Source] = (),
    *,
    fields: tuple[str, ...] = ("chol",),
    k: int = 3,
    limits: _Limits | None = None,
    seed: int = 0,
) -> tuple[_Source, bool, tuple[int, ...]]:
    """`Means.pick` on `own` and `other` at `MICRO`'s limits unless `limits` is given."""
    means = Means(k, CLAUSES, UCI_FRAME, _Limits(2, 1, k) if limits is None else limits)
    return means.pick(YOUNG_WOMEN_1, fields, own, other, np.random.default_rng(seed))


def young_women(values: Sequence[Mapping[str, object]], men: int = 0) -> pd.DataFrame:
    """Young women of label 1 with `values` each, then `men` young men of label 1."""
    women = [planted(age=40, sex=0, target=1, **changes) for changes in values]
    return frame(women + [planted(age=41, sex=1, target=1) for _ in range(men)])


def test_a_microaggregated_value_stays_on_its_concepts_side_of_the_cutoff():
    pressures = [128, 129, 129, 127, 120, 125, 130, 130, 131, 133, 150, 140] * 3
    train = young_women([{"trestbps": value} for value in pressures])
    pressure = train["trestbps"].to_dict()
    for seed in range(5):
        drawn = draw(train, {YOUNG_WOMEN_1: 10}, seed=seed)
        assert {ELEVATED in d.concepts for d in drawn} == {True, False}
        for found in drawn:
            assert (found.record["trestbps"] >= 130) == (ELEVATED in found.concepts)
            donors = [pressure[i] >= 130 for i in found.averaged["trestbps"]]
            assert set(donors) == {ELEVATED in found.concepts}


def test_a_mean_differs_from_each_donor_on_a_spread_fixture():
    (source, chosen), fell, ids = mean(sources("chol", [1, 2, 6]))
    assert chosen["chol"] == 3
    assert chosen["chol"] not in {1, 2, 6}
    assert sorted(ids) == [1, 2, 3]
    assert source == ids[0]
    assert not fell


def test_a_pool_that_repeats_a_record_still_averages_distinct_donors():
    own = sources("chol", [1, 2, 6])
    repeated = [own[0], own[0], own[0], own[1], own[0], own[2]]
    for seed in range(10):
        _, fell, ids = mean(repeated, seed=seed)
        assert sorted(ids) == [1, 2, 3]
        assert not fell
    _, fell, ids = mean([own[0]] * 4 + [own[1]], other=sources("chol", [5, 5, 5], start=7))
    assert fell, "two distinct records are fewer than k, however often one repeats"
    assert sorted(ids) == [7, 8, 9]


def test_a_mean_only_donor_counts_one_kth():
    own = sources("chol", [1, 2, 6])
    other = [*own, *sources("chol", [3, 4, 5], start=4)]
    limits = _Limits(2, 1, 3)
    for _ in range(3):
        _, fell, ids = mean(own, other, limits=limits)
        assert (sorted(ids), fell) == ([1, 2, 3], False)
    assert limits.used[("chol",), 1] == 3
    _, fell, ids = mean(own, other, limits=limits)
    assert (sorted(ids), fell) == ([4, 5, 6], True)


def test_a_donor_already_in_a_mean_is_refused_as_a_primary_of_a_coded_unit():
    fields = ("oldpeak", "slope")
    own = sources("oldpeak", [0.5, 0.6, 0.7, 0.8])
    entered = []
    for seed in range(20):
        limits = _Limits(2, 1, 3)
        limits.used[fields, 1] = 1
        (source, _), _, ids = mean(own, fields=fields, limits=limits, seed=seed)
        assert source != 1
        assert limits.used[fields, source] == 3, "a primary of a coded unit weighs k"
        if 1 in ids:
            entered.append(seed)
            assert limits.used[fields, 1] == 2, "a mean-only donor weighs 1"
    assert entered, "donor 1 never entered a mean as a non-primary donor"


def test_an_own_pool_with_fewer_than_k_free_donors_falls_back():
    train = young_women([{} for _ in range(19)], men=400)
    drawn = draw(train, {YOUNG_WOMEN_1: 100})
    assert len(drawn) == 100
    fell = [index for index, d in enumerate(drawn) if "chol" in d.fallback]
    assert fell
    assert 0 not in fell
    assert any(i > 19 for index in fell for i in drawn[index].averaged["chol"])


def test_a_two_record_cell_falls_back_at_k_3():
    train = young_women([{}, {}], men=60)
    for found in draw(train, {YOUNG_WOMEN_1: 5}):
        assert found.fallback >= AVERAGED
        assert set(found.averaged) == AVERAGED
        assert all(i > 2 for ids in found.averaged.values() for i in ids)


def test_a_coded_only_group_is_picked_as_capped_picks_it():
    train = varied()
    drawn = draw(train, {YOUNG_WOMEN_1: 6})
    coded = ("cp", "exang", "fbs", "restecg", "ca", "thal")
    for found in drawn:
        assert set(found.averaged) == AVERAGED
        for field in coded:
            assert found.record[field] == train.loc[found.sources[field], field]
    assert max(Counter((f, d.sources[f]) for d in drawn for f in coded).values()) == 1


def test_age_is_copied_not_averaged():
    train = varied()
    for found in draw(train, {YOUNG_WOMEN_1: 6}):
        assert "age" not in found.averaged
        assert found.record["age"] == train.loc[found.sources["age"], "age"]


def test_an_integer_mean_rounds_half_even():
    (_, low), _, _ = mean(sources("chol", [1, 2]), k=2)
    (_, high), _, _ = mean(sources("chol", [2, 3]), k=2)
    assert (low["chol"], high["chol"]) == (2, 2)


def test_a_fractional_mean_rounds_to_the_frames_digits():
    (_, chosen), _, _ = mean(sources("oldpeak", [0.1, 0.2, 0.2]), fields=("oldpeak", "slope"))
    assert repr(chosen["oldpeak"]) == "0.2"


def test_a_fractional_mean_at_a_half_rounds_to_the_even_digit():
    (_, chosen), _, _ = mean(sources("oldpeak", [0.3, 0.4]), fields=("oldpeak", "slope"), k=2)
    assert repr(chosen["oldpeak"]) == "0.4"


def test_a_mean_rounded_to_zero_is_never_negative_zero():
    own = sources("oldpeak", [-0.1, 0.0, 0.0])
    (_, chosen), _, _ = mean(own, fields=("oldpeak", "slope"))
    assert repr(chosen["oldpeak"]) == "0.0"


def test_a_means_donors_share_the_side_of_a_numeric_cutoff():
    chol = dict(enumerate([200, 201, 202, 300, 301, 302], start=1))
    sides = set()
    for seed in range(20):
        _, fell, ids = mean(sources("chol", list(chol.values())), seed=seed)
        assert not fell
        (side,) = {chol[i] >= 240 for i in ids}
        sides.add(side)
    assert sides == {True, False}


def test_a_means_donors_share_the_units_coded_values():
    own = [(1 + i, {"oldpeak": 0.1 * i, "slope": 1 + i // 3}) for i in range(6)]
    slope = {source: values["slope"] for source, values in own}
    slopes = set()
    for seed in range(20):
        _, fell, ids = mean(own, fields=("oldpeak", "slope"), seed=seed)
        assert not fell
        (shared,) = {slope[i] for i in ids}
        slopes.add(shared)
    assert slopes == {1, 2}


def test_a_missing_averaged_value_is_refused_naming_its_field():
    own = sources("chol", [200, math.nan, 210, 220])
    with pytest.raises(SamplingError, match="record 2 holds no chol value to average"):
        mean(own)


def test_no_donor_class_anywhere_is_refused():
    own = sources("chol", [1, 2])
    other = [*own, *sources("chol", [250, 260], start=3)]
    with pytest.raises(SamplingError, match="cannot draw chol: no class of 3 donors is left"):
        mean(own, other)


def micro_draw(value: float, ids: tuple[int, ...]) -> Draw:
    """A draw whose `chol` is `value`, averaged over `ids`, the first its primary."""
    record = {"chol": value}
    return Draw(
        YOUNG_WOMEN_1,
        record,
        frozenset(),
        frozenset(),
        frozenset(),
        {"chol": ids[0]},
        {"chol": ids},
    )


def test_the_micro_field_line_counts_equal_and_primary_over_two_classes(caplog):
    caplog.set_level(logging.INFO)
    records = sources("chol", [200, 210, 230, 250, 250, 250])
    draws = [micro_draw(213, (1, 2, 3)), micro_draw(250, (4, 5, 6))]
    Means(3, CLAUSES, UCI_FRAME, _Limits(2, 1, 3)).log(0, "gap", YOUNG_WOMEN_1, draws, records)
    assert (
        "micro-field holdout=0 fill=gap key=<55/female/1 field=chol floor=3 values=2 equal=1"
        " primary=1 under=0 classless=0 sse=466.667 sst=2483.33"
    ) in caplog.messages


def test_each_averaged_field_has_a_micro_field_line_in_column_order(caplog):
    caplog.set_level(logging.INFO)
    Means(4, CLAUSES, UCI_FRAME, _Limits(5, 1, 4)).log(1, "gap", YOUNG_WOMEN_1, (), ())
    assert caplog.messages == [
        f"micro-field holdout=1 fill=gap key=<55/female/1 field={name} floor=5 values=0"
        " equal=0 primary=0 under=0 classless=0 sse=0 sst=0"
        for name in ("trestbps", "chol", "thalach", "oldpeak")
    ]
    caplog.clear()
    draw(young_women([{} for _ in range(6)]), {YOUNG_WOMEN_1: 1})
    fields = [m.split(" field=")[1].split()[0] for m in caplog.messages if "micro-field" in m]
    assert fields == ["trestbps", "chol", "thalach", "oldpeak"]


def tallied(own: Sequence[_Source], other: Sequence[_Source], **used: int) -> Means:
    """A `Means` after one `chol` pick on `own` and `other`, `used` seeding record ids' counts."""
    means = Means(3, CLAUSES, UCI_FRAME, _Limits(2, 1, 3))
    for source, count in used.items():
        means.limits.used[("chol",), int(source.removeprefix("r"))] = count
    means.pick(YOUNG_WOMEN_1, ("chol",), own, other, np.random.default_rng(0))
    return means


def test_an_own_pool_under_the_floor_counts_a_redirect_and_under(caplog):
    caplog.set_level(logging.INFO)
    own = sources("chol", [200, 201])
    means = tallied(own, [*own, *sources("chol", [202, 203, 204], start=3)])
    assert (means.limits.tally.redirects, means.limits.tally.skips) == (1, 0)
    means.log(0, "gap", YOUNG_WOMEN_1, (), ())
    (line,) = [m for m in caplog.messages if " field=chol " in m]
    assert " floor=3 " in line
    assert " under=1 classless=0 " in line


def test_an_own_pool_with_no_class_of_k_counts_classless(caplog):
    caplog.set_level(logging.INFO)
    own = sources("chol", [200, 250, 260])
    means = tallied(own, [*own, *sources("chol", [270, 280], start=4)])
    assert (means.limits.tally.redirects, means.limits.tally.skips) == (0, 0)
    means.log(0, "gap", YOUNG_WOMEN_1, (), ())
    (line,) = [m for m in caplog.messages if " field=chol " in m]
    assert " under=0 classless=1 " in line
    caplog.clear()
    means.log(0, "gap", YOUNG_WOMEN_1, (), ())
    (line,) = [m for m in caplog.messages if " field=chol " in m]
    assert " under=0 classless=0 " in line, "each log restarts the counts"


def test_a_cap_skip_is_counted_at_the_picks_own_weight():
    fields = ("oldpeak", "slope")
    limits = _Limits(2, 1, 3)
    limits.used[fields, 1] = 1
    mean(sources("oldpeak", [0.5, 0.6, 0.7, 0.8]), fields=fields, limits=limits)
    assert limits.tally.skips == 1, "donor 1 is free for a mean, not as a primary of weight 3"
    limits = _Limits(2, 1, 3)
    limits.used[("chol",), 1] = 2
    mean(sources("chol", [200, 201, 202, 203]), limits=limits)
    assert limits.tally.skips == 0, "donor 1 is free at a numeric unit's weight of 1"


def test_microaggregate_logs_only_when_set(caplog):
    assert MICRO.params() == {"min_donors": 2, "reuse_cap": 1, "microaggregate": 3}
    assert MICRO.tokens() == " min_donors=2 reuse_cap=1 microaggregate=3"
    assert "microaggregate" not in DEFAULT_SPEC.params()
    assert DEFAULT_SPEC.values()["microaggregate"] is None
    caplog.set_level(logging.INFO)
    for found in draw(varied(), {YOUNG_WOMEN_1: 6}, CAPPED):
        assert found.averaged == {}
    assert not any(m.startswith("micro-field ") for m in caplog.messages)


def test_microaggregate_alone_draws_every_averaged_field():
    drawn = draw(varied(), {YOUNG_WOMEN_1: 6}, DrawSpec(microaggregate=3))
    assert len(drawn) == 6
    for found in drawn:
        assert set(found.averaged) == AVERAGED
        assert all(len(set(ids)) == 3 for ids in found.averaged.values())


def test_a_fill_with_no_class_of_k_anywhere_names_its_holdout_and_fill():
    message = (
        f"{NAME} frame=uci holdout=0 fill=gap cannot draw: <55/female/1 cannot draw"
        " trestbps: no class of 3 donors is left"
    )
    with pytest.raises(SamplingError, match=f"^{message}"):
        draw(young_women([{}, {}]), {YOUNG_WOMEN_1: 1})


def test_microaggregate_below_2_is_refused():
    with pytest.raises(ValueError, match="microaggregate 1 must be at least 2"):
        DrawSpec(microaggregate=1)


@pytest.mark.parametrize("value", [True, 3.0])
def test_microaggregate_other_than_an_int_is_refused(value):
    with pytest.raises(ValueError, match=f"microaggregate {value!r} must be an int"):
        DrawSpec(microaggregate=value)


def test_microaggregate_without_the_mapping_table_is_refused():
    with pytest.raises(ValueError, match="needs the mapping table and presence shares"):
        DrawSpec(microaggregate=3, mapped=False)
