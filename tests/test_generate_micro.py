"""The micro draw: `DrawSpec.microaggregate` averages a unit's numeric fields over k donors.

ADR `research/sampler-donor-limits` fixes the rule each test reads.
"""

from __future__ import annotations

import logging
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
from bko.generate.micro import pick_mean
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
    """`pick_mean` on `own` and `other` at `MICRO`'s limits unless `limits` is given."""
    return pick_mean(
        YOUNG_WOMEN_1,
        fields,
        own,
        other,
        np.random.default_rng(seed),
        k=k,
        limits=_Limits(2, 1, k) if limits is None else limits,
        frame=UCI_FRAME,
        clauses=CLAUSES,
    )


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


def test_a_donor_used_twice_in_means_is_refused_as_a_primary():
    fields = ("oldpeak", "slope")
    own = sources("oldpeak", [0.5, 0.6, 0.7, 0.8])
    entered = []
    for seed in range(20):
        limits = _Limits(2, 1, 3)
        limits.used[fields, 1] = 2
        (source, _), _, ids = mean(own, fields=fields, limits=limits, seed=seed)
        assert source != 1
        assert limits.used[fields, source] == 3, "a primary of a coded unit weighs k"
        if 1 in ids:
            entered.append(seed)
            assert limits.used[fields, 1] == 3, "a mean-only donor weighs 1"
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


def test_no_donor_class_anywhere_is_refused():
    own = sources("chol", [1, 2])
    other = [*own, *sources("chol", [250, 260], start=3)]
    with pytest.raises(SamplingError, match="cannot draw chol: no class of 3 donors is left"):
        mean(own, other)


def test_the_micro_field_line_counts_equal_and_primary(caplog):
    caplog.set_level(logging.INFO)
    train = young_women([{"chol": 1}, {"chol": 2}, {"chol": 2}])
    (found,) = draw(train, {YOUNG_WOMEN_1: 1})
    assert found.record["chol"] == 2
    primary = int(train.loc[found.sources["chol"], "chol"] == 2)
    assert (
        f"micro-field holdout=0 fill=gap key=<55/female/1 field=chol values=1 equal=1"
        f" primary={primary} sse=0.666667 sst=0.666667"
    ) in caplog.messages
    fields = [m.split(" field=")[1].split()[0] for m in caplog.messages if "micro-field" in m]
    assert fields == ["trestbps", "chol", "thalach", "oldpeak"]


def test_microaggregate_logs_only_when_set(caplog):
    assert MICRO.params() == {"min_donors": 2, "reuse_cap": 1, "microaggregate": 3}
    assert MICRO.tokens() == " min_donors=2 reuse_cap=1 microaggregate=3"
    assert "microaggregate" not in DEFAULT_SPEC.params()
    assert DEFAULT_SPEC.values()["microaggregate"] is None
    caplog.set_level(logging.INFO)
    for found in draw(varied(), {YOUNG_WOMEN_1: 6}, CAPPED):
        assert found.averaged == {}
    assert not any(m.startswith("micro-field ") for m in caplog.messages)


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
