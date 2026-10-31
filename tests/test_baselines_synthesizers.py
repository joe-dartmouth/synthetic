"""CTGAN and TVAE: the rejection fill on a fake model, and short real fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

from bko.baselines.errors import BaselineError, ShortfallError
from bko.baselines.records import unit_weights
from bko.baselines.request import Request
from bko.baselines.synthesizers import (
    CTGAN_PARAMETERS,
    TVAE_PARAMETERS,
    build_synthesizer,
    ctgan,
    fill,
    seeded,
    tvae,
)
from tests._baselines_fixtures import (
    QUOTA,
    TINY_CTGAN,
    TINY_TVAE,
    assert_contract,
    cell_list,
    full_records,
)


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TRAIN = full_records()
ADAPTERS = {"ctgan": (ctgan, TINY_CTGAN), "tvae": (tvae, TINY_TVAE)}


class FakeModel:
    """Draws from `pool` with replacement; `calls` records each request size."""

    def __init__(self, pool: pd.DataFrame, short: int = 0) -> None:
        self.pool, self.short = pool, short
        self.calls: list[int] = []

    def __call__(self, rows: int) -> pd.DataFrame:
        self.calls.append(rows)
        return self.pool.sample(rows - self.short, replace=True, random_state=len(self.calls))


def test_fill_meets_the_quota_stratum_by_stratum():
    request = Request(quota=QUOTA)
    got = fill(FakeModel(TRAIN), request, method="fake", batch=50, limit=10_000)
    assert cell_list(got) == [stratum for stratum, n in request.quota.items() for _ in range(n)]
    assert got.index.tolist() == list(range(-1, -len(got) - 1, -1))


def test_fill_takes_a_stratums_first_matches_in_draw_order():
    pool = TRAIN.iloc[:40]
    stratum = cell_list(pool)[0]
    got = fill(FakeModel(pool), Request(quota={stratum: 2}), method="fake", batch=40, limit=40)
    first = FakeModel(pool)(40)
    expected = first.loc[[cell == stratum for cell in cell_list(first)]].head(2)
    assert got.to_numpy().tolist() == expected.to_numpy().tolist()


def test_fill_refuses_a_stratum_the_model_never_reaches():
    model = FakeModel(TRAIN.loc[TRAIN["sex"] == 1])
    with pytest.raises(ShortfallError, match="fake asked for 200 rows and still lacks") as caught:
        fill(model, Request(quota=QUOTA), method="fake", batch=50, limit=200)
    assert model.calls == [50, 50, 50, 50]
    error = caught.value
    assert error.baseline is None
    assert set(error.short) == {"/".join(map(str, s)) for s in QUOTA if s[1] == "female"}
    kept = {"/".join(map(str, s)) for s in cell_list(error.added)}
    assert kept.isdisjoint(error.short)
    assert len(error.added) == sum(QUOTA.values()) - sum(error.short.values())


def test_fill_tolerates_a_short_batch():
    model = FakeModel(TRAIN, short=1)
    got = fill(model, Request(quota=QUOTA), method="fake", batch=50, limit=10_000)
    assert sorted(cell_list(got)) == sorted(s for s, n in QUOTA.items() for _ in range(n))


def test_seeded_restores_the_global_generators_and_threads():
    np.random.seed(11)
    torch.manual_seed(11)
    threads = torch.get_num_threads()
    with seeded(5):
        assert torch.get_num_threads() == 1
        np.random.random()
        torch.rand(1)
    after = (np.random.random(), torch.rand(1).item(), torch.get_num_threads())
    np.random.seed(11)
    torch.manual_seed(11)
    assert after == (np.random.random(), torch.rand(1).item(), threads)


@pytest.fixture(scope="module", params=sorted(ADAPTERS))
def runs(request):
    """Three runs per synthesizer: (ambient 1, seed 3), (ambient 2, seed 3), (ambient 1, seed 4).

    Outputs are shared, never fitted models: sampling advances a fitted model's stream.
    """
    adapter, parameters = ADAPTERS[request.param]

    def run(ambient, seed):
        np.random.seed(ambient)
        torch.manual_seed(ambient)
        return adapter(TRAIN, Request(quota=QUOTA, seed=seed), parameters=parameters)

    return run(1, 3), run(2, 3), run(1, 4)


def test_a_short_fit_meets_the_contract(runs):
    assert_contract(runs[0].added, TRAIN, QUOTA)
    pd.testing.assert_series_equal(runs[0].weights, unit_weights(TRAIN))


def test_the_seed_alone_decides_the_rows(runs):
    pd.testing.assert_frame_equal(runs[0].added, runs[1].added)


def test_a_different_seed_gives_different_rows(runs):
    """The positive control: an adapter that seeds with a constant passes the test above."""
    assert not runs[0].added.equals(runs[2].added)


@pytest.mark.parametrize(
    ("factory", "parameters"), [(CTGANSynthesizer, TINY_CTGAN), (TVAESynthesizer, TINY_TVAE)]
)
def test_training_is_built_for_the_cpu(factory, parameters):
    """Without `enable_gpu=False`, ctgan picks MPS on Apple Silicon and CUDA where present."""
    assert build_synthesizer(factory, parameters).enable_gpu is False


@pytest.mark.parametrize(
    ("adapter", "parameters"), [(ctgan, CTGAN_PARAMETERS), (tvae, TVAE_PARAMETERS)]
)
def test_the_locked_parameters_fit_this_frame(adapter, parameters):
    """SDV's defaults with one epoch: batch 500 on 112 records, the geometry the real run uses."""
    result = adapter(TRAIN, Request(quota=QUOTA), parameters={**parameters, "epochs": 1})
    assert_contract(result.added, TRAIN, QUOTA)


def test_fill_refuses_a_request_beyond_its_limit():
    with pytest.raises(BaselineError, match="draws at most 5"):
        fill(FakeModel(TRAIN), Request(quota=QUOTA), method="fake", batch=5, limit=5)


def _raise_while_seeded() -> None:
    with seeded(5):
        np.random.random()
        msg = "boom"
        raise RuntimeError(msg)


def test_seeded_restores_the_generators_when_the_body_raises():
    np.random.seed(11)
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="boom"):
        _raise_while_seeded()
    after = np.random.random()
    np.random.seed(11)
    assert (after, torch.get_num_threads()) == (np.random.random(), threads)
