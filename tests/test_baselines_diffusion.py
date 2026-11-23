"""TabDDPM's runner: seeding, the fill contract, its fit line, shortfalls and pinned settings."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch

from bko.baselines import diffusion
from bko.baselines.ddpm import Denoiser, cosine_schedule, sample
from bko.baselines.diffusion import (
    TABDDPM_PARAMETERS,
    Codec,
    Encoded,
    Parameters,
    _batches,
    encode,
    fit_denoiser,
    tabddpm,
    validation_loss,
)
from bko.baselines.errors import BaselineError, ShortfallError
from bko.baselines.fills import CURRENT_FILL, Fill
from bko.baselines.records import Baseline, unit_weights
from bko.baselines.request import Request, Stratum
from bko.baselines.synthesizers import MAX_DRAWN_ROWS, fill
from bko.datasets.uci import UCI_FRAME
from tests._baselines_fixtures import EDGE, QUOTA, TINY_TABDDPM, assert_contract, full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TRAIN = full_records()


def run(seed: int = 3, quota: dict[Stratum, int] | None = None) -> Baseline:
    request = Request(quota=QUOTA if quota is None else quota, seed=seed, frame=UCI_FRAME)
    return tabddpm(TRAIN, request, parameters=TINY_TABDDPM)


@pytest.fixture(scope="module")
def runs():
    """Three runs: (ambient threads 1, seed 3), (ambient threads 2, seed 3), (ambient 1, seed 4)."""
    threads = torch.get_num_threads()

    def at(ambient, seed):
        torch.set_num_threads(ambient)
        np.random.seed(ambient)
        torch.manual_seed(ambient)
        try:
            return run(seed)
        finally:
            torch.set_num_threads(threads)

    return at(1, 3), at(2, 3), at(1, 4)


def test_the_seed_alone_decides_the_rows(runs):
    assert runs[0].added.to_csv() == runs[1].added.to_csv()


def test_a_different_seed_gives_different_rows(runs):
    """The positive control: a runner that seeds with a constant passes the test above."""
    assert runs[0].added.to_csv() != runs[2].added.to_csv()


def test_a_tiny_fit_fills_a_two_stratum_request():
    quota: dict[Stratum, int] = {(f"<{EDGE}", "female", 1): 3, (f">={EDGE}", "male", 0): 2}
    result = run(quota=quota)
    assert_contract(result.added, TRAIN, quota)
    pd.testing.assert_series_equal(result.weights, unit_weights(TRAIN))
    assert result.params == {
        "steps": 2,
        "timesteps": 4,
        "layers": (16,),
        "dim_t": 8,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch": 32,
        "torch_threads": 1,
    }


def test_training_runs_at_one_thread_and_restores_state(monkeypatch):
    seen: list[int] = []
    real = diffusion.loss

    def counting(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return real(*args, **kwargs)

    monkeypatch.setattr(diffusion, "loss", counting)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        np.random.seed(11)
        torch.manual_seed(11)
        run()
        after = (np.random.random(), torch.rand(1).item(), torch.get_num_threads())
    finally:
        torch.set_num_threads(threads)
    np.random.seed(11)
    torch.manual_seed(11)
    assert after == (np.random.random(), torch.rand(1).item(), 2)
    assert seen == [1] * TINY_TABDDPM.steps


def test_the_fit_line_names_the_harness_fill(caplog):
    caplog.set_level(logging.INFO, logger=diffusion.logger.name)
    token = CURRENT_FILL.set(Fill(3, "replica"))
    try:
        run()
    finally:
        CURRENT_FILL.reset(token)
    fits = [m for m in caplog.messages if m.startswith("fit ")]
    assert len(fits) == 1
    assert re.fullmatch(r"fit holdout=3 fill=replica method=tabddpm seconds=\d+\.\d", fits[0])


def test_a_shortfall_hands_back_its_rows_as_a_baseline(monkeypatch):
    monkeypatch.setattr(diffusion, "fill", partial(fill, batch=10, limit=10))
    with pytest.raises(ShortfallError) as caught:
        run()
    error = caught.value
    assert error.baseline is not None
    assert error.baseline.added.equals(error.added)
    assert error.baseline.params["short"] == dict(error.short)
    assert error.baseline.params["torch_threads"] == 1


def test_a_non_finite_loss_names_its_step(monkeypatch):
    monkeypatch.setattr(diffusion, "loss", lambda *args, **kwargs: torch.tensor(float("nan")))
    with pytest.raises(BaselineError, match="tabddpm's loss is nan at step 1 of 2"):
        run()


def test_the_model_stays_on_the_cpu(monkeypatch):
    devices: list[torch.device] = []
    real = diffusion.sample

    def recording(model, schedule, y, **kwargs):
        devices.extend({p.device for p in model.parameters()} | {y.device})
        return real(model, schedule, y, **kwargs)

    monkeypatch.setattr(diffusion, "sample", recording)
    with torch.device("meta"):
        run()
    assert devices
    assert set(devices) == {torch.device("cpu")}


def test_the_adr_constants_are_pinned():
    assert (
        Parameters(
            steps=20000,
            timesteps=1000,
            layers=(256, 256),
            dim_t=128,
            dropout=0.0,
            lr=0.001,
            weight_decay=1e-05,
            batch=4096,
        )
        == TABDDPM_PARAMETERS
    )


def request(seed: int = 3, quota: dict[Stratum, int] | None = None) -> Request:
    return Request(quota=QUOTA if quota is None else quota, seed=seed, frame=UCI_FRAME)


def test_the_learning_rate_anneals_after_each_update(monkeypatch):
    """Step k updates at the rate set after step k - 1: steps 1 and 2 both run at `lr`."""
    used: list[float] = []

    class Spy(torch.optim.AdamW):
        def step(self, closure=None):
            used.append(self.param_groups[0]["lr"])
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "AdamW", Spy)
    parameters = replace(TINY_TABDDPM, steps=4, lr=0.1)
    _, encoded = encode(TRAIN, request())
    fit_denoiser(
        encoded,
        cosine_schedule(parameters.timesteps),
        parameters,
        torch.Generator().manual_seed(0),
        labels=2,
    )
    assert used == pytest.approx([0.1, 0.1, 0.1 * 3 / 4, 0.1 * 2 / 4])


def test_batches_cut_each_epoch_from_a_fresh_permutation():
    generator = torch.Generator().manual_seed(0)
    batches = list(_batches(10, 4, 7, generator))
    assert [len(b) for b in batches] == [4, 4, 2, 4, 4, 2, 4]
    first = torch.cat(batches[:3])
    second = torch.cat(batches[3:6])
    assert first.sort().values.tolist() == list(range(10))
    assert second.sort().values.tolist() == list(range(10))
    assert first.tolist() != second.tolist()


def test_batches_stop_mid_epoch_at_the_step_count():
    generator = torch.Generator().manual_seed(0)
    assert len(list(_batches(10, 4, 2, generator))) == 2
    assert list(_batches(10, 4, 0, generator)) == []


def test_batches_refuse_no_rows():
    with pytest.raises(ValueError, match="cannot batch 0 rows"):
        next(_batches(0, 4, 3, torch.Generator()))


@pytest.mark.parametrize(("rows", "quantiles"), [(5, 5), (100, 10), (6000, 200), (31000, 1000)])
def test_the_quantile_count_is_n_over_30_clamped_and_capped_at_n(rows, quantiles):
    train = TRAIN.sample(rows, replace=True, random_state=0)
    codec, _ = encode(train, request())
    assert codec.transform is not None
    assert codec.transform.n_quantiles == quantiles


def test_states_follow_the_codebook_and_omit_absent_codes():
    """Code 6 of `thal` is absent and 7 appears first: the states are (3, 7)."""
    rest = TRAIN[TRAIN["thal"] != 6]
    train = pd.concat([rest[rest["thal"] == 7], rest[rest["thal"] != 7]])
    codec, encoded = encode(train, request())
    i = UCI_FRAME.categorical.index("thal")
    assert codec.states[i] == (3, 7)
    assert encoded.sizes[i] == 2
    assert encoded.coded[:, i].tolist() == [1 if code == 7 else 0 for code in train["thal"]]
    for j, field in enumerate(UCI_FRAME.categorical):
        present = set(train[field])
        assert codec.states[j] == tuple(c for c in UCI_FRAME.codebook[field] if c in present)


def test_decode_inverts_encode():
    codec, encoded = encode(TRAIN, request())
    decoded = codec.decode(encoded.numeric, encoded.coded, encoded.y)
    kept = [*UCI_FRAME.categorical, UCI_FRAME.label]
    pd.testing.assert_frame_equal(
        decoded[kept].astype("int64"),
        TRAIN[kept].astype("int64").reset_index(drop=True),
    )
    measured = list(UCI_FRAME.numeric)
    np.testing.assert_allclose(
        decoded[measured].to_numpy(), TRAIN[measured].to_numpy("float64"), atol=1e-3
    )
    assert isinstance(codec, Codec)


def test_draw_time_labels_follow_the_training_label_counts(monkeypatch):
    """One label-1 row in eight: the drawn share is near 1/8, and `sample` and `decode` share it."""
    label = UCI_FRAME.label
    train = pd.concat([TRAIN[TRAIN[label] == 0], TRAIN[TRAIN[label] == 1].head(8)])
    share = 8 / len(train)
    given: list[torch.Tensor] = []
    decoded: list[torch.Tensor] = []
    real_sample, real_decode = diffusion.sample, Codec.decode

    def sampling(model, schedule, y, **kwargs):
        given.append(y)
        return real_sample(model, schedule, y, **kwargs)

    def decoding(self, numeric, coded, y):
        decoded.append(y)
        return real_decode(self, numeric, coded, y)

    monkeypatch.setattr(diffusion, "sample", sampling)
    monkeypatch.setattr(Codec, "decode", decoding)
    monkeypatch.setattr(diffusion, "fill", partial(fill, batch=1000, limit=1000))
    quota: dict[Stratum, int] = {(f"<{EDGE}", "female", 1): 1}
    tabddpm(train, request(quota=quota), parameters=TINY_TABDDPM)
    assert len(given) == 1
    assert len(given[0]) == 1000
    assert torch.equal(given[0], decoded[0])
    assert abs(given[0].double().mean().item() - share) < 0.05


def test_validation_loss_averages_its_draws_without_gradients(monkeypatch):
    values = iter([1.0, 2.0, 6.0])
    seen: list[tuple[bool, bool]] = []
    model = Denoiser(4, 2, layers=(4,), dim_t=8, dropout=0.0)

    def fake(model, *args, **kwargs):
        seen.append((torch.is_grad_enabled(), model.training))
        return torch.tensor(next(values))

    monkeypatch.setattr(diffusion, "loss", fake)
    _, encoded = encode(TRAIN, request())
    schedule = cosine_schedule(4)
    for before in (True, False):
        values = iter([1.0, 2.0, 6.0])
        seen.clear()
        model.train(before)
        result = validation_loss(
            model, schedule, encoded, draws=3, generator=torch.Generator().manual_seed(0)
        )
        assert result == pytest.approx(3.0)
        assert seen == [(False, False)] * 3
        assert model.training is before


def test_the_seed_drives_the_draws_with_the_model_init_held_fixed(monkeypatch):
    real = diffusion.Denoiser

    def fixed(*args, **kwargs):
        torch.manual_seed(0)
        return real(*args, **kwargs)

    labels: list[torch.Tensor] = []
    real_sample = diffusion.sample

    def sampling(model, schedule, y, **kwargs):
        labels.append(y)
        return real_sample(model, schedule, y, **kwargs)

    monkeypatch.setattr(diffusion, "Denoiser", fixed)
    monkeypatch.setattr(diffusion, "sample", sampling)
    first = tabddpm(TRAIN, request(seed=3), parameters=TINY_TABDDPM)
    split = len(labels)
    second = tabddpm(TRAIN, request(seed=4), parameters=TINY_TABDDPM)
    assert labels[0].shape == labels[split].shape
    assert not torch.equal(labels[0], labels[split])
    assert first.added.to_csv() != second.added.to_csv()


def test_a_fit_learns_a_dominant_state():
    """One state in 90% of rows: a trained draw's share of it sits nearer 0.9 than the untrained."""
    rows = 100
    index = torch.arange(rows)[:, None]
    coded = torch.where(index % 10 == 0, (index // 10) % 2 + 1, 0)
    numeric = torch.randn(rows, 1, generator=torch.Generator().manual_seed(0))
    encoded = Encoded(numeric, coded, torch.zeros(rows, dtype=torch.long), (3,))
    parameters = replace(TINY_TABDDPM, steps=200, lr=1e-2)
    schedule = cosine_schedule(parameters.timesteps)
    torch.manual_seed(0)
    untrained = Denoiser(
        4, 1, layers=parameters.layers, dim_t=parameters.dim_t, dropout=parameters.dropout
    )
    torch.manual_seed(0)
    trained = fit_denoiser(
        encoded, schedule, parameters, torch.Generator().manual_seed(1), labels=1
    )

    def share(model: Denoiser) -> float:
        y = torch.zeros(200, dtype=torch.long)
        generator = torch.Generator().manual_seed(2)
        _, drawn = sample(model, schedule, y, numeric=1, sizes=(3,), generator=generator)
        return float((drawn[:, 0] == 0).double().mean())

    before, after = share(untrained), share(trained)
    assert abs(after - 0.9) < abs(after - before)


def test_the_fit_builds_its_denoiser_and_optimizer_from_the_parameters(monkeypatch):
    decays: list[float] = []

    class Spy(torch.optim.AdamW):
        def __init__(self, params, **kwargs):
            decays.append(kwargs["weight_decay"])
            super().__init__(params, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", Spy)
    parameters = replace(TINY_TABDDPM, steps=1, layers=(5, 7), dim_t=6, dropout=0.25)
    parameters = replace(parameters, weight_decay=0.05)
    _, encoded = encode(TRAIN, request())
    model = fit_denoiser(
        encoded,
        cosine_schedule(parameters.timesteps),
        parameters,
        torch.Generator().manual_seed(0),
        labels=2,
    )
    linears = [m.out_features for m in model.blocks if isinstance(m, torch.nn.Linear)]
    dropouts = [m.p for m in model.blocks if isinstance(m, torch.nn.Dropout)]
    assert linears == [5, 7]
    assert dropouts == [0.25, 0.25]
    assert model.dim_t == 6
    assert model.projection.out_features == 6
    assert decays == [0.05]


def test_encode_refuses_an_empty_split():
    with pytest.raises(BaselineError, match="tabddpm cannot fit an empty training split"):
        encode(TRAIN.head(0), request())


def test_encode_refuses_a_code_outside_the_codebook_naming_its_field():
    train = TRAIN.copy()
    train.loc[train.index[0], "thal"] = 5
    with pytest.raises(BaselineError, match=r"cannot encode thal: codes \[5\] are outside"):
        encode(train, request())


def test_a_request_above_the_draw_limit_is_refused_before_fitting(monkeypatch):
    def unfit(*args, **kwargs):
        raise AssertionError

    monkeypatch.setattr(diffusion, "fit_denoiser", unfit)
    quota: dict[Stratum, int] = {(f"<{EDGE}", "female", 1): MAX_DRAWN_ROWS + 1}
    expected = f"tabddpm cannot add {MAX_DRAWN_ROWS + 1} rows; it draws at most {MAX_DRAWN_ROWS}"
    with pytest.raises(BaselineError, match=re.escape(expected)):
        tabddpm(TRAIN, request(quota=quota), parameters=TINY_TABDDPM)


def test_batches_draw_no_permutation_after_the_last_slice():
    generator = torch.Generator().manual_seed(0)
    assert len(list(_batches(10, 5, 2, generator))) == 2
    reference = torch.Generator().manual_seed(0)
    torch.randperm(10, generator=reference)
    assert torch.equal(generator.get_state(), reference.get_state())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("timesteps", 0),
        ("batch", 0),
        ("dim_t", 1),
        ("lr", -1e-3),
        ("weight_decay", -1e-5),
        ("dropout", -0.1),
        ("dropout", 1.0),
    ],
)
def test_parameters_refuse_settings_no_fit_can_run(field, value):
    with pytest.raises(ValueError, match=f"tabddpm's {field} is {value}"):
        replace(TINY_TABDDPM, **{field: value})


def test_the_fit_logs_its_progress(monkeypatch, caplog):
    monkeypatch.setattr(diffusion, "LOG_EVERY", 2)
    caplog.set_level(logging.INFO, logger=diffusion.logger.name)
    parameters = replace(TINY_TABDDPM, steps=5)
    _, encoded = encode(TRAIN, request())
    fit_denoiser(
        encoded,
        cosine_schedule(parameters.timesteps),
        parameters,
        torch.Generator().manual_seed(0),
        labels=2,
    )
    lines = [m for m in caplog.messages if m.startswith("tabddpm step=")]
    steps = [
        re.fullmatch(r"tabddpm step=(\d+) of=5 loss=\d+\.\d{4} seconds=\d+\.\d", m) for m in lines
    ]
    assert [int(m.group(1)) for m in steps if m] == [2, 4, 5]
    assert len(lines) == 3


def test_a_non_finite_draw_is_refused_naming_it(monkeypatch):
    real = diffusion.sample

    def poisoned(*args, **kwargs):
        numeric, coded = real(*args, **kwargs)
        numeric[0, 0] = float("inf")
        return numeric, coded

    monkeypatch.setattr(diffusion, "sample", poisoned)
    with pytest.raises(BaselineError, match="tabddpm sampled a non-finite value in draw 1"):
        run()
