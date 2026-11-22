"""`bko.baselines.ddpm`: the cosine schedule, both diffusion processes and the denoiser."""

from __future__ import annotations

import math

import torch

from bko.baselines.ddpm import (
    COSINE_OFFSET,
    MAX_BETA,
    Denoiser,
    Schedule,
    _log_posterior,
    _loss_at,
    _multinomial_kl,
    _multinomial_terms,
    _one_hot,
    cosine_schedule,
    loss,
    sample,
)


ROWS = 8
NUMERIC = 2
SIZES = (2, 3)
TIMESTEPS = 10


def batch(sizes: tuple[int, ...] = SIZES, seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    numeric = torch.randn(ROWS, NUMERIC, generator=generator)
    coded = torch.stack([torch.randint(0, k, (ROWS,), generator=generator) for k in sizes], dim=1)
    y = torch.randint(0, 2, (ROWS,), generator=generator)
    return numeric, coded, y


def model(sizes: tuple[int, ...] = SIZES) -> Denoiser:
    torch.manual_seed(0)
    return Denoiser(NUMERIC + sum(sizes), 2, layers=(16, 16), dim_t=8, dropout=0.0)


def test_cosine_betas_lie_in_the_open_unit_interval():
    for timesteps in (1, 4, 100, 1000):
        betas = cosine_schedule(timesteps).betas
        assert betas.dtype == torch.float32
        assert bool((betas > 0).all())
        assert bool((betas <= 0.999).all())


def test_the_multinomial_posterior_sums_to_one_per_block():
    schedule = cosine_schedule(TIMESTEPS)
    generator = torch.Generator().manual_seed(1)
    _, coded, _ = batch()
    x_t = _one_hot(torch.roll(coded, 1, dims=0), SIZES)
    log_x0 = torch.cat(
        [torch.log_softmax(torch.randn(ROWS, k, generator=generator), dim=1) for k in SIZES], dim=1
    )
    t = torch.arange(ROWS) % TIMESTEPS
    log_post = _log_posterior(schedule, x_t, log_x0, t, SIZES)
    for block in torch.split(log_post, list(SIZES), dim=1):
        assert torch.allclose(torch.logsumexp(block, dim=1), torch.zeros(ROWS), atol=1e-5)


def test_the_multinomial_kl_is_zero_for_equal_inputs_and_never_negative():
    generator = torch.Generator().manual_seed(2)
    log_q = torch.log_softmax(torch.randn(ROWS, 5, generator=generator), dim=1)
    log_p = torch.log_softmax(torch.randn(ROWS, 5, generator=generator), dim=1)
    assert torch.equal(_multinomial_kl(log_q, log_q), torch.zeros(ROWS))
    assert bool((_multinomial_kl(log_q, log_p) >= 0).all())


def test_a_one_state_block_is_valid():
    sizes = (1, 3)
    schedule = cosine_schedule(TIMESTEPS)
    numeric, coded, y = batch(sizes)
    denoiser = model(sizes)
    found = loss(denoiser, schedule, numeric, coded, y, sizes, torch.Generator().manual_seed(3))
    assert bool(torch.isfinite(found))
    drawn_numeric, drawn_coded = sample(
        denoiser,
        schedule,
        y,
        numeric=NUMERIC,
        sizes=sizes,
        generator=torch.Generator().manual_seed(4),
    )
    assert drawn_numeric.shape == (ROWS, NUMERIC)
    assert drawn_coded.shape == (ROWS, len(sizes))
    assert bool((drawn_coded[:, 0] == 0).all())
    assert bool(((drawn_coded[:, 1] >= 0) & (drawn_coded[:, 1] < 3)).all())


def test_the_loss_is_finite_at_the_first_and_last_timestep():
    schedule = cosine_schedule(TIMESTEPS)
    numeric, coded, y = batch()
    denoiser = model()
    for step in (0, TIMESTEPS - 1):
        t = torch.full((ROWS,), step, dtype=torch.long)
        found = _loss_at(
            denoiser, schedule, numeric, coded, y, SIZES, t, torch.Generator().manual_seed(5)
        )
        assert found.ndim == 0
        assert bool(torch.isfinite(found))


def test_one_generator_seed_gives_equal_samples():
    schedule = cosine_schedule(TIMESTEPS)
    _, _, y = batch()
    denoiser = model()
    draws = [
        sample(
            denoiser,
            schedule,
            y,
            numeric=NUMERIC,
            sizes=SIZES,
            generator=torch.Generator().manual_seed(6),
        )
        for _ in range(2)
    ]
    (num_a, cat_a), (num_b, cat_b) = draws
    assert num_a.numpy().tobytes() == num_b.numpy().tobytes()
    assert cat_a.numpy().tobytes() == cat_b.numpy().tobytes()


def bayes_posterior(schedule: Schedule, x_t: int, x0: list[float], step: int) -> list[float]:
    """`q(x_{t-1} | x_t, x_0)` by Bayes' rule over every state of `x_{t-1}`, in float64."""
    k = len(x0)
    alpha = float(schedule.alphas[step])
    alpha_bar_prev = float(schedule.alpha_bars_prev[step]) if step else 1.0
    joint = [
        (alpha * (x_t == s) + (1 - alpha) / k) * (alpha_bar_prev * x0[s] + (1 - alpha_bar_prev) / k)
        for s in range(k)
    ]
    return [p / sum(joint) for p in joint]


def test_the_multinomial_posterior_equals_bayes_rule_at_every_step_and_state():
    k = 4
    schedule = cosine_schedule(TIMESTEPS)
    generator = torch.Generator().manual_seed(7)
    log_x0 = torch.log_softmax(3 * torch.randn(3, k, generator=generator), dim=1)
    cases = [(step, x_t, i) for step in range(TIMESTEPS) for x_t in range(k) for i in range(3)]
    t = torch.tensor([step for step, _, _ in cases])
    x_t = _one_hot(torch.tensor([[x] for _, x, _ in cases]), (k,))
    found = torch.exp(_log_posterior(schedule, x_t, log_x0[[i for *_, i in cases]], t, (k,)))
    x0 = torch.exp(log_x0.double()).tolist()
    expected = torch.tensor(
        [bayes_posterior(schedule, x, x0[i], step) for step, x, i in cases], dtype=torch.float64
    )
    assert float((found.double() - expected).abs().max()) < 1e-5


def test_the_posterior_at_the_first_step_returns_a_one_hot_x0():
    schedule = cosine_schedule(TIMESTEPS)
    _, coded, _ = batch()
    x0 = _one_hot(coded, SIZES)
    x_t = _one_hot(torch.roll(coded, 1, dims=0), SIZES)
    t = torch.zeros(ROWS, dtype=torch.long)
    found = torch.exp(_log_posterior(schedule, x_t, torch.log(x0.clamp(min=1e-30)), t, SIZES))
    assert torch.allclose(found, x0, atol=1e-6)


def test_the_cosine_alpha_bars_follow_the_closed_form_where_unclipped():
    for timesteps in (10, 1000):
        schedule = cosine_schedule(timesteps)

        def f(step: int, timesteps: int = timesteps) -> float:
            angle = (step / timesteps + COSINE_OFFSET) / (1 + COSINE_OFFSET) * math.pi / 2
            return math.cos(angle) ** 2

        unclipped = 0
        while unclipped < timesteps and 1 - f(unclipped + 1) / f(unclipped) < MAX_BETA:
            unclipped += 1
        assert unclipped > 0
        expected = torch.tensor([f(step + 1) / f(0) for step in range(unclipped)])
        assert torch.allclose(schedule.alpha_bars[:unclipped], expected, rtol=1e-5, atol=1e-7)


def test_another_generator_seed_gives_other_samples():
    schedule = cosine_schedule(TIMESTEPS)
    _, _, y = batch()
    denoiser = model()
    draws = [
        sample(
            denoiser,
            schedule,
            y,
            numeric=NUMERIC,
            sizes=SIZES,
            generator=torch.Generator().manual_seed(seed),
        )
        for seed in (6, 7)
    ]
    (num_a, cat_a), (num_b, cat_b) = draws
    assert not torch.equal(num_a, num_b)
    assert not torch.equal(cat_a, cat_b)


def test_the_multinomial_term_weights_each_step_by_t_and_decodes_at_the_first():
    k = 4
    schedule = cosine_schedule(TIMESTEPS)
    x0 = _one_hot(torch.tensor([[1]]), (k,))
    x_t = _one_hot(torch.tensor([[2]]), (k,))
    uniform = torch.zeros(1, k)
    states = torch.full((1, k), float(k))
    log_prior = torch.log(schedule.alpha_bars[-1] * x0 + (1 - schedule.alpha_bars[-1]) / states)
    prior = _multinomial_kl(log_prior, torch.log(1 / states))

    last = torch.tensor([TIMESTEPS - 1])
    true = torch.tensor(bayes_posterior(schedule, 2, [0.0, 1.0, 0.0, 0.0], TIMESTEPS - 1))
    model_post = torch.tensor(bayes_posterior(schedule, 2, [1 / k] * k, TIMESTEPS - 1))
    step_kl = float((true * (torch.log(true) - torch.log(model_post))).sum())
    found = _multinomial_terms(schedule, uniform, x_t, x0, last, (k,))
    assert math.isclose(float(found[0]), TIMESTEPS * step_kl + float(prior[0]), rel_tol=1e-4)

    first = torch.tensor([0])
    decoder = -math.log(bayes_posterior(schedule, 2, [1 / k] * k, 0)[1])
    found = _multinomial_terms(schedule, uniform, x_t, x0, first, (k,))
    assert math.isclose(float(found[0]), TIMESTEPS * decoder + float(prior[0]), rel_tol=1e-4)


def test_the_decoder_nll_keeps_a_gradient_for_a_confident_wrong_prediction():
    k = 4
    schedule = cosine_schedule(TIMESTEPS)
    x0 = _one_hot(torch.tensor([[1]]), (k,))
    logits = torch.tensor([[80.0, 0.0, 0.0, 0.0]], requires_grad=True)
    found = _multinomial_terms(schedule, logits, x0, x0, torch.tensor([0]), (k,))
    found.sum().backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 1]) < -0.5
