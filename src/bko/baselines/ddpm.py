"""TabDDPM's diffusion processes and denoiser: Gaussian for measured fields, multinomial for codes.

Parts of this file follow tab-ddpm (https://github.com/yandex-research/tab-ddpm, commit
b476257dd460b778ba09eb97f7a51d6490fa17f8), by Akim Kotelnikov et al., which is licensed under the
MIT License: the time embedding and the denoiser's forward. Its notice:

MIT License

Copyright (c) 2022 Akim Kotelnikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This file is released under the GNU General Public License, version 3 only (`LICENSE`), as the
repository is.

The equations are Kotelnikov et al. 2023 (arXiv:2209.15421), section 3 and section 4. A measured
field follows Gaussian diffusion (Ho et al. 2020): `q(x_t | x_0) = N(sqrt(abar_t) x_0,
(1 - abar_t) I)`, trained by the noise-prediction mean squared error (Equation 2) and sampled
ancestrally through `mu_theta` with the variance `beta_t` (Ho et al.'s fixed large choice, which the
authors sample with). Each coded field is its own multinomial diffusion (Hoogeboom et al. 2021):
`q(x_t | x_0) = Cat(abar_t x_0 + (1 - abar_t) / K)`, with the posterior `q(x_{t-1} | x_t, x_0)`
proportional to `[alpha_t x_t + (1 - alpha_t) / K] * [abar_{t-1} x_0 + (1 - abar_{t-1}) / K]`,
computed in log space by `torch.logaddexp` as Hoogeboom et al. do. The reverse step is that
posterior at the denoiser's predicted `x_0`. A coded field's term is Hoogeboom et al.'s unbiased
estimate of the variational bound under a uniformly drawn step, the authors' objective: `T L_t` plus
the prior's KL divergence `KL(q(x_T | x_0) || Cat(1 / K))`, where `L_t` is the KL divergence from
the true posterior to the model's, or the decoder's negative log-likelihood at the first step. The
coded fields' sum is divided by their number (Equation 3). The denoiser is Equations 4 and 5: an MLP
over the noised record plus a sinusoidal time embedding and a label embedding. The schedule is the
cosine schedule of Nichol and Dhariwal 2021 with betas clipped at 0.999.

Timesteps are indexed from 0, so `abar_{-1}` is 1 and the step at index 0 decodes `x_0`. Every
random draw takes the caller's `torch.Generator`. A probability is clamped at `TINY` before its
logarithm only where it must be formed (a one-hot code, a schedule term); the predicted states stay
log-probabilities on the loss path. A coded field enters the denoiser one-hot and leaves it as one
logit per state; a drawn code is the index of its state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional
from torch import Tensor, nn


TINY: Final = 1e-30
"""The floor a probability is clamped at before its logarithm."""

COSINE_OFFSET: Final = 0.008
"""The cosine schedule's offset `s` (Nichol and Dhariwal 2021)."""

MAX_BETA: Final = 0.999
"""The cosine schedule's ceiling on each beta."""

MAX_PERIOD: Final = 10_000
"""The longest period of the sinusoidal time embedding."""


@dataclass(frozen=True, slots=True)
class Schedule:
    """A noise schedule's per-timestep terms, each a float32 tensor of length `timesteps`.

    Attributes:
        betas: `beta_t`, the noise each step adds.
        alphas: `alpha_t = 1 - beta_t`.
        alpha_bars: `abar_t`, the product of `alpha` up to and including `t`.
        alpha_bars_prev: `abar_{t-1}`, 1 at the first step.
    """

    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor
    alpha_bars_prev: Tensor

    @property
    def timesteps(self) -> int:
        """The number of diffusion steps."""
        return int(self.betas.shape[0])


def cosine_schedule(timesteps: int) -> Schedule:
    """The cosine schedule over `timesteps` steps, each beta clipped at `MAX_BETA`."""
    grid = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    f = torch.cos((grid + COSINE_OFFSET) / (1 + COSINE_OFFSET) * math.pi / 2) ** 2
    betas = torch.clamp(1 - f[1:] / f[:-1], max=MAX_BETA)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
    return Schedule(
        betas=betas.float(),
        alphas=alphas.float(),
        alpha_bars=alpha_bars.float(),
        alpha_bars_prev=alpha_bars_prev.float(),
    )


def _time_embedding(t: Tensor, dim: int) -> Tensor:
    """Each timestep as `dim` sinusoids: cosines then sines at geometric frequencies."""
    half = dim // 2
    frequencies = torch.exp(-math.log(MAX_PERIOD) * torch.arange(half, dtype=torch.float32) / half)
    angles = t.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class Denoiser(nn.Module):
    """TabDDPM's MLP (Equations 4 and 5): noise for measured fields, logits for coded ones."""

    def __init__(
        self, width: int, labels: int, *, layers: tuple[int, ...], dim_t: int, dropout: float
    ) -> None:
        """Build the layers.

        Args:
            width: The record's width: measured fields plus every coded field's states.
            labels: The number of label values conditioned on.
            layers: Each hidden block's width.
            dim_t: The width of the input projection and of both embeddings.
            dropout: Each hidden block's dropout probability.
        """
        super().__init__()
        self.dim_t: int = dim_t
        self.projection: nn.Linear = nn.Linear(width, dim_t)
        self.time: nn.Sequential = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )
        self.label: nn.Embedding = nn.Embedding(labels, dim_t)
        blocks: list[nn.Module] = []
        previous = dim_t
        for size in layers:
            blocks += [nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)]
            previous = size
        self.blocks: nn.Sequential = nn.Sequential(*blocks)
        self.head: nn.Linear = nn.Linear(previous, width)

    # torch declares `forward` a writable `Callable[..., Any]` attribute; a method overrides it.
    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: Tensor, t: Tensor, y: Tensor
    ) -> Tensor:
        """Predict the noise and the coded fields' logits for noised records `x` at steps `t`."""
        hidden = (
            self.projection(x)
            + self.time(_time_embedding(t, self.dim_t))
            + torch.nn.functional.silu(self.label(y))
        )
        return self.head(self.blocks(hidden))


def _one_hot(coded: Tensor, sizes: Sequence[int]) -> Tensor:
    """State indices `coded` (rows by fields) as concatenated one-hot blocks of `sizes`."""
    blocks = [torch.nn.functional.one_hot(coded[:, i], k).float() for i, k in enumerate(sizes)]
    return torch.cat(blocks, dim=1) if blocks else torch.zeros(coded.shape[0], 0)


def _states(sizes: Sequence[int]) -> Tensor:
    """Each one-hot column's block size `K`, as a float row."""
    counts = torch.tensor(list(sizes), dtype=torch.long)
    return torch.repeat_interleave(counts, counts).float()[None, :]


def _log(probabilities: Tensor) -> Tensor:
    """The logarithm of `probabilities`, each clamped at `TINY` first."""
    return torch.log(probabilities.clamp(min=TINY))


def _block_log_softmax(logits: Tensor, sizes: Sequence[int]) -> Tensor:
    """Normalize each block of `logits` to log-probabilities."""
    if not sizes:
        return logits
    blocks = torch.split(logits, list(sizes), dim=1)
    return torch.cat([torch.log_softmax(block, dim=1) for block in blocks], dim=1)


def _gumbel_max(
    log_probabilities: Tensor, sizes: Sequence[int], generator: torch.Generator
) -> Tensor:
    """One state index per block, drawn from its log-probabilities by the Gumbel-max trick."""
    rows = log_probabilities.shape[0]
    if not sizes:
        return torch.zeros(rows, 0, dtype=torch.long)
    uniform = torch.rand(log_probabilities.shape, generator=generator)
    gumbel = -_log(-_log(uniform))
    blocks = torch.split(log_probabilities + gumbel, list(sizes), dim=1)
    return torch.stack([block.argmax(dim=1) for block in blocks], dim=1)


def _log_mix(log_x: Tensor, keep: Tensor, k: Tensor) -> Tensor:
    """`log(keep x + (1 - keep) / K)` from `log x`, by `torch.logaddexp` of the two log terms."""
    return torch.logaddexp(log_x + _log(keep), _log(1 - keep) - torch.log(k))


def _log_posterior(
    schedule: Schedule,
    x_t: Tensor,
    log_x0: Tensor,
    t: Tensor,
    sizes: Sequence[int],
    *,
    k: Tensor | None = None,
) -> Tensor:
    """`log q(x_{t-1} | x_t, x_0)` per block, for one-hot `x_t` and log-probabilities `log_x0`.

    At the first step `abar_{-1}` is 1, so `q(x_{t-1} | x_0)` is `x_0` itself and `log_x0` enters
    unmixed. `k` is `_states(sizes)`, computed here when not given.
    """
    if k is None:
        k = _states(sizes)
    from_x_t = _log_mix(_log(x_t), schedule.alphas[t][:, None], k)
    from_x0 = torch.where(
        (t == 0)[:, None], log_x0, _log_mix(log_x0, schedule.alpha_bars_prev[t][:, None], k)
    )
    return _block_log_softmax(from_x_t + from_x0, sizes)


def _multinomial_kl(log_q: Tensor, log_p: Tensor) -> Tensor:
    """Each row's KL divergence from `log_p` to `log_q`, summed over its columns."""
    return (torch.exp(log_q) * (log_q - log_p)).sum(dim=1)


def _multinomial_terms(
    schedule: Schedule, logits: Tensor, x_t: Tensor, x0: Tensor, t: Tensor, sizes: Sequence[int]
) -> Tensor:
    """Each row's multinomial term summed over its blocks: `T L_t` plus the prior's KL divergence.

    Args:
        schedule: The noise schedule.
        logits: The denoiser's logits for the coded fields.
        x_t: The noised coded fields, one-hot.
        x0: The clean coded fields, one-hot.
        t: Each row's timestep.
        sizes: Each coded field's number of states.

    Returns:
        `T` times the KL divergence from the true posterior to the model's (the decoder's negative
        log-likelihood at step 0), plus `KL(q(x_T | x_0) || Cat(1 / K))`, per row.
    """
    k = _states(sizes)
    log_x0 = _log(x0)
    log_model = _log_posterior(schedule, x_t, _block_log_softmax(logits, sizes), t, sizes, k=k)
    kl = _multinomial_kl(_log_posterior(schedule, x_t, log_x0, t, sizes, k=k), log_model)
    decoder = -(x0 * log_model).sum(dim=1)
    log_prior = _log_mix(log_x0, schedule.alpha_bars[-1], k)
    prior = _multinomial_kl(log_prior, -torch.log(k).expand_as(log_prior))
    return schedule.timesteps * torch.where(t == 0, decoder, kl) + prior


def loss(
    model: Denoiser,
    schedule: Schedule,
    numeric: Tensor,
    coded: Tensor,
    y: Tensor,
    sizes: Sequence[int],
    generator: torch.Generator,
) -> Tensor:
    """TabDDPM's loss (Equation 3) on a batch, each row at a uniformly drawn timestep.

    Args:
        model: The denoiser.
        schedule: The noise schedule.
        numeric: The measured fields, rows by fields, already normalized.
        coded: The coded fields as state indices, rows by fields.
        y: The label each row is conditioned on.
        sizes: Each coded field's number of states.
        generator: The source of every draw.

    Returns:
        The Gaussian mean squared error plus the multinomial terms' mean over rows divided by the
        number of coded fields. Each row's multinomial term is `T L_t` plus the prior's KL
        divergence, Hoogeboom et al. 2021's unbiased estimate of the variational bound under a
        uniformly drawn step and the authors' objective.
    """
    t = torch.randint(0, schedule.timesteps, (numeric.shape[0],), generator=generator)
    return _loss_at(model, schedule, numeric, coded, y, sizes, t, generator)


def _loss_at(
    model: Denoiser,
    schedule: Schedule,
    numeric: Tensor,
    coded: Tensor,
    y: Tensor,
    sizes: Sequence[int],
    t: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """`loss` with each row's timestep given."""
    alpha_bar = schedule.alpha_bars[t][:, None]
    noise = torch.randn(numeric.shape, generator=generator)
    noised = torch.sqrt(alpha_bar) * numeric + torch.sqrt(1 - alpha_bar) * noise
    x0 = _one_hot(coded, sizes)
    k = _states(sizes)
    x_t = _one_hot(_gumbel_max(_log_mix(_log(x0), alpha_bar, k), sizes, generator), sizes)
    out = model(torch.cat([noised, x_t], dim=1), t, y)
    width = numeric.shape[1]
    terms: list[Tensor] = []
    if width:
        terms.append(torch.nn.functional.mse_loss(out[:, :width], noise))
    if sizes:
        multinomial = _multinomial_terms(schedule, out[:, width:], x_t, x0, t, sizes)
        terms.append(multinomial.mean() / len(sizes))
    return torch.stack(terms).sum()


def sample(
    model: Denoiser,
    schedule: Schedule,
    y: Tensor,
    *,
    numeric: int,
    sizes: Sequence[int],
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Draw one record per label in `y` by ancestral sampling through every timestep.

    Args:
        model: The trained denoiser, put in evaluation mode while sampling.
        schedule: The noise schedule it was trained on.
        y: The label each record is conditioned on.
        numeric: The number of measured fields.
        sizes: Each coded field's number of states.
        generator: The source of every draw.

    Returns:
        The measured fields, rows by fields, and the coded fields as state indices.
    """
    rows = y.shape[0]
    k = _states(sizes)
    training = model.training
    try:
        model.eval()
        x = torch.randn((rows, numeric), generator=generator)
        codes = _gumbel_max((-torch.log(k)).expand(rows, -1), sizes, generator)
        with torch.no_grad():
            for step in reversed(range(schedule.timesteps)):
                t = torch.full((rows,), step, dtype=torch.long)
                x_t = _one_hot(codes, sizes)
                out = model(torch.cat([x, x_t], dim=1), t, y)
                x = _gaussian_step(schedule, x, out[:, :numeric], step, generator)
                log_predicted = _block_log_softmax(out[:, numeric:], sizes)
                log_post = _log_posterior(schedule, x_t, log_predicted, t, sizes, k=k)
                codes = _gumbel_max(log_post, sizes, generator)
    finally:
        model.train(training)
    return x, codes


def _gaussian_step(
    schedule: Schedule, x: Tensor, noise: Tensor, step: int, generator: torch.Generator
) -> Tensor:
    """`x_{t-1}` from `x_t` and the predicted noise.

    The added noise has variance `beta_t` (Ho et al. 2020's fixed large choice, the authors'), and
    none is added at the first step.
    """
    beta = schedule.betas[step]
    alpha_bar = schedule.alpha_bars[step]
    mean = (x - beta / torch.sqrt(1 - alpha_bar) * noise) / torch.sqrt(schedule.alphas[step])
    if step == 0:
        return mean
    return mean + torch.sqrt(beta) * torch.randn(x.shape, generator=generator)
