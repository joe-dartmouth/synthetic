"""TabDDPM (Kotelnikov et al. 2023): fit on the training split, filled by rejection per stratum.

Parts of this file follow tab-ddpm (https://github.com/yandex-research/tab-ddpm, commit
b476257dd460b778ba09eb97f7a51d6490fa17f8), by Akim Kotelnikov et al., which is licensed under the
MIT License: the quantile transform, the loader and the learning-rate anneal. Its notice:

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

The diffusion processes and the denoiser are `bko.baselines.ddpm`'s; this module encodes the
records, fits the denoiser and draws whole records from it. Settings follow the authors' base
configuration. A measured field passes through the authors' quantile transform to a standard
normal; each coded field is one multinomial block whose states are the codes the training split
holds, in codebook order. The label is conditioned on and never diffused: each drawn record's label
comes from the training label counts first.

The fit follows the authors' trainer: `AdamW`, the learning rate annealed linearly to 0 over the
steps, each epoch one permutation of the training rows cut into batches (the last one shorter), and
no exponential moving average, since the authors sample from the plain model. Every draw after the
model is built (the batches, each loss's steps and noise, the labels and the samples) reads one
`torch.Generator` seeded from the request's seed; the model's initial weights read torch's global
generator under `seeded`, which also pins torch's threads and restores both. So a seed gives the
same rows on one machine and lock. Training and sampling run on the CPU.

Each batch of draws keeps the rows that fall in a stratum still short
(`bko.baselines.synthesizers.fill`), as CTGAN and TVAE do. The `fit` line covers the encoding and
the fit, naming the harness fill (`bko.baselines.fills`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor

from bko.baselines.ddpm import Denoiser, Schedule, cosine_schedule, loss, sample
from bko.baselines.errors import BaselineError, ShortfallError
from bko.baselines.fills import fill_names
from bko.baselines.records import Baseline, derived_seed, unit_weights
from bko.baselines.request import Request
from bko.baselines.synthesizers import MAX_DRAWN_ROWS, TORCH_THREADS, fill, holding, seeded
from bko.datasets.frames import FrameSpec


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Parameters:
    """TabDDPM's settings.

    Attributes:
        steps: Optimizer steps, one batch each.
        timesteps: Diffusion timesteps.
        layers: Each hidden block's width.
        dim_t: The width of the denoiser's input projection and embeddings.
        dropout: Each hidden block's dropout probability.
        lr: `AdamW`'s initial learning rate, annealed linearly to 0 over `steps`.
        weight_decay: `AdamW`'s weight decay.
        batch: Training rows per batch.
    """

    steps: int
    timesteps: int
    layers: tuple[int, ...]
    dim_t: int
    dropout: float
    lr: float
    weight_decay: float
    batch: int

    def __post_init__(self) -> None:
        """Refuse settings no fit can run.

        Raises:
            ValueError: If `steps`, `timesteps` or `batch` is below 1, `dim_t` is below 2, `lr`,
                `weight_decay` or `dropout` is negative, or `dropout` is 1 or more.
        """
        for name, value, least in (
            ("steps", self.steps, 1),
            ("timesteps", self.timesteps, 1),
            ("batch", self.batch, 1),
            ("dim_t", self.dim_t, 2),
            ("lr", self.lr, 0),
            ("weight_decay", self.weight_decay, 0),
            ("dropout", self.dropout, 0),
        ):
            if value < least:
                msg = f"tabddpm's {name} is {value}; it must be at least {least}"
                raise ValueError(msg)
        if self.dropout >= 1:
            msg = f"tabddpm's dropout is {self.dropout}; it must be below 1"
            raise ValueError(msg)


TABDDPM_PARAMETERS: Final = Parameters(
    steps=20000,
    timesteps=1000,
    layers=(256, 256),
    dim_t=128,
    dropout=0.0,
    lr=0.001,
    weight_decay=1e-05,
    batch=4096,
)
"""The authors' base configuration, with `steps` fixed by validation loss (ADR
`research/diffusion-baseline`)."""

MAX_QUANTILES: Final = 1000
"""The quantile transform's most quantiles."""

MIN_QUANTILES: Final = 10
"""The quantile transform's fewest quantiles, unless the split holds fewer rows."""

ROWS_PER_QUANTILE: Final = 30
"""Training rows per quantile below `MAX_QUANTILES`."""

QUANTILE_SUBSAMPLE: Final = 1_000_000_000
"""The quantile transform's subsample: above any split, so every row is used."""

LOG_EVERY: Final = 2000
"""Optimizer steps between the fit's progress lines."""


@dataclass(frozen=True, slots=True)
class Encoded:
    """Records as the denoiser reads them.

    Attributes:
        numeric: The measured fields after the quantile transform, rows by fields.
        coded: Each coded field's state index, rows by fields.
        y: Each row's label index into the frame's label codes.
        sizes: Each coded field's number of states.
    """

    numeric: Tensor
    coded: Tensor
    y: Tensor
    sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Codec:
    """The fitted encoding of one training split: its quantile transform and each field's states."""

    frame: FrameSpec
    transform: QuantileTransformer | None
    states: tuple[tuple[int, ...], ...]

    def encode(self, rows: pd.DataFrame) -> Encoded:
        """`rows` of the frame as the denoiser reads them."""
        frame = self.frame
        numeric = (
            torch.from_numpy(
                self.transform.transform(rows.loc[:, list(frame.numeric)].to_numpy("float64"))
            ).float()
            if self.transform is not None
            else torch.zeros(len(rows), 0)
        )
        columns = [
            [codes.index(int(code)) for code in rows[field]]
            for field, codes in zip(frame.categorical, self.states, strict=True)
        ]
        coded = torch.tensor(columns, dtype=torch.long).reshape(len(columns), len(rows)).T
        y = torch.tensor([frame.label_codes.index(int(code)) for code in rows[frame.label]])
        return Encoded(numeric, coded.contiguous(), y, tuple(map(len, self.states)))

    def decode(self, numeric: Tensor, coded: Tensor, y: Tensor) -> pd.DataFrame:
        """Drawn records as the frame's columns, measured values back on their own scale."""
        frame = self.frame
        out: dict[str, np.ndarray] = {}
        if self.transform is not None:
            values = self.transform.inverse_transform(numeric.double().numpy())
            out |= {field: values[:, i] for i, field in enumerate(frame.numeric)}
        for i, (field, codes) in enumerate(zip(frame.categorical, self.states, strict=True)):
            out[field] = np.asarray(codes)[coded[:, i].numpy()]
        out[frame.label] = np.asarray(frame.label_codes)[y.numpy()]
        return pd.DataFrame(out).loc[:, list(frame.columns)]


def tabddpm(
    train: pd.DataFrame, request: Request, *, parameters: Parameters = TABDDPM_PARAMETERS
) -> Baseline:
    """Fit TabDDPM on `train` and fill `request` by rejection.

    Raises:
        BaselineError: If `train` is empty, if a coded value lies outside the frame's codebook,
            naming its field, if the request asks for more than `MAX_DRAWN_ROWS` rows, before
            fitting, if a training loss is not finite, naming its step, if a draw holds a
            non-finite value, naming the draw, or, as `fill`, if the draws cannot meet the quota.
        ShortfallError: As `fill`, holding the rows kept as the method's baseline, which a harness
            may score as a partial fill.
    """
    if request.rows > MAX_DRAWN_ROWS:
        msg = f"tabddpm cannot add {request.rows} rows; it draws at most {MAX_DRAWN_ROWS}"
        raise BaselineError(msg)
    params = {**asdict(parameters), "torch_threads": TORCH_THREADS}
    generator = torch.Generator().manual_seed(derived_seed(request.seed, "tabddpm"))
    with seeded(request.seed), torch.device("cpu"):
        start = time.perf_counter()
        codec, encoded = encode(train, request)
        schedule = cosine_schedule(parameters.timesteps)
        labels = len(request.frame.label_codes)
        model = fit_denoiser(encoded, schedule, parameters, generator, labels=labels)
        logger.info(
            "fit holdout=%d fill=%s method=%s seconds=%.1f",
            *fill_names(),
            "tabddpm",
            time.perf_counter() - start,
        )
        counts = torch.bincount(encoded.y, minlength=labels).double()
        draws = 0

        def draw(rows: int) -> pd.DataFrame:
            nonlocal draws
            draws += 1
            y = torch.multinomial(counts, rows, replacement=True, generator=generator)
            numeric, coded = sample(
                model,
                schedule,
                y,
                numeric=encoded.numeric.shape[1],
                sizes=encoded.sizes,
                generator=generator,
            )
            if not bool(torch.isfinite(numeric).all()):
                msg = f"tabddpm sampled a non-finite value in draw {draws}"
                raise BaselineError(msg)
            return codec.decode(numeric, coded, y)

        try:
            added = fill(draw, request, method="tabddpm")
        except ShortfallError as exc:
            raise holding(exc, train, params) from None
    return Baseline(added=added, weights=unit_weights(train), params=params)


def encode(train: pd.DataFrame, request: Request) -> tuple[Codec, Encoded]:
    """Fit the encoding on `train` and encode it.

    The quantile transform takes `_quantiles` of the split's rows.

    Raises:
        BaselineError: If `train` is empty, or if a coded value lies outside the frame's codebook,
            naming its field.
    """
    frame = request.frame
    rows = len(train)
    if not rows:
        msg = "tabddpm cannot fit an empty training split"
        raise BaselineError(msg)
    for field in frame.categorical:
        outside = sorted(set(train[field].astype(int)) - set(frame.codebook[field]))
        if outside:
            msg = f"tabddpm cannot encode {field}: codes {outside} are outside its codebook"
            raise BaselineError(msg)
    transform = None
    if frame.numeric:
        transform = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=_quantiles(rows),
            subsample=QUANTILE_SUBSAMPLE,
            random_state=request.seed,
        ).fit(train.loc[:, list(frame.numeric)].to_numpy("float64"))
    states = []
    for field in frame.categorical:
        present = set(train[field].astype(int))
        states.append(tuple(code for code in frame.codebook[field] if code in present))
    codec = Codec(frame, transform, tuple(states))
    return codec, codec.encode(train)


def _quantiles(rows: int) -> int:
    """The authors' quantile count: `rows // 30` held between 10 and 1000, never above `rows`."""
    count = min(max(rows // ROWS_PER_QUANTILE, MIN_QUANTILES), MAX_QUANTILES)
    return min(count, rows)


def _batches(rows: int, size: int, steps: int, generator: torch.Generator) -> Iterator[Tensor]:
    """`steps` batches of row indices: each epoch one permutation cut into slices of `size`.

    Raises:
        ValueError: If `rows` is below 1, which no epoch could cut.
    """
    if rows < 1:
        msg = f"cannot batch {rows} rows"
        raise ValueError(msg)
    used = 0
    while used < steps:
        order = torch.randperm(rows, generator=generator)
        for begin in range(0, rows, size):
            if used == steps:
                return
            yield order[begin : begin + size]
            used += 1


def fit_denoiser(
    encoded: Encoded,
    schedule: Schedule,
    parameters: Parameters,
    generator: torch.Generator,
    *,
    labels: int,
) -> Denoiser:
    """Train a fresh denoiser on `encoded`, conditioned on `labels` label values, as the authors do.

    The learning rate is set to `lr * (1 - step / steps)` after each step's update, where the
    authors set it, so the first two steps both run at `lr`. A progress line is logged every
    `LOG_EVERY` steps and at the last.

    Raises:
        BaselineError: If a step's loss is not finite.
    """
    width = encoded.numeric.shape[1] + sum(encoded.sizes)
    model = Denoiser(
        width,
        labels,
        layers=parameters.layers,
        dim_t=parameters.dim_t,
        dropout=parameters.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=parameters.lr, weight_decay=parameters.weight_decay
    )
    model.train()
    rows = encoded.numeric.shape[0]
    start = time.perf_counter()
    for step, index in enumerate(_batches(rows, parameters.batch, parameters.steps, generator)):
        optimizer.zero_grad()
        value = loss(
            model,
            schedule,
            encoded.numeric[index],
            encoded.coded[index],
            encoded.y[index],
            encoded.sizes,
            generator,
        )
        if not torch.isfinite(value):
            msg = f"tabddpm's loss is {value.item()} at step {step + 1} of {parameters.steps}"
            raise BaselineError(msg)
        value.backward()
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = parameters.lr * (1 - step / parameters.steps)
        done = step + 1
        if done % LOG_EVERY == 0 or done == parameters.steps:
            logger.info(
                "tabddpm step=%d of=%d loss=%.4f seconds=%.1f",
                done,
                parameters.steps,
                value.item(),
                time.perf_counter() - start,
            )
    return model


def validation_loss(
    model: Denoiser, schedule: Schedule, encoded: Encoded, *, draws: int, generator: torch.Generator
) -> float:
    """The loss of `model` on `encoded`, averaged over `draws` draws of every row's step and noise.

    It is the measure that selected `TABDDPM_PARAMETERS.steps` (ADR `research/diffusion-baseline`);
    the fit does not use it. The model runs in evaluation mode without gradients and is put back in
    its own mode after.
    """
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            total = sum(
                float(
                    loss(
                        model,
                        schedule,
                        encoded.numeric,
                        encoded.coded,
                        encoded.y,
                        encoded.sizes,
                        generator,
                    )
                )
                for _ in range(draws)
            )
    finally:
        model.train(training)
    return total / draws
