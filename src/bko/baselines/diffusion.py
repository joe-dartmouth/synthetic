"""TabDDPM (Kotelnikov et al. 2023): fit on the training split, filled by rejection per stratum.

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
from bko.baselines.synthesizers import TORCH_THREADS, fill, holding, seeded
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
        BaselineError: If a training loss is not finite, naming its step, or, as `fill`, if the
            draws cannot meet the quota.
        ShortfallError: As `fill`, holding the rows kept as the method's baseline, which a harness
            may score as a partial fill.
    """
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

        def draw(rows: int) -> pd.DataFrame:
            y = torch.multinomial(counts, rows, replacement=True, generator=generator)
            numeric, coded = sample(
                model,
                schedule,
                y,
                numeric=encoded.numeric.shape[1],
                sizes=encoded.sizes,
                generator=generator,
            )
            return codec.decode(numeric, coded, y)

        try:
            added = fill(draw, request, method="tabddpm")
        except ShortfallError as exc:
            raise holding(exc, train, params) from None
    return Baseline(added=added, weights=unit_weights(train), params=params)


def encode(train: pd.DataFrame, request: Request) -> tuple[Codec, Encoded]:
    """Fit the encoding on `train` and encode it.

    The quantile transform takes the authors' quantile count, `n // 30` held between 10 and 1000,
    and never more than the split's rows.
    """
    frame = request.frame
    rows = len(train)
    transform = None
    if frame.numeric:
        quantiles = min(max(min(rows // ROWS_PER_QUANTILE, MAX_QUANTILES), MIN_QUANTILES), rows)
        transform = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=quantiles,
            subsample=QUANTILE_SUBSAMPLE,
            random_state=request.seed,
        ).fit(train.loc[:, list(frame.numeric)].to_numpy("float64"))
    states = []
    for field in frame.categorical:
        present = set(train[field].astype(int))
        states.append(tuple(code for code in frame.codebook[field] if code in present))
    codec = Codec(frame, transform, tuple(states))
    return codec, codec.encode(train)


def _batches(rows: int, size: int, steps: int, generator: torch.Generator) -> Iterator[Tensor]:
    """`steps` batches of row indices: each epoch one permutation cut into slices of `size`.

    Raises:
        ValueError: If `rows` is below 1, which no epoch could cut.
    """
    if rows < 1:
        msg = f"cannot batch {rows} rows"
        raise ValueError(msg)
    used = 0
    while True:
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
    authors set it, so the first two steps both run at `lr`.

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
    return model


def validation_loss(
    model: Denoiser, schedule: Schedule, encoded: Encoded, *, draws: int, generator: torch.Generator
) -> float:
    """The loss of `model` on `encoded`, averaged over `draws` draws of every row's step and noise.

    The model runs in evaluation mode without gradients and is put back in its own mode after.
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
