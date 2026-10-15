"""CTGAN and TVAE from SDV at SDV's defaults, filled stratum by stratum by rejection.

SDV's neural synthesizers cannot condition on a cell: SDV falls back to rejection sampling, and an
age band is no column of the data. So each draws unconditioned batches and keeps the rows whose own
values fall in a stratum still short, until every quota is met or `MAX_DRAWN_ROWS` rows are drawn.
SDV seeds no fit, so the run's seed reaches the fit through numpy's and torch's global generators,
restored afterwards; SDV seeds its own sampling after every fit. Training runs on the CPU with
`TORCH_THREADS` threads, so a seed gives the same rows on one machine and lock. ADR
`research/baseline-augmentation-design` records the choices.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

from bko.baselines.errors import BaselineError
from bko.baselines.records import FRAME_COLUMNS, Baseline, number, unit_weights, whole
from bko.baselines.request import Request, Stratum, stratum_keys
from bko.datasets.uci import FRACTIONAL, NUMERIC


logger = logging.getLogger(__name__)

CTGAN_PARAMETERS: Final[Mapping[str, object]] = MappingProxyType({})
"""Departures from SDV's CTGAN defaults: none (ADR `research/baseline-augmentation-design`)."""

TVAE_PARAMETERS: Final[Mapping[str, object]] = MappingProxyType({})
"""Departures from SDV's TVAE defaults: none."""

TORCH_THREADS: Final = 1
"""torch's intra-op threads while fitting and sampling: fixes the reduction order."""

DRAW_BATCH: Final = 10_000
"""Rows drawn per unconditioned batch."""

MAX_DRAWN_ROWS: Final = 1_000_000
"""Rows drawn before a quota still short is refused."""


def _sdtype(name: str) -> dict[str, str]:
    if name not in NUMERIC:
        return {"sdtype": "categorical"}
    representation = "Float" if name in FRACTIONAL else "Int64"
    return {"sdtype": "numerical", "computer_representation": representation}


METADATA: Final[Mapping[str, object]] = MappingProxyType(
    {"tables": {"uci": {"columns": {name: _sdtype(name) for name in FRAME_COLUMNS}}}}
)
"""The evaluation frame for SDV, in its column order: measured predictors numerical, coded ones and
the label categorical."""

type Draw = Callable[[int], pd.DataFrame]
"""Returns that many unconditioned rows from a fitted synthesizer."""

type Factory = type[CTGANSynthesizer] | type[TVAESynthesizer]
"""One of SDV's two neural single-table synthesizers."""


def ctgan(
    train: pd.DataFrame, request: Request, *, parameters: Mapping[str, object] = CTGAN_PARAMETERS
) -> Baseline:
    """Fit SDV's CTGAN on `train` and fill `request` by rejection."""
    return _synthesize(CTGANSynthesizer, train, request, parameters, method="ctgan")


def tvae(
    train: pd.DataFrame, request: Request, *, parameters: Mapping[str, object] = TVAE_PARAMETERS
) -> Baseline:
    """Fit SDV's TVAE on `train` and fill `request` by rejection."""
    return _synthesize(TVAESynthesizer, train, request, parameters, method="tvae")


def build_synthesizer(
    factory: Factory, parameters: Mapping[str, object]
) -> CTGANSynthesizer | TVAESynthesizer:
    """An unfitted synthesizer on `METADATA`, on the CPU, with `parameters` over the defaults."""
    return factory(Metadata.load_from_dict(dict(METADATA)), enable_gpu=False, **parameters)


@contextmanager
def seeded(seed: int) -> Generator[None]:
    """Seed numpy's and torch's global generators and pin torch's threads; restore all three."""
    threads = torch.get_num_threads()
    state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(TORCH_THREADS)
        try:
            yield
        finally:
            torch.set_num_threads(threads)
            np.random.set_state(state)


def fill(
    draw: Draw,
    request: Request,
    *,
    method: str,
    batch: int = DRAW_BATCH,
    limit: int = MAX_DRAWN_ROWS,
) -> pd.DataFrame:
    """Draw batches until each stratum holds its quota.

    Each stratum keeps its first matching rows in draw order; the strata follow the quota's order.

    Raises:
        BaselineError: If the request exceeds `limit`, or `limit` rows are asked for before
            every quota is met.
    """
    if request.rows > limit:
        msg = f"{method} cannot add {request.rows} rows; it draws at most {limit}"
        raise BaselineError(msg)
    needed = dict(request.quota)
    kept: dict[Stratum, list[pd.DataFrame]] = {stratum: [] for stratum in needed}
    drawn = 0
    while any(needed.values()):
        if drawn >= limit:
            short = ", ".join(
                f"{'/'.join(map(str, stratum))}={n}" for stratum, n in needed.items() if n
            )
            msg = f"{method} asked for {drawn} rows and still lacks {short}"
            raise BaselineError(msg)
        rows = whole(draw(batch))
        drawn += batch
        keys = stratum_keys(rows, request.strata)
        for stratum, need in needed.items():
            if need:
                hits = rows.loc[[key == stratum for key in keys]].head(need)
                kept[stratum].append(hits)
                needed[stratum] = need - len(hits)
    logger.info("filled method=%s drawn=%d added=%d", method, drawn, request.rows)
    return number(pd.concat([frame for frames in kept.values() for frame in frames]))


def _synthesize(
    factory: Factory,
    train: pd.DataFrame,
    request: Request,
    parameters: Mapping[str, object],
    *,
    method: str,
) -> Baseline:
    synthesizer = build_synthesizer(factory, parameters)
    with seeded(request.seed):
        start = time.perf_counter()
        synthesizer.fit(train.loc[:, list(FRAME_COLUMNS)])
        logger.info("fit method=%s seconds=%.1f", method, time.perf_counter() - start)
        added = fill(lambda rows: synthesizer.sample(num_rows=rows), request, method=method)
    params = {**synthesizer.get_parameters(), "torch_threads": TORCH_THREADS}
    return Baseline(added=added, weights=unit_weights(train), params=params)
