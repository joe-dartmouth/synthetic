"""Full-feature training frames, Cleveland text built from them, and a hand-written oracle.

The oracle bands ages itself rather than through `cell_keys`, and takes the evaluation frame's
columns and dtypes from the UCI loader, not from `bko.baselines`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd

from bko.baselines.request import Stratum
from bko.datasets.cells import AGE_BAND_EDGES, LABEL
from bko.datasets.uci import (
    CATEGORICAL,
    CODEBOOK,
    FEATURES,
    FRACTIONAL,
    NUMERIC,
    complete_records,
    parse_cleveland,
)
from tests._datasets_fixtures import cleveland_line, records


EDGE = AGE_BAND_EDGES[0]

EVALUATION = complete_records(parse_cleveland(cleveland_line() + "\n"))
"""One real-layout record through the UCI loader: the evaluation frame's columns and dtypes."""

EVALUATION_COLUMNS = list(EVALUATION.columns)
EVALUATION_DTYPES = EVALUATION.dtypes.astype(str).to_dict()

QUOTA: dict[Stratum, int] = {
    (f">={EDGE}", "female", 1): 5,
    (f"<{EDGE}", "female", 1): 3,
    (f">={EDGE}", "male", 0): 2,
}
"""Rows to add in three cells, given out of grid order."""

TINY_CTGAN = MappingProxyType(
    {
        "epochs": 2,
        "batch_size": 20,
        "pac": 10,
        "embedding_dim": 8,
        "generator_dim": (16,),
        "discriminator_dim": (16,),
    }
)
"""A small CTGAN for the gate; the batch is even and a multiple of `pac`, as ctgan asserts."""

TINY_TVAE = MappingProxyType(
    {
        "epochs": 2,
        "batch_size": 20,
        "embedding_dim": 4,
        "compress_dims": (16,),
        "decompress_dims": (16,),
    }
)
"""A small TVAE for the gate."""


def full_records(per_cell: int = 14) -> pd.DataFrame:
    """`per_cell` records in each of the 8 cells, two ages per band, every predictor varied.

    Ages 54 and 55 straddle the band edge, so a row placed in the wrong band is caught. Coded
    predictors cycle through every code inside each cell, so a codebook check has teeth; measured
    ones come from a fixed generator. Fewer than 500 rows keeps CTGAN's transformer in one process.
    """
    ages = (44, EDGE - 1, EDGE, 63)
    spec = {(age, sex, label): per_cell // 2 for age in ages for sex in (0, 1) for label in (0, 1)}
    frame = records(spec)
    position = frame.groupby([frame["age"] >= EDGE, "sex", LABEL]).cumcount()
    for offset, name in enumerate(CATEGORICAL):
        if name != "sex":
            codes = sorted(CODEBOOK[name])
            frame[name] = [codes[(p + offset) % len(codes)] for p in position]
    rng = np.random.default_rng(0)
    n = len(frame)
    frame["trestbps"] = rng.integers(94, 201, n)
    frame["chol"] = rng.integers(126, 565, n)
    frame["thalach"] = rng.integers(71, 203, n)
    frame["oldpeak"] = np.round(rng.uniform(0.0, 6.2, n), 1)
    return frame.loc[:, EVALUATION_COLUMNS].astype(EVALUATION_DTYPES)


def cleveland_text(frame: pd.DataFrame) -> str:
    """Write `frame` as `processed.cleveland.data` lines, `num` 1 for a positive label."""
    lines = []
    for _, row in frame.iterrows():
        fields = {name: f"{float(row[name])}" for name in FEATURES if name not in {"age", "sex"}}
        age, sex, label = int(row["age"]), int(row["sex"]), int(row[LABEL])
        lines.append(cleveland_line(age=age, sex=sex, num=label, **fields))
    return "\n".join(lines) + "\n"


CLI_TEXT = cleveland_text(full_records(per_cell=8))
"""8 records per cell: 2 held out and 6 in training, enough for SMOTE's 5 neighbours."""

UNBALANCED_TEXT = cleveland_text(full_records(per_cell=8).query(f"{LABEL} == 1 or sex == 1"))
"""Without women free of disease: 12 training records with label 0 against 24 with label 1."""


def cell_list(frame: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Each row's cell, in row order, banded by hand."""
    sexes = {0: "female", 1: "male"}
    return [
        (f">={EDGE}" if age >= EDGE else f"<{EDGE}", sexes[sex], label)
        for age, sex, label in zip(frame["age"], frame["sex"], frame[LABEL], strict=True)
    ]


def cells_of(frame: pd.DataFrame) -> Counter[tuple[str, str, int]]:
    """How many rows each cell holds, banded by hand."""
    return Counter(cell_list(frame))


def assert_contract(added: pd.DataFrame, train: pd.DataFrame, quota: Mapping[Stratum, int]) -> None:
    """Every promise of `Baseline.added`, checked by hand."""
    assert list(added.columns) == EVALUATION_COLUMNS
    assert added.dtypes.astype(str).to_dict() == EVALUATION_DTYPES
    assert added.index.tolist() == list(range(-1, -len(added) - 1, -1))
    assert cells_of(added) == Counter(quota)
    for name in CATEGORICAL:
        assert added[name].isin(list(CODEBOOK[name])).all(), name
    for name in NUMERIC:
        assert added[name].between(train[name].min(), train[name].max()).all(), name
    for name in FRACTIONAL:
        tenths = added[name].to_numpy() * 10
        assert np.allclose(tenths, np.rint(tenths)), name
