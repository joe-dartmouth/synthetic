"""A frame shaped like the archive's cells, fast methods, a small experiment and a classifier spy.

`GRADED` keeps the archive's asymmetry: a cell of 2 records that is never held out and a cell under
the threshold with test records of its own, beside full cells.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bko.baselines.methods import METHODS
from bko.baselines.request import Method
from bko.baselines.synthesizers import ctgan, tvae
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.classifier import build
from bko.evaluate.data import Data, digest
from bko.evaluate.experiment import BKO_ARMS, E1
from bko.evaluate.run import method_arms
from tests._baselines_fixtures import EDGE, TINY_CTGAN, TINY_TVAE, cell_list, full_records


FAST_METHODS = MappingProxyType(
    {
        **METHODS,
        Method.CTGAN: partial(ctgan, parameters=TINY_CTGAN),
        Method.TVAE: partial(tvae, parameters=TINY_TVAE),
    }
)
"""Every method, with small SDV models; `method_arms(FAST_METHODS)` makes them arms."""

FLAGGED = ((f"<{EDGE}", "female", 1), (f">={EDGE}", "female", 1))
"""The graded frame's underrepresented cells, in grid order."""


def graded() -> pd.DataFrame:
    """26 records per cell, but 2 in `<55 female 1` and 23 in `>=55 female 1`.

    On every seed: 146 training and 35 test records, those two cells flagged with quotas 18 and 2,
    and no test record in `<55 female 1`.
    """
    base = full_records(26)
    cells = pd.Series(cell_list(base), index=base.index)
    sizes = dict(zip(FLAGGED, (2, 23), strict=True))
    position = base.groupby(cells).cumcount()
    keep = [rank < sizes.get(cell, 26) for rank, cell in zip(position, cells, strict=True)]
    return base.loc[keep]


FAST_ARMS = MappingProxyType(
    {**method_arms(FAST_METHODS), **{name: method_arms(FAST_METHODS)["smote"] for name in BKO_ARMS}}
)
"""Every E1 arm, fast: each BKO name runs SMOTE, so no test reaches the API."""

GRADED = graded()


def uci_data(records: pd.DataFrame, name: str = "uci") -> Data:
    """`records` as UCI's `Data`, with no field rows."""
    return Data(records, UCI_FRAME, (), name, digest(records))


GRADED_DATA = uci_data(GRADED)
"""`GRADED` as UCI's `Data`."""

SMALL = dataclasses.replace(E1, holdout_seeds=(0, 1))
"""E1 on two holdouts."""


class Recording(LogisticRegression):
    """A logistic regression that keeps the labels and weights of its last fit."""

    def fit(self, features, labels, sample_weight=None):
        self.seen = (np.asarray(labels), np.asarray(sample_weight))
        return super().fit(features, labels, sample_weight=sample_weight)


class Spy:
    """A classifier factory that keeps every model it builds."""

    def __init__(self) -> None:
        self.models: list[Recording] = []

    def __call__(self, frame: FrameSpec) -> Pipeline:
        model = Recording(**build(frame).named_steps["model"].get_params())
        self.models.append(model)
        pipeline = build(frame)
        pipeline.set_params(model=model)
        return pipeline
