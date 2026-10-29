"""Every method by name, and `augment`, the one call the command and the harness make."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import pandas as pd

from bko.baselines.classical import cart, gaussian_copula
from bko.baselines.controls import class_weighting, no_augmentation
from bko.baselines.errors import BaselineError
from bko.baselines.oversample import adasyn, smote
from bko.baselines.records import Baseline, check
from bko.baselines.request import Method, Request
from bko.baselines.synthesizers import ctgan, tvae


type Runner = Callable[[pd.DataFrame, Request], Baseline]
"""Builds one method's `Baseline` from a training split and a request."""

METHODS: Final[Mapping[Method, Runner]] = MappingProxyType(
    {
        Method.NONE: no_augmentation,
        Method.CLASS_WEIGHT: class_weighting,
        Method.SMOTE: smote,
        Method.ADASYN: adasyn,
        Method.CTGAN: ctgan,
        Method.TVAE: tvae,
        Method.GAUSSIAN_COPULA: gaussian_copula,
        Method.CART: cart,
    }
)
"""Each built method's runner; the command offers exactly these."""


def augment(
    train: pd.DataFrame,
    method: Method,
    request: Request,
    *,
    methods: Mapping[Method, Runner] = METHODS,
) -> Baseline:
    """Run `method` on the training split `train` and check its output against the contract.

    Raises:
        BaselineError: If a control is asked for rows, a row-adding method for none, or the method
            cannot fill the request.
        OutputContractError: If the method's output breaks the contract.
    """
    if method.adds_rows != (request.rows > 0):
        need = "at least one row" if method.adds_rows else "no rows"
        msg = f"{method} needs {need}; the request adds {request.rows}"
        raise BaselineError(msg)
    baseline = methods[method](train, request)
    check(baseline, train, request)
    return baseline
