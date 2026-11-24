"""The statistics every membership attack of the privacy reading shares: its scopes and its ROC.

An attack scores each record, a higher score predicting membership. `scopes` splits members and
non-members into `SCOPES`, and `attack` reads one scope's AUC (`roc_auc_score`, ties at their
midrank) and its true positive rate at each false positive rate of `FPRS`. The distance attack of
`bko.evaluate.membership` and the match attack of `bko.evaluate.donor_attacks` both read through
them (ADR `research/membership-inference-reading`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from bko.baselines.request import Stratum
from bko.datasets.cells import LABEL, cell_keys


RATES: Final[Mapping[float, str]] = MappingProxyType({0.01: "tpr01", 0.001: "tpr001"})
"""Each false positive rate a true positive rate is read at, by its measure's statistic name."""

FPRS: Final = tuple(RATES)
"""The false positive rates each scope's true positive rate is read at."""

SCOPES: Final = ("outside", "cell", "label")
"""The attack's scopes: outside the removed cell, inside it, and label 1 outside it."""

SCOPE_FPRS: Final[Mapping[str, tuple[float, ...]]] = MappingProxyType(
    {"outside": FPRS, "cell": FPRS[:1], "label": FPRS}
)
"""Each scope's rates; the cell scope's few records define no rate at 0.001."""


@dataclass(frozen=True, slots=True)
class Attack:
    """One attack's reading: its AUC and its true positive rate at each false positive rate."""

    auc: float
    tpr: Mapping[float, float]


def attack(
    member_d: np.ndarray, nonmember_d: np.ndarray, *, fprs: Sequence[float] = FPRS
) -> Attack:
    """The attack on members at distances `member_d` and non-members at `nonmember_d`.

    Each record's score is minus its distance. With no member, no non-member, or a missing
    distance, every value is NaN. A rate below one over the non-members, the least false positive
    rate one non-member makes, reads NaN: the curve does not resolve it.
    """
    distances = np.concatenate([member_d, nonmember_d])
    if not len(member_d) or not len(nonmember_d) or np.isnan(distances).any():
        return Attack(math.nan, dict.fromkeys(fprs, math.nan))
    truth = np.concatenate([np.ones(len(member_d)), np.zeros(len(nonmember_d))])
    fpr, tpr, _ = roc_curve(truth, -distances, drop_intermediate=False)
    auc = float(roc_auc_score(truth, -distances))
    floor = 1 / len(nonmember_d)
    return Attack(
        auc,
        {rate: float(np.interp(rate, fpr, tpr)) if rate >= floor else math.nan for rate in fprs},
    )


def scopes(records: pd.DataFrame, cell: Stratum, *, edges: Sequence[int]) -> dict[str, np.ndarray]:
    """Each scope's mask over `records`, by `SCOPES` name."""
    keys = cell_keys(records, edges=edges)
    inside = np.ones(len(records), dtype=bool)
    for column, value in zip(keys.columns, cell, strict=True):
        inside &= (keys[column] == value).to_numpy()
    labelled = (keys[LABEL] == 1).to_numpy()
    return {"outside": ~inside, "cell": inside, "label": ~inside & labelled}
