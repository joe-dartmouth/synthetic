"""The field-group match attack of `python -m bko.evaluate privacy`.

The match attack reads verbatim copies. Its units are the frame's field groups
(`bko.generate.fields.field_groups`) plus a singleton for every column in no group but `sex` and
the label, the same for every arm. A record's score is the largest number of units on which one
added row equals it on every field, a missing value equal to a missing one. Its members,
non-members, scopes and statistics are the distance attack's (`bko.evaluate.attack_stats`). ADR
`research/membership-inference-reading` records the attack.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.request import Stratum
from bko.datasets.frames import FrameSpec
from bko.evaluate.attack_stats import RATES, SCOPE_FPRS, SCOPES, attack, scopes
from bko.evaluate.scores import Measure
from bko.generate.fields import field_groups
from bko.ontology.schema import FieldMapping


CHUNK: Final = 1024
"""Records per chunk of `match_scores`; a chunk's equalities take `CHUNK` times the added rows."""


def match(statistic: str, scope: str) -> Measure:
    """The match attack's measure of `statistic` (`auc`, `tpr01` or `tpr001`) at `scope`."""
    return Measure(f"match-{statistic}-{scope}")


MATCH_MEASURES: Final[tuple[Measure, ...]] = tuple(
    match(statistic, scope)
    for scope in SCOPES
    for statistic in ("auc", *(RATES[rate] for rate in SCOPE_FPRS[scope]))
)
"""The match attack's measures, by scope, as the distance attack's are."""


def match_groups(rows: Sequence[FieldMapping], *, frame: FrameSpec) -> tuple[tuple[str, ...], ...]:
    """The match attack's units, ordered by each unit's first column.

    They are `rows`' field groups, then each other column but `sex` and the label as a singleton.
    """
    grouped = [group.fields for group in field_groups(rows, frame=frame)]
    held = {name for fields in grouped for name in fields}
    single = [(name,) for name in frame.columns if name not in {*held, "sex", frame.label}]
    order = {name: index for index, name in enumerate(frame.columns)}
    return tuple(sorted([*grouped, *single], key=lambda unit: order[unit[0]]))


def match_scores(
    records: pd.DataFrame, added: pd.DataFrame, groups: Sequence[tuple[str, ...]]
) -> np.ndarray:
    """Each record's score: the most units of `groups` one row of `added` holds as it does.

    Each unit is coded once over records and rows together, and the equalities are summed a
    chunk of `CHUNK` records at a time, so no records by rows by units array is built.
    """
    found = np.zeros(len(records), dtype=np.int64)
    if not len(added) or not len(records):
        return found
    codes = [_unit_codes(records, added, unit) for unit in groups]
    for start in range(0, len(records), CHUNK):
        stop = min(start + CHUNK, len(records))
        held = np.zeros((stop - start, len(added)), dtype=np.int64)
        for ours, theirs in codes:
            held += ours[start:stop, None] == theirs[None, :]
        found[start:stop] = held.max(axis=1)
    return found


def _unit_codes(
    records: pd.DataFrame, added: pd.DataFrame, unit: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """One code per record and per row for `unit`'s values; equal values share a code."""
    joint = np.zeros(len(records) + len(added), dtype=np.int64)
    for name in unit:
        values = pd.concat([records[name], added[name]], ignore_index=True)
        column, uniques = pd.factorize(values, use_na_sentinel=False)
        joint, _ = pd.factorize(joint * len(uniques) + column)
    return joint[: len(records)], joint[len(records) :]


def match_measures(
    added: pd.DataFrame,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cell: Stratum,
    frame: FrameSpec,
    groups: Sequence[tuple[str, ...]],
) -> dict[Measure, float]:
    """One arm's `MATCH_MEASURES` on one holdout; every one NaN for an arm with no rows.

    Members are the training records and non-members the test records, read in the distance
    attack's scopes at its rates.
    """
    if not len(added):
        return dict.fromkeys(MATCH_MEASURES, math.nan)
    member_s = match_scores(train, added, groups).astype(float)
    nonmember_s = match_scores(test, added, groups).astype(float)
    members, nonmembers = (
        scopes(train, cell, edges=frame.edges),
        scopes(test, cell, edges=frame.edges),
    )
    found: dict[Measure, float] = {}
    for scope in SCOPES:
        read = attack(
            -member_s[members[scope]], -nonmember_s[nonmembers[scope]], fprs=SCOPE_FPRS[scope]
        )
        found[match("auc", scope)] = read.auc
        found |= {match(RATES[rate], scope): value for rate, value in read.tpr.items()}
    return found
