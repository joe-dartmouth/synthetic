"""The training records a direct draw copies field values from: label groups and donor picks.

`label_group` observes a label's training records as the group whose prior a share reads, and
`_pick` draws a donor record from a cell's matches, else from its label group's (ADRs
`research/bkb-direct-generation` and `research/data-contradiction-handling`). `_pick_limited`
draws under a donor floor and a reuse cap, and `_Limits` holds one fill attempt's counts (ADR
`research/sampler-donor-limits`). `_pick_pooled` draws by the pooled rule (ADR
`research/matched-knowledge-controls`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell, Observed, cell_key, observe_records
from bko.generate.errors import SamplingError
from bko.ontology.schema import FieldMapping


type _Source = tuple[int, dict[str, float]]
"""A training record with its id."""

type _Used = Counter[tuple[tuple[str, ...], int]]
"""How many times each (unit, record id) has donated in one fill attempt."""


@dataclass(frozen=True, slots=True, eq=False)
class _Group:
    """A label's training records: observed, as records, and each record's age band."""

    observed: Observed
    records: list[_Source]
    bands: list[str]


def _pick(
    cell: Cell,
    fields: tuple[str, ...],
    own: Sequence[_Source],
    other: Sequence[_Source],
    rng: np.random.Generator,
) -> tuple[_Source, bool]:
    """A record drawn from `own`, else from `other`, with its id, and whether it fell back.

    Raises:
        SamplingError: If both are empty.
    """
    if own:
        return own[int(rng.integers(len(own)))], False
    if other:
        return other[int(rng.integers(len(other)))], True
    msg = f"{cell_key(cell)} cannot draw {'+'.join(fields)}: no training record matches the draw"
    raise SamplingError(msg)


def _pick_pooled(
    cell: Cell,
    fields: tuple[str, ...],
    own: Sequence[_Source],
    other: Sequence[_Source],
    rng: np.random.Generator,
    *,
    weight: int,
) -> tuple[_Source, bool]:
    """A record by the pooled rule, with its id, and whether it came from `other`.

    One uniform keeps `own`, the cell's n records, with probability n/(n + `weight`); one more
    picks uniformly from `own`, else from `other`, the whole label group.

    Raises:
        SamplingError: If the pool the uniform chose is empty, which only `other` can be.
    """
    count = len(own)
    if rng.random() < (count / (count + weight) if count else 0.0):
        return own[int(rng.integers(count))], False
    if other:
        return other[int(rng.integers(len(other)))], True
    msg = f"{cell_key(cell)} cannot draw {'+'.join(fields)}: the label group holds no record"
    raise SamplingError(msg)


def _pick_limited(
    cell: Cell,
    fields: tuple[str, ...],
    own: Sequence[_Source],
    other: Sequence[_Source],
    rng: np.random.Generator,
    *,
    floor: int,
    cap: int | None,
    used: _Used,
    tally: _Tally | None = None,
) -> tuple[_Source, bool]:
    """A record drawn as `_pick` draws, under a donor floor and a reuse cap, and its fallback.

    With at least `floor` records in `own`, the pool is `own` less the donors at `cap` for
    `fields`; when that is empty, or the floor fails, the pool is `other` less the donors at the
    cap, and less `own`'s records when the floor fails. One uniform draw picks from the pool, and
    the pick counts one donation in `used`. `tally`, when given, counts a floor redirect when
    `own` holds at least one but fewer than `floor` records, and a cap skip when the first pool
    held a donor at the cap.

    Raises:
        SamplingError: If the pool is empty.
    """
    under = len(own) < floor
    if under:
        barred = {source for source, _ in own}
        first = [r for r in other if r[0] not in barred]
    else:
        first = list(own)
    pool = _below_cap(first, fields, cap, used)
    skipped = len(pool) < len(first)
    fell = under or not pool
    if fell and not under:
        pool = _below_cap(other, fields, cap, used)
    if tally is not None:
        if under and own:
            tally.redirects += 1
        if skipped:
            tally.skips += 1
    if not pool:
        msg = f"{cell_key(cell)} cannot draw {'+'.join(fields)}: no matching donor is left at"
        msg += f" min_donors={floor} reuse_cap={cap}"
        raise SamplingError(msg)
    chosen = pool[int(rng.integers(len(pool)))]
    used[fields, chosen[0]] += 1
    return chosen, fell


def _below_cap(
    pool: Sequence[_Source], fields: tuple[str, ...], cap: int | None, used: _Used
) -> list[_Source]:
    return [r for r in pool if cap is None or used[fields, r[0]] < cap]


@dataclass(slots=True, eq=False)
class _Tally:
    """The picks each limit changed since the last `_Limits.taken`.

    Attributes:
        redirects: Picks whose cell held at least one but fewer than `floor` matching records.
        skips: Picks whose first pool held a donor at the cap.
    """

    redirects: int = 0
    skips: int = 0


@dataclass(slots=True, eq=False)
class _Limits:
    """One fill attempt's floor, cap and donations, with the picks each limit changed.

    Attributes:
        floor: The spec's `min_donors`.
        cap: The spec's `reuse_cap`.
        used: Each (unit, record id)'s donations so far.
        tally: The redirects and skips `_pick_limited` counted since the last `taken`.
    """

    floor: int
    cap: int | None
    used: _Used = field(default_factory=Counter)
    tally: _Tally = field(default_factory=_Tally)

    def taken(self) -> tuple[int, int]:
        """The redirects and skips since the last call, which restarts both at 0."""
        counts = (self.tally.redirects, self.tally.skips)
        self.tally = _Tally()
        return counts

    def pick(
        self,
        cell: Cell,
        fields: tuple[str, ...],
        own: Sequence[_Source],
        other: Sequence[_Source],
        rng: np.random.Generator,
    ) -> tuple[_Source, bool]:
        """`_pick_limited` at these limits, counting into `tally`."""
        return _pick_limited(
            cell,
            fields,
            own,
            other,
            rng,
            floor=self.floor,
            cap=self.cap,
            used=self.used,
            tally=self.tally,
        )


def label_group(
    train: pd.DataFrame,
    label: int,
    rows: Sequence[FieldMapping],
    *,
    name: str,
    frame: FrameSpec,
    pairs: Sequence[tuple[str, str]] = (),
) -> Observed:
    """The training records of `label`, observed as the label group whose prior a share reads.

    `pairs` are counted as `observe_records` counts them.

    Raises:
        GapError: As `observe_records`.
    """
    return _observe_group(
        members(train, label, frame), label, rows, name=name, frame=frame, pairs=pairs
    )


def members(train: pd.DataFrame, label: int, frame: FrameSpec) -> pd.DataFrame:
    """The training records of `label`: its label group."""
    return train.loc[train[frame.label].eq(label)]


def cell_records(
    train: pd.DataFrame, keys: pd.DataFrame, cell: Cell, frame: FrameSpec
) -> list[_Source]:
    """`_records` of the training records whose `keys`, `cell_keys(train)`, are `cell`."""
    return _records(
        train.loc[keys.eq(pd.Series(cell, index=keys.columns)).all(axis=1).to_numpy()], frame
    )


def _observe_group(
    members: pd.DataFrame,
    label: int,
    rows: Sequence[FieldMapping],
    *,
    name: str,
    frame: FrameSpec,
    pairs: Sequence[tuple[str, str]] = (),
) -> Observed:
    """`members`, the training records of `label`, observed under the label group's name."""
    return observe_records(members, rows, name=f"{name}/{frame.label}={label}", pairs=pairs)


def _label_groups(
    train: pd.DataFrame,
    keys: pd.DataFrame,
    labels: Collection[int],
    rows: Sequence[FieldMapping],
    name: str,
    frame: FrameSpec,
    pairs: Sequence[tuple[str, str]] = (),
) -> dict[int, _Group]:
    groups: dict[int, _Group] = {}
    for label in sorted(labels):
        group = members(train, label, frame)
        observed = _observe_group(group, label, rows, name=name, frame=frame, pairs=pairs)
        bands = [str(band) for band in keys.loc[group.index, keys.columns[0]]]
        groups[label] = _Group(observed, _records(group, frame), bands)
    return groups


def _records(frame: pd.DataFrame, spec: FrameSpec) -> list[_Source]:
    """Each record of `frame` with its id, in row order."""
    columns = [str(column) for column in spec.columns]
    return [
        (int(index), dict(zip(columns, values, strict=True)))
        for index, *values in frame[columns].itertuples(name=None)
    ]
