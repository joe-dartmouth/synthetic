"""Microaggregated donor picks: a unit's numeric fields averaged over k donors of one key class.

At a spec that sets `microaggregate`, a unit holding a numeric field other than `age` is
averaged: `pick_mean` draws a primary donor and k-1 more from the primary's key class, copies the
unit's coded fields from the primary and averages each numeric field over the k donors. Every
other unit is picked as `bko.generate.donors._pick_limited` picks it. `log_fields` writes one
`micro-field` line per cell and averaged field (ADR `research/sampler-donor-limits`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np

from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell, cell_key
from bko.generate.donors import _below_cap, _Limits, _Source
from bko.generate.draws import Draw
from bko.generate.errors import SamplingError
from bko.ontology.schema import AtLeast, FieldMapping


logger = logging.getLogger(__name__)


def averaged_fields(fields: Sequence[str], frame: FrameSpec) -> tuple[str, ...]:
    """The fields of a unit a micro draw averages: its numeric ones but `age`, in unit order."""
    return tuple(name for name in fields if name in frame.numeric and name != "age")


def at_least(rows: Sequence[FieldMapping]) -> tuple[AtLeast, ...]:
    """Every `AtLeast` clause of `rows`, once each, in row order."""
    return tuple(dict.fromkeys(c for row in rows for c in row.when if isinstance(c, AtLeast)))


def pick_mean(
    cell: Cell,
    fields: tuple[str, ...],
    own: Sequence[_Source],
    other: Sequence[_Source],
    rng: np.random.Generator,
    *,
    k: int,
    limits: _Limits,
    frame: FrameSpec,
    clauses: Sequence[AtLeast],
) -> tuple[_Source, bool, tuple[int, ...]]:
    """A unit averaged over `k` distinct donors of one key class, its fallback and their ids.

    Records are keyed by the unit's coded values, the truth of each of `clauses` on its averaged
    fields and each averaged field's missingness. A record repeated in a pool counts once. The
    pool is formed as `_pick_limited` forms it, at the floor `max(limits.floor, k)`: the cell's
    records, else the label group's, less the cell's when the floor fails. Counts are in k-ths:
    a primary weighs k when the unit holds a coded field and 1 otherwise, every other donor 1,
    and a pick is admitted while `used + weight <= cap * k`. A class with fewer than `k` admitted
    distinct ids counts as empty, so a pool with no class left falls back to `other`, as an empty
    pool does. The primary is uniform over the remaining classes' records it may lead; `k`-1
    others come from its class without replacement. `limits.tally` counts as `_pick_limited`
    counts.

    Returns:
        The primary's id with its record, the averaged fields holding the donors' means (rounded
        half-even to a whole number, or to `frame.fraction_digits` for a fractional field),
        whether the pick fell back, and the `k` donor ids, the primary's first.

    Raises:
        SamplingError: If neither pool holds a class of `k` admitted donors.
    """
    unit = _Unit(fields, averaged_fields(fields, frame), tuple(clauses), k, limits)
    leads, classes, fell = unit.pools(own, other)
    if not leads:
        msg = f"{cell_key(cell)} cannot draw {'+'.join(fields)}: no class of {k} donors is left"
        msg += f" at min_donors={limits.floor} reuse_cap={limits.cap} microaggregate={k}"
        raise SamplingError(msg)
    donors = unit.donors(leads, classes, rng)
    record = dict(donors[0][1])
    for name in unit.numeric:
        record[name] = _mean([values[name] for _, values in donors], name, frame)
    return (donors[0][0], record), fell, tuple(source for source, _ in donors)


type _Classes = dict[Hashable, list[_Source]]
"""Records by key class."""


@dataclass(frozen=True, slots=True, eq=False)
class _Unit:
    """An averaged unit at one pick: its fields, its averaged ones, its cutoffs and its limits.

    Attributes:
        fields: The unit's fields.
        numeric: The fields it averages.
        clauses: The `AtLeast` clauses of the mapping table; those on `numeric` key a record.
        k: The donors per mean.
        limits: The fill attempt's floor, cap and donations, kept in k-ths.
    """

    fields: tuple[str, ...]
    numeric: tuple[str, ...]
    clauses: tuple[AtLeast, ...]
    k: int
    limits: _Limits

    @property
    def lead(self) -> int:
        """What the primary weighs: k when the unit holds a coded field, else 1."""
        return self.k if len(self.numeric) < len(self.fields) else 1

    def key(self, record: Mapping[str, float]) -> Hashable:
        """`record`'s class: its coded values, its cutoffs' truth and its fields' missingness."""
        return (
            tuple(record[name] for name in self.fields if name not in self.numeric),
            tuple(c.holds(record[c.field]) for c in self.clauses if c.field in self.numeric),
            tuple(math.isnan(record[name]) for name in self.numeric),
        )

    def free(self, pool: Sequence[_Source], weight: int) -> list[_Source]:
        """The records of `pool` a pick of `weight` admits under the cap."""
        limits = self.limits
        return _below_cap(pool, self.fields, limits.cap, limits.used, weight, self.k)

    def admitted(self, pool: Sequence[_Source]) -> tuple[list[_Source], _Classes]:
        """The records of `pool` that may lead a mean, and each class of k admitted donors."""
        classes: _Classes = {}
        for record in self.free(pool, 1):
            classes.setdefault(self.key(record[1]), []).append(record)
        classes = {name: kept for name, kept in classes.items() if len(kept) >= self.k}
        return [r for r in self.free(pool, self.lead) if self.key(r[1]) in classes], classes

    def pools(
        self, own: Sequence[_Source], other: Sequence[_Source]
    ) -> tuple[list[_Source], _Classes, bool]:
        """`admitted` on the pool `_pick_limited` would form, and whether it fell back."""
        mine = _distinct(own)
        under = len(mine) < max(self.limits.floor, self.k)
        barred = {source for source, _ in mine}
        first = [r for r in _distinct(other) if r[0] not in barred] if under else mine
        leads, classes = self.admitted(first)
        fell = under or not leads
        if fell and not under:
            leads, classes = self.admitted(_distinct(other))
        if under and mine:
            self.limits.tally.redirects += 1
        if len(self.free(first, 1)) < len(first):
            self.limits.tally.skips += 1
        return leads, classes, fell

    def donors(
        self, leads: Sequence[_Source], classes: _Classes, rng: np.random.Generator
    ) -> list[_Source]:
        """A primary uniform over `leads`, then k-1 more of its class; each counted in k-ths."""
        primary = leads[int(rng.integers(len(leads)))]
        rest = [r for r in classes[self.key(primary[1])] if r[0] != primary[0]]
        picks = rng.choice(len(rest), size=self.k - 1, replace=False)
        donors = [primary, *(rest[int(i)] for i in picks)]
        self.limits.used[self.fields, primary[0]] += self.lead
        for source, _ in donors[1:]:
            self.limits.used[self.fields, source] += 1
        return donors


def log_fields(
    holdout: int, fill: str, cell: Cell, draws: Sequence[Draw], records: Sequence[_Source]
) -> None:
    """Log one `micro-field` line per averaged field of `cell`'s draws, in column order.

    `values` counts the draws averaging the field; `equal` those whose value equals one of its
    own donors' values; `primary` those whose value equals its primary donor's; `sse` sums each
    draw's donors' squared deviations from their own mean and `sst` every donor value's from the
    mean of all of them, each written to 6 significant digits. `records` must hold every donor:
    the cell's label group.
    """
    if not draws:
        return
    by_id = dict(records)
    for name in [name for name in draws[0].record if any(name in d.averaged for d in draws)]:
        averaged = [d for d in draws if name in d.averaged]
        groups = [[float(by_id[i][name]) for i in d.averaged[name]] for d in averaged]
        pooled = [value for group in groups for value in group if not math.isnan(value)]
        sse = sum(_squares(group) for group in groups if not any(map(math.isnan, group)))
        logger.info(
            "micro-field holdout=%d fill=%s key=%s field=%s values=%d equal=%d primary=%d"
            " sse=%s sst=%s",
            holdout,
            fill,
            cell_key(cell),
            name,
            len(averaged),
            sum(d.record[name] in group for d, group in zip(averaged, groups, strict=True)),
            sum(d.record[name] == by_id[d.sources[name]][name] for d in averaged),
            f"{sse:.6g}",
            f"{_squares(pooled):.6g}",
        )


def _squares(values: Sequence[float]) -> float:
    """The sum of `values`' squared deviations from their mean; 0 for none."""
    if not values:
        return 0.0
    centre = sum(values) / len(values)
    return sum((value - centre) ** 2 for value in values)


def _mean(values: Sequence[float], field: str, frame: FrameSpec) -> float:
    """`values`' mean, rounded half-even as `field` is stored; NaN when they are missing."""
    if any(math.isnan(value) for value in values):
        return math.nan
    total = sum((Decimal(str(value)) for value in values), Decimal(0)) / len(values)
    digits = frame.fraction_digits if field in frame.fractional else 0
    return float(total.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN))


def _distinct(pool: Sequence[_Source]) -> list[_Source]:
    """`pool` with each record id kept once, at its first place."""
    kept: dict[int, _Source] = {}
    for source, values in pool:
        kept.setdefault(source, (source, values))
    return list(kept.values())
