"""Microaggregated donor picks: a unit's numeric fields averaged over k donors of one key class.

At a spec that sets `microaggregate`, a unit holding a numeric field other than `age` is
averaged: `Means.pick` draws a primary donor and k-1 more from the primary's key class, copies
the unit's coded fields from the primary and averages each numeric field over the k donors. Every
other unit is picked as `bko.generate.donors._pick_limited` picks it. `Means.log` writes one
`micro-field` line per cell and averaged field (ADR `research/sampler-donor-limits`).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np

from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell, cell_key
from bko.generate.donors import _Limits, _Source
from bko.generate.draw_spec import DrawSpec
from bko.generate.draws import Draw
from bko.generate.errors import SamplingError
from bko.ontology.schema import AtLeast, FieldMapping


logger = logging.getLogger(__name__)


class MeanError(SamplingError):
    """A micro pick that cannot draw; `bko.generate.direct.sample` names its holdout and fill."""


def averaged_fields(fields: Sequence[str], frame: FrameSpec) -> tuple[str, ...]:
    """The fields of a unit a micro draw averages: its numeric ones but `age`, in unit order."""
    return tuple(name for name in fields if name in frame.numeric and name != "age")


def at_least(rows: Sequence[FieldMapping]) -> tuple[AtLeast, ...]:
    """Every `AtLeast` clause of `rows`, once each, in row order."""
    return tuple(dict.fromkeys(c for row in rows for c in row.when if isinstance(c, AtLeast)))


def fill_limits(
    spec: DrawSpec, rows: Sequence[FieldMapping], frame: FrameSpec
) -> tuple[_Limits | None, Means | None]:
    """A fill attempt's `_Limits` and `Means`, each None when `spec` sets nothing it holds.

    Limits come with a floor, a cap or `microaggregate`, whose k-ths they count in; `Means` with
    `microaggregate`, keyed by the `AtLeast` clauses of `rows`.
    """
    k = spec.microaggregate
    if not spec.limited and k is None:
        return None, None
    limits = _Limits(spec.min_donors, spec.reuse_cap, k or 1)
    return limits, None if k is None else Means(k, at_least(rows), frame, limits)


type _Classes = dict[Hashable, list[_Source]]
"""Records by key class."""


@dataclass(slots=True, eq=False)
class Means:
    """One fill attempt's averaged picks: k, the cutoffs, the limits and each record's class.

    Attributes:
        k: The donors per mean, the spec's `microaggregate`.
        clauses: The `AtLeast` clauses of the mapping table; those on a unit's averaged fields
            key its records.
        frame: The frame, which says which fields are numeric and how a mean is rounded.
        limits: The fill attempt's floor, cap and donations, kept in k-ths.
        keys: Each unit's record classes by record id, each computed at its first pick, so a
            later pick only filters by the cap.
        under: Each averaged field's picks since the last `log` whose cell pool held fewer
            distinct records than `floor`.
        classless: Each averaged field's picks since the last `log` whose cell pool met the floor
            but held no class of k admitted donors.
    """

    k: int
    clauses: tuple[AtLeast, ...]
    frame: FrameSpec
    limits: _Limits
    keys: dict[tuple[str, ...], dict[int, Hashable]] = field(default_factory=dict)
    under: Counter[str] = field(default_factory=Counter)
    classless: Counter[str] = field(default_factory=Counter)

    @property
    def floor(self) -> int:
        """An averaged unit's effective floor: `max(limits.floor, k)`."""
        return max(self.limits.floor, self.k)

    def pick(
        self,
        cell: Cell,
        fields: tuple[str, ...],
        own: Sequence[_Source],
        other: Sequence[_Source],
        rng: np.random.Generator,
    ) -> tuple[_Source, bool, tuple[int, ...]]:
        """A unit averaged over `k` distinct donors of one key class, its fallback and their ids.

        Records are keyed by the unit's coded values and the truth of each of `clauses` on its
        averaged fields. A record repeated in a pool counts once. The pool is formed as
        `_pick_limited` forms it, at `floor`: the cell's records, else the label group's, less
        the cell's when the floor fails. Counts are in k-ths: a primary weighs k when the unit
        holds a coded field and 1 otherwise, every other donor 1, and a pick is admitted while
        `used + weight <= cap * k`. A class with fewer than `k` admitted distinct ids counts as
        empty, so a pool with no class left falls back to `other`, as an empty pool does. The
        primary is uniform over the remaining classes' records it may lead; `k`-1 others come
        from its class without replacement. `limits.tally` counts as `_pick_limited` counts, a
        skip at the primary's weight, and `under` or `classless` counts the fallback's cause.

        Returns:
            The primary's id with its record, the averaged fields holding the donors' means
            (rounded half-even to a whole number, or to `frame.fraction_digits` for a fractional
            field, never -0.0), whether the pick fell back, and the `k` donor ids, the primary's
            first.

        Raises:
            MeanError: If neither pool holds a class of `k` admitted donors, or a record it
                keys holds NaN in an averaged field, which no frame holds.
        """
        unit = _Unit(cell, fields, averaged_fields(fields, self.frame), self)
        leads, classes, fell = unit.pools(own, other)
        if not leads:
            msg = f"{cell_key(cell)} cannot draw {'+'.join(fields)}: no class of {self.k} donors"
            msg += f" is left at min_donors={self.limits.floor} reuse_cap={self.limits.cap}"
            raise MeanError(msg + f" microaggregate={self.k}")
        donors = unit.donors(leads, classes, rng)
        record = dict(donors[0][1])
        for name in unit.numeric:
            record[name] = _mean([values[name] for _, values in donors], name, self.frame)
        return (donors[0][0], record), fell, tuple(source for source, _ in donors)

    def log(
        self,
        holdout: int,
        fill: str,
        cell: Cell,
        draws: Sequence[Draw],
        records: Sequence[_Source],
    ) -> None:
        """Log one `micro-field` line per numeric field but `age`, in column order.

        `floor` is the effective floor; `values` counts the draws averaging the field; `equal`
        those whose value equals one of its own donors' values; `primary` those whose value
        equals its primary donor's; `under` and `classless` the picks since the last call that
        fell back for each cause; `sse` sums each draw's donors' squared deviations from their
        own mean and `sst` every donor value's from the mean of all of them, each written to 6
        significant digits. `records` must hold every donor: the cell's label group. The call
        restarts `under` and `classless` at 0.
        """
        by_id = dict(records)
        for name in averaged_fields(self.frame.columns, self.frame):
            averaged = [d for d in draws if name in d.averaged]
            groups = [[float(by_id[i][name]) for i in d.averaged[name]] for d in averaged]
            logger.info(
                "micro-field holdout=%d fill=%s key=%s field=%s floor=%d values=%d equal=%d"
                " primary=%d under=%d classless=%d sse=%s sst=%s",
                holdout,
                fill,
                cell_key(cell),
                name,
                self.floor,
                len(averaged),
                sum(d.record[name] in group for d, group in zip(averaged, groups, strict=True)),
                sum(d.record[name] == by_id[d.sources[name]][name] for d in averaged),
                self.under[name],
                self.classless[name],
                f"{sum(map(_squares, groups)):.6g}",
                f"{_squares([value for group in groups for value in group]):.6g}",
            )
        self.under.clear()
        self.classless.clear()


@dataclass(frozen=True, slots=True, eq=False)
class _Unit:
    """An averaged unit at one pick: its cell, its fields, its averaged ones and its `Means`.

    Attributes:
        cell: The cell drawn, named by a refusal.
        fields: The unit's fields.
        numeric: The fields it averages.
        means: The fill attempt's `Means`.
    """

    cell: Cell
    fields: tuple[str, ...]
    numeric: tuple[str, ...]
    means: Means

    @property
    def lead(self) -> int:
        """What the primary weighs: k when the unit holds a coded field, else 1."""
        return self.means.k if len(self.numeric) < len(self.fields) else 1

    def key(self, source: _Source) -> Hashable:
        """`source`'s class: its coded values and its cutoffs' truth, computed once per id.

        Raises:
            MeanError: If the record holds NaN in an averaged field.
        """
        keys = self.means.keys.setdefault(self.fields, {})
        found = keys.get(source[0])
        if found is None:
            record = source[1]
            for name in self.numeric:
                if math.isnan(record[name]):
                    msg = f"{cell_key(self.cell)} cannot draw {'+'.join(self.fields)}: record"
                    raise MeanError(msg + f" {source[0]} holds no {name} value to average")
            clauses = self.means.clauses
            found = keys[source[0]] = (
                tuple(record[name] for name in self.fields if name not in self.numeric),
                tuple(c.holds(record[c.field]) for c in clauses if c.field in self.numeric),
            )
        return found

    def free(self, pool: Sequence[_Source], weight: int) -> list[_Source]:
        """The records of `pool` a pick of `weight` admits: `used + weight <= cap * k`."""
        limits = self.means.limits
        if limits.cap is None:
            return list(pool)
        top, used = limits.cap * self.means.k - weight, limits.used
        return [r for r in pool if used[self.fields, r[0]] <= top]

    def admitted(self, pool: Sequence[_Source]) -> tuple[list[_Source], _Classes]:
        """The records of `pool` that may lead a mean, and each class of k admitted donors."""
        classes: _Classes = {}
        for record in self.free(pool, 1):
            classes.setdefault(self.key(record), []).append(record)
        classes = {name: kept for name, kept in classes.items() if len(kept) >= self.means.k}
        return [r for r in self.free(pool, self.lead) if self.key(r) in classes], classes

    def pools(
        self, own: Sequence[_Source], other: Sequence[_Source]
    ) -> tuple[list[_Source], _Classes, bool]:
        """`admitted` on the pool `_pick_limited` would form, and whether it fell back."""
        mine = _distinct(own)
        under = len(mine) < self.means.floor
        barred = {source for source, _ in mine}
        first = [r for r in _distinct(other) if r[0] not in barred] if under else mine
        leads, classes = self.admitted(first)
        fell = under or not leads
        if fell and not under:
            leads, classes = self.admitted(_distinct(other))
        tally = self.means.limits.tally
        if under and mine:
            tally.redirects += 1
        if len(self.free(first, self.lead)) < len(first):
            tally.skips += 1
        if fell:
            (self.means.under if under else self.means.classless).update(self.numeric)
        return leads, classes, fell

    def donors(
        self, leads: Sequence[_Source], classes: _Classes, rng: np.random.Generator
    ) -> list[_Source]:
        """A primary uniform over `leads`, then k-1 more of its class; each counted in k-ths."""
        primary = leads[int(rng.integers(len(leads)))]
        rest = [r for r in classes[self.key(primary)] if r[0] != primary[0]]
        picks = rng.choice(len(rest), size=self.means.k - 1, replace=False)
        donors = [primary, *(rest[int(i)] for i in picks)]
        used = self.means.limits.used
        used[self.fields, primary[0]] += self.lead
        for source, _ in donors[1:]:
            used[self.fields, source] += 1
        return donors


def _squares(values: Sequence[float]) -> float:
    """The sum of `values`' squared deviations from their mean; 0 for none."""
    if not values:
        return 0.0
    centre = sum(values) / len(values)
    return sum((value - centre) ** 2 for value in values)


def _mean(values: Sequence[float], field: str, frame: FrameSpec) -> float:
    """`values`' mean, rounded half-even as `field` is stored; a mean rounded to zero is 0.0."""
    total = sum((Decimal(str(value)) for value in values), Decimal(0)) / len(values)
    digits = frame.fraction_digits if field in frame.fractional else 0
    return float(total.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)) or 0.0


def _distinct(pool: Sequence[_Source]) -> list[_Source]:
    """`pool` with each record id kept once, at its first place."""
    kept: dict[int, _Source] = {}
    for source, values in pool:
        kept.setdefault(source, (source, values))
    return list(kept.values())
