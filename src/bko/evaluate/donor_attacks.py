"""The field-group match attack and the donor reuse of `python -m bko.evaluate privacy`.

The match attack reads verbatim copies. Its units are the frame's field groups
(`bko.generate.fields.field_groups`) plus a singleton for every column in no group but `sex` and
the label, the same for every arm. A record's score is the largest number of units on which one
added row equals it on every field, a missing value equal to a missing one. Its members,
non-members, scopes and statistics are the distance attack's (`bko.evaluate.attack_stats`).

Donor reuse reads an arm of `DONOR_ARMS` from a re-draw of its gap fill at its `arm` line's seed
(`bko.evaluate.redraw.redraw`) under `trace`'s identity checks: a direct arm's re-drawn rows hash
to the line's sha256, and a perturbation arm's stored rows are their draws outside the perturbed
fields. A training record donates to an added row when the row holds, verbatim, a unit every
field of which its draw read from that record. ADR `research/membership-inference-reading`
records both readings.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.request import Request, Stratum
from bko.datasets.frames import FrameSpec
from bko.evaluate.attack_stats import RATES, SCOPE_FPRS, SCOPES, attack, scopes
from bko.evaluate.data import Data, digest
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import DIRECT_ARMS
from bko.evaluate.holdout_candidates import PERTURBS, Candidate
from bko.evaluate.perturb import unmapped
from bko.evaluate.redraw import redraw
from bko.evaluate.run_folder import fields
from bko.evaluate.scores import Measure
from bko.evaluate.shared import rebuilt
from bko.evaluate.trace_direct import arm_spec
from bko.evaluate.trace_lines import Fill, ordered, returned_lines
from bko.evaluate.trace_perturb import kept_draws, line_draws
from bko.generate.direct import Draw
from bko.generate.draw_spec import DEFAULT_SPEC
from bko.generate.fields import field_groups
from bko.ontology.schema import FieldMapping
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.evaluate")

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


DONOR_ARMS: Final = frozenset({*DIRECT_ARMS, *PERTURBS})
"""The arms whose fills a re-draw reproduces, and so whose donors the reading names."""

DONOR_MEASURES: Final = (Measure.DONOR_MAX, Measure.DONOR_P95)
"""What `donor_measures` returns, in order."""

QUANTILE: Final = 0.95
"""The upper quantile of the donations `donor_measures` reads."""


def donor_counts(
    fill: Fill,
    stored: pd.DataFrame,
    train: pd.DataFrame,
    request: Request,
    *,
    lines: Sequence[Candidate],
    data: Data,
    ontologies: Mapping[str, Ontology],
    budget: int,
    groups: Sequence[tuple[str, ...]],
) -> Counter[int]:
    """How many of `stored`'s rows each training record donates a verbatim unit to.

    The fill is re-drawn at its `arm` line's seed, a direct arm's at its spec and a perturbation
    arm's at `bko-direct`'s, and each stored row is aligned with its draw in the order
    `bko.evaluate.shared.rebuilt` places them. A direct arm's draws are the re-draw's kept ones;
    a perturbation arm's are those of its candidate `lines` that its own attempt kept, each
    returned line paired with its draw in generation order as `trace` pairs them
    (`bko.evaluate.trace_perturb.line_draws`).

    Raises:
        IdentityError: If a direct arm's re-drawn rows do not hash to its `arm` line's sha256; if
            a perturbation arm's `lines` rebuild rows that do not, are not one per draw or
            differ from their draws outside the perturbed fields; if the stored rows are not one
            per kept draw, or a perturbation arm's differ from their draws outside the perturbed
            fields; or as `arm_spec`, `ordered` and `rebuilt`.
        EvaluationError: As `arm_spec` and `bko.evaluate.perturb.unmapped`.
    """
    where = f"holdout {fill.holdout}'s {fill.arm} fill"
    perturbed = fill.name in PERTURBS
    spec = DEFAULT_SPEC if perturbed else arm_spec(fill, data, ontologies)
    seeded = dataclasses.replace(request, seed=fill.seed)
    tag = (fill.holdout, fill.kind)
    redrawn = redraw(
        train, seeded, data=data, ontologies=ontologies, budget=budget, tag=tag, spec=spec
    )
    logged = fields(fill.line)["sha256"]
    free = set(unmapped(data.field_rows, frame=data.frame)) if perturbed else set()
    if perturbed:
        draws = _perturbed_draws(
            lines, redrawn.drawn, seeded, logged=logged, where=where, free=free
        )
    else:
        if digest(redrawn.rows) != logged:
            msg = f"{where} re-drawn at seed {fill.seed} hashes to {digest(redrawn.rows)}, not"
            msg += f" its arm line's {logged}"
            raise IdentityError(msg)
        order = ordered([draw.record for draw in redrawn.kept], seeded, where=where)
        draws = [redrawn.kept[k] for k in order]
    if len(draws) != len(stored):
        msg = f"{where} holds {len(stored)} rows, not one per kept draw of {len(draws)}"
        raise IdentityError(msg)
    rows = [{str(name): value for name, value in row.items()} for row in stored.to_dict("records")]
    if perturbed:
        names = {
            name
            for row, draw in zip(rows, draws, strict=True)
            for name in draw.record
            if name not in free and not _same(row[name], draw.record[name])
        }
        if names:
            msg = f"{where}: columns no perturbation changes differ from the re-draw:"
            msg += f" {','.join(sorted(names))}"
            raise IdentityError(msg)
    return donations(rows, draws, groups)


def _perturbed_draws(
    lines: Sequence[Candidate],
    drawn: Sequence[Draw],
    request: Request,
    *,
    logged: str,
    where: str,
    free: Collection[str],
) -> list[Draw]:
    """The draws of a perturbation fill's kept lines, in row order, once `trace`'s checks hold.

    Raises:
        IdentityError: If `lines` rebuild rows that do not hash to `logged`, are not one per
            draw or differ from their draws outside `free`; or as `rebuilt` and `kept_draws`.
    """
    found = digest(rebuilt(lines, request, where=where))
    if found != logged:
        msg = f"{where}'s rows rebuilt from its lines hash to {found}, not its arm line's {logged}"
        raise IdentityError(msg)
    _, returned = returned_lines(lines)
    paired, differs = line_draws(returned, drawn, sorted(free))
    if differs:
        msg = f"{where}: {differs}"
        raise IdentityError(msg)
    return [draw for _, _, draw in kept_draws(paired, request, where=where)]


def donations(
    rows: Sequence[Mapping[str, object]], draws: Sequence[Draw], groups: Sequence[tuple[str, ...]]
) -> Counter[int]:
    """How many of `rows` each training record donates to, row i over draw i.

    A record donates to a row when the row holds, as drawn, every field of a unit of `groups`
    that the draw read from that record; a field a perturbation changed stops counting.
    """
    found: Counter[int] = Counter()
    for row, draw in zip(rows, draws, strict=True):
        donors = set()
        for unit in groups:
            sources = {draw.sources.get(name) for name in unit}
            if len(sources) != 1:
                continue
            (source,) = sources
            if source is not None and all(_same(row[n], draw.record[n]) for n in unit):
                donors.add(source)
        found.update(donors)
    return found


def donor_measures(counts: Counter[int], *, rows: int) -> dict[Measure, float]:
    """The most of a fill's `rows` one record donates to, and the 95th percentile over donors.

    The percentile is linear, over the records that donate. A fill with rows but no donation
    reads a maximum of 0 and a NaN percentile, over no donor; a fill with no rows reads NaN for
    both.
    """
    if not counts:
        return {
            Measure.DONOR_MAX: 0.0 if rows else math.nan,
            Measure.DONOR_P95: math.nan,
        }
    values = np.fromiter(counts.values(), dtype=float)
    return {
        Measure.DONOR_MAX: float(values.max()),
        Measure.DONOR_P95: float(np.quantile(values, QUANTILE, method="linear")),
    }


def read_donors(
    fill: Fill,
    stored: pd.DataFrame,
    train: pd.DataFrame,
    request: Request,
    *,
    lines: Sequence[Candidate],
    data: Data,
    ontologies: Mapping[str, Ontology],
    budget: int,
    groups: Sequence[tuple[str, ...]],
) -> dict[Measure, float]:
    """`donor_measures` of `donor_counts`, logged with the values the CSV holds.

    The line is `privacy donor holdout=%d arm=%s donors=%d max=%.0f p95=%.2f seconds=%.1f`,
    `seconds` the time the re-draw and its reading took.

    Raises:
        IdentityError: As `donor_counts`.
        EvaluationError: As `donor_counts`.
    """
    started = time.perf_counter()
    counts = donor_counts(
        fill,
        stored,
        train,
        request,
        lines=lines,
        data=data,
        ontologies=ontologies,
        budget=budget,
        groups=groups,
    )
    found = donor_measures(counts, rows=len(stored))
    logger.info(
        "privacy donor holdout=%d arm=%s donors=%d max=%.0f p95=%.2f seconds=%.1f",
        fill.holdout,
        fill.arm,
        len(counts),
        found[Measure.DONOR_MAX],
        found[Measure.DONOR_P95],
        time.perf_counter() - started,
    )
    return found


def _same(ours: object, theirs: object) -> bool:
    """Whether two values are equal, a missing value equal to a missing one."""
    if _missing(ours) and _missing(theirs):
        return True
    return bool(ours == theirs)


def _missing(value: object) -> bool:
    """Whether `value` is a missing float."""
    return isinstance(value, float) and math.isnan(value)
