"""`python -m bko.evaluate privacy`: the membership attack and the privacy reading of a stored run.

The attack scores each record by minus its DCR-metric distance to the nearest of an arm's added
rows, every measured column's span read from the training split (`privacy.nearest_from`). Members
are the holdout's prepared training split; non-members are its whole test split, copied records
included; the removed records are neither. It reads three scopes: the records outside the removed
cell, the removed cell's (undefined with no member in it, as at a removal fraction of 1), and
label 1's outside the cell. Each scope reads the AUC (`roc_auc_score`, ties at their midrank) and
the true positive rate at false positive rates 0.01 and 0.001 (`bko.evaluate.attack_stats`), by
linear interpolation on `roc_curve(drop_intermediate=False)`; the cell scope reads no rate at
0.001, and a rate below one over the scope's non-members reads NaN, below what the curve
resolves. Each arm also reads the NNDR and DCR of its added rows to the training split, and the
held-out rows the same as a baseline (`privacy.distances_and_nndr`).

Each arm also reads the match attack and, for an arm of `DONOR_ARMS`, donor reuse
(`bko.evaluate.donor_attacks`); both attacks read their scopes and statistics through
`bko.evaluate.attack_stats`.

`privacy` reads one or more `rows` stores (`bko.evaluate.rows_read`), one per holdout slice of
the named run, makes no API call, and writes the scores CSV. It logs the reading rule over
`GROUPS` against the run family's `REFERENCE` arm (`bko.evaluate.reference`) through
`bko.evaluate.privacy_rule`, which judges no criterion; each scope's attack statistics, each
scope's match statistics and the distances are separate families. ADR
`research/membership-inference-reading` records the attacks, donor reuse and the rule.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from bko.baselines.request import Stratum
from bko.datasets.frames import FrameSpec
from bko.datasets.run_folders import runs_dir_argument
from bko.evaluate.attack_stats import (
    RATES,
    SCOPE_FPRS,
    SCOPES,
    attack,
    scopes,
)
from bko.evaluate.data import Data
from bko.evaluate.donor_attacks import (
    DONOR_ARMS,
    DONOR_MEASURES,
    match,
    match_groups,
    match_measures,
    read_donors,
)
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment, gap_request
from bko.evaluate.named_runs import DEFAULT_RUN, NamedRun
from bko.evaluate.prepare import Prepared, held_out_ratio
from bko.evaluate.privacy import distances_and_nndr, nearest_from
from bko.evaluate.privacy_rule import Group, log_readings, readings
from bko.evaluate.privacy_stores import (
    SHOWN,  # ruff: ignore[unused-import]  (re-exported)
    Store,
    _child,  # ruff: ignore[unused-import]  (re-exported)
    _shown,  # ruff: ignore[unused-import]  (re-exported)
    checked,
    read,
    read_fills,
)
from bko.evaluate.reference import (
    REFERENCE,  # ruff: ignore[unused-import]  (re-exported)
    reference_of,
)
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, render
from bko.evaluate.trace_lines import Fill
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.evaluate")


def mi(statistic: str, scope: str) -> Measure:
    """The attack's measure of `statistic` (`auc`, `tpr01` or `tpr001`) at `scope`."""
    return Measure(f"mi-{statistic}-{scope}")


MI_MEASURES: Final = tuple(
    mi(statistic, scope)
    for scope in SCOPES
    for statistic in ("auc", *(RATES[rate] for rate in SCOPE_FPRS[scope]))
)
"""The attack's measures, by scope; the cell scope's few records define no rate at 0.001."""

DISTANCE_MEASURES: Final = (
    Measure.DCR_P5,
    Measure.DCR_MEDIAN,
    Measure.NNDR_P5,
    Measure.NNDR_MEDIAN,
    Measure.NNDR_TIES,
    Measure.NNDR_EXACT,
)
"""What `privacy.distances_and_nndr` returns, in order."""

ADDED_MEASURES: Final = (*MI_MEASURES, *DISTANCE_MEASURES)
"""Each arm's measures, in the order the CSV writes them."""

GROUPS: Final = (
    *(Group(scope, (mi("auc", scope), mi("tpr01", scope))) for scope in SCOPES),
    Group(Scope.ALL.value, DISTANCE_MEASURES[:4], higher_leaks=False),
    *(Group(f"match-{scope}", (match("auc", scope), match("tpr01", scope))) for scope in SCOPES),
)
"""The rule's families: each scope's AUC and TPR at 0.01, the DCR and NNDR quantiles, which
stay descriptive and leak more the lower they read, and each scope's match attack AUC and TPR at
0.01 (`bko.evaluate.donor_attacks`). Each is a separate family at family-wise 0.95, whose size is
its pairs of arm and measure with a defined contrast."""


def exposure(
    records: pd.DataFrame, added: pd.DataFrame, *, train: pd.DataFrame, frame: FrameSpec
) -> np.ndarray:
    """Each record's distance to its nearest added row on `train`'s spans; NaN with no rows."""
    if not len(added):
        return np.full(len(records), np.nan)
    return nearest_from(records, added, frame=frame, spans_from=train)


def arm_measures(
    added: pd.DataFrame,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cell: Stratum,
    frame: FrameSpec,
    holdout: int,
    arm: str,
) -> dict[Measure, float]:
    """One arm's `ADDED_MEASURES` on one holdout: its attack by scope, then its distances.

    Every measure of an arm with no added rows is NaN. An arm with rows whose attack is undefined
    in a scope logs `privacy holdout=%d arm=%s scope=%s undefined members=%d nonmembers=%d
    nan_distances=%d`: at INFO for a cell scope with no member, as at a removal fraction of 1,
    else at WARNING.

    Raises:
        ValueError: As `privacy.distances_and_nndr`.
    """
    member_d = exposure(train, added, train=train, frame=frame)
    nonmember_d = exposure(test, added, train=train, frame=frame)
    members, nonmembers = (
        scopes(train, cell, edges=frame.edges),
        scopes(test, cell, edges=frame.edges),
    )
    found: dict[Measure, float] = {}
    for scope in SCOPES:
        ours, theirs = member_d[members[scope]], nonmember_d[nonmembers[scope]]
        read = attack(ours, theirs, fprs=SCOPE_FPRS[scope])
        found[mi("auc", scope)] = read.auc
        found |= {mi(RATES[rate], scope): value for rate, value in read.tpr.items()}
        if len(added) and math.isnan(read.auc):
            expected = scope == "cell" and not len(ours)
            logger.log(
                logging.INFO if expected else logging.WARNING,
                "privacy holdout=%d arm=%s scope=%s undefined members=%d nonmembers=%d"
                " nan_distances=%d",
                holdout,
                arm,
                scope,
                len(ours),
                len(theirs),
                int(np.isnan(ours).sum() + np.isnan(theirs).sum()),
            )
    if not len(added):
        return {**found, **dict.fromkeys(DISTANCE_MEASURES, math.nan)}
    return {**found, **distances_and_nndr(added, train, frame=frame)}


def holdout_scores(
    ready: Prepared,
    fills: Mapping[str, pd.DataFrame],
    *,
    cell: Stratum,
    frame: FrameSpec,
    extra: Mapping[str, Mapping[Measure, float]] = MappingProxyType({}),
) -> list[Score]:
    """One holdout's scores: the held-out rows' own, then each arm's, in `fills`' order.

    Each arm's `ADDED_MEASURES` are followed by its measures in `extra`, the match attack's and
    donor reuse's.

    It logs each arm's `privacy holdout=%d arm=%s added=%d` line, at WARNING for an arm that adds
    no row.

    Raises:
        EvaluationError: If a distance cannot be read, naming the holdout and the arm.
    """
    train, test, seed = ready.ablated.train, ready.ablated.test, ready.ablated.seed
    try:
        distances = distances_and_nndr(test, train, frame=frame)
    except ValueError as error:
        msg = f"holdout {seed}'s held-out rows: {error}"
        raise EvaluationError(msg) from error
    held = {Measure.HELD_OUT_RATIO: held_out_ratio(ready.sized, edges=frame.edges), **distances}
    scores = [
        Score(seed, HELD_OUT_ARM, Source.HELD_OUT, measure, Scope.ALL, value)
        for measure, value in held.items()
    ]
    for arm, added in fills.items():
        level = logging.INFO if len(added) else logging.WARNING
        logger.log(level, "privacy holdout=%d arm=%s added=%d", seed, arm, len(added))
        try:
            found = arm_measures(
                added, train=train, test=test, cell=cell, frame=frame, holdout=seed, arm=arm
            )
        except ValueError as error:
            msg = f"holdout {seed}'s arm {arm}: {error}"
            raise EvaluationError(msg) from error
        found |= extra.get(arm, {})
        scores += [
            Score(seed, arm, Source.ADDED, measure, Scope.ALL, value)
            for measure, value in found.items()
        ]
    return scores


def privacy(
    run: str,
    stores: Sequence[Path],
    *,
    experiment: Experiment,
    data: Data,
    runs: Mapping[str, NamedRun],
    root: Path,
    ontologies: Mapping[str, Ontology] | None = None,
) -> str:
    """Read `run`'s privacy from its `stores`; return the scores CSV.

    `ontologies` are the releases of the data's mapped concepts, which an arm of `DONOR_ARMS`
    re-draws its fill under.

    It logs `privacy run=%s stores=%d holdouts=%d reference=%s cell=%s root=%s`, `read`'s
    store lines, `privacy read stores=%d holdouts=%d`, then per holdout `match_measures`'
    `undefined` lines, `read_donors`' `privacy donor` line per arm of `DONOR_ARMS`,
    `holdout_scores`' lines and a `privacy holdout=%d done=%d/%d seconds=%.1f elapsed=%.1f`
    line (`privacy holdout=%d failed` at ERROR before a holdout's error); then each `Reading`'s
    line, then `privacy rows=%d sha256=%s readings=%d undefined=%d seconds=%.1f`, `undefined`
    counting the readings with no interval.

    Raises:
        EvaluationError: If the run names an arm of `DONOR_ARMS` and `ontologies` is None, or as
            `reference_of`, `holdout_scores` and `read_donors`.
        IdentityError: As `read` and `read_donors`, or if a holdout's `split` line is not the
            one its data prepares, or as `frame_of`, naming the store.
    """
    begun = time.perf_counter()
    donating = sorted(DONOR_ARMS & set(experiment.arms))
    if donating and ontologies is None:
        msg = f"the run {run} names the donor arms {','.join(donating)}, whose fills are"
        msg += " re-drawn under the releases of the data's mapped concepts, and no release"
        raise EvaluationError(msg)
    reference, cell = reference_of(run, experiment, runs)
    total = len(experiment.holdout_seeds)
    logger.info(
        "privacy run=%s stores=%d holdouts=%d reference=%s cell=%s root=%s",
        run,
        len(stores),
        total,
        reference,
        "/".join(str(value) for value in cell),
        root,
    )
    held = read(run, stores, data, experiment=experiment, runs=runs, root=root)
    logger.info("privacy read stores=%d holdouts=%d", len(stores), len(held))
    scores: list[Score] = []
    for done, seed in enumerate(experiment.holdout_seeds, start=1):
        started = time.perf_counter()
        try:
            scores += _holdout(
                seed, held[seed], data, experiment=experiment, cell=cell, ontologies=ontologies
            )
        except EvaluationError:
            logger.error("privacy holdout=%d failed", seed)  # ruff: ignore[error-instead-of-exception]
            raise
        now = time.perf_counter()
        logger.info(
            "privacy holdout=%d done=%d/%d seconds=%.1f elapsed=%.1f",
            seed,
            done,
            total,
            now - started,
            now - begun,
        )
    found = readings(scores, GROUPS, arms=experiment.arms, reference=reference)
    log_readings(found)
    text = render(scores)
    sha = hashlib.sha256(text.encode()).hexdigest()
    logger.info(
        "privacy rows=%d sha256=%s readings=%d undefined=%d seconds=%.1f",
        len(scores),
        sha,
        len(found),
        sum(r.leaks_more is None for r in found),
        time.perf_counter() - begun,
    )
    return text


def _holdout(
    seed: int,
    store: Store,
    data: Data,
    *,
    experiment: Experiment,
    cell: Stratum,
    ontologies: Mapping[str, Ontology] | None,
) -> list[Score]:
    """Holdout `seed`'s scores from `store`, once its split line matches the data's.

    Raises:
        EvaluationError: As `holdout_scores` and `read_donors`, or if no cell is flagged.
        IdentityError: If the split line differs, or as `frame_of`, naming the store, or as
            `read_donors`.
    """
    frame = data.frame
    ready = checked(seed, store, data, experiment)
    fills = read_fills(seed, store, experiment.arms, frame)
    train, groups = ready.ablated.train, match_groups(data.field_rows, frame=frame)
    request = None
    if DONOR_ARMS & set(fills):
        request = gap_request(train, experiment, seed=seed, frame=frame)
    extra: dict[str, dict[Measure, float]] = {}
    for arm, added in fills.items():
        found = match_measures(
            added,
            train=train,
            test=ready.ablated.test,
            cell=cell,
            frame=frame,
            groups=groups,
            holdout=seed,
            arm=arm,
        )
        donors = dict.fromkeys(DONOR_MEASURES, math.nan)
        if request is not None and arm in DONOR_ARMS:
            donors = read_donors(
                Fill(seed, arm, store.folder.arms[seed, arm]),
                added,
                train,
                request,
                lines=store.folder.lines.get((seed, arm, "gap"), ()),
                data=data,
                ontologies=ontologies or {},
                budget=experiment.budget,
                groups=groups,
            )
        extra[arm] = found | donors
    return holdout_scores(ready, fills, cell=cell, frame=frame, extra=extra)


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `privacy` subcommand to `commands`."""
    reading = commands.add_parser(
        "privacy", help="read a run's membership attack and distances from its stores"
    )
    reading.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    reading.add_argument(
        "--rows",
        dest="stores",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="a store `rows` wrote, one per slice of the run; repeatable",
    )
    reading.add_argument(
        "--data-dir", type=Path, default=None, help="the data root; default: the repository's"
    )
    reading.add_argument(
        "--runs-dir",
        type=runs_dir_argument,
        default=None,
        help="the root the stores' folders sit under; default: runs/ beside the code",
    )
