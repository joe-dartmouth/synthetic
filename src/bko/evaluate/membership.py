"""`python -m bko.evaluate privacy`: the membership attack and the privacy reading of a stored run.

The attack scores each record by minus its DCR-metric distance to the nearest of an arm's added
rows, every measured column's span read from the training split (`privacy.nearest_from`). Members
are the holdout's prepared training split; non-members are its whole test split, copied records
included; the removed records are neither. It reads three scopes: the records outside the removed
cell, the removed cell's (undefined with no member in it, as at a removal fraction of 1), and
label 1's outside the cell. Each scope reads the AUC (`roc_auc_score`, ties at their midrank) and
the true positive rate at each false positive rate of `FPRS`, by linear interpolation on
`roc_curve(drop_intermediate=False)`; the cell scope reads no rate at 0.001. Each arm also reads
the NNDR and DCR of its added rows to the training split, and the held-out rows the same as a
baseline (`privacy.distances_and_nndr`).

`privacy` reads one or more `rows` stores (`bko.evaluate.rows_read`), one per holdout slice of
the named run, makes no API call, and writes the scores CSV. It logs the reading rule over
`GROUPS` against the run family's `REFERENCE` arm (`bko.evaluate.privacy_rule`), which judges no
criterion. ADR `research/membership-inference-reading` records the attack and the rule.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from bko.baselines.request import Stratum
from bko.datasets.cells import LABEL, cell_keys
from bko.datasets.frames import FrameSpec
from bko.datasets.split import split
from bko.evaluate.data import Data
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import REAL_ARM, Experiment
from bko.evaluate.named_runs import DEFAULT_RUN, NamedRun, slice_of
from bko.evaluate.prepare import Prepared, held_out_ratio, prepared, split_line
from bko.evaluate.privacy import distances_and_nndr, nearest_from
from bko.evaluate.privacy_rule import Group, log_readings, readings
from bko.evaluate.rows_read import Rows, bind, frame_of, read_rows
from bko.evaluate.run_folder import Folder, read_folder
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, render


logger = logging.getLogger("bko.evaluate")

FPRS: Final = (0.01, 0.001)
"""The false positive rates each scope's true positive rate is read at."""

SCOPES: Final = ("outside", "cell", "label")
"""The attack's scopes: outside the removed cell, inside it, and label 1 outside it."""

REFERENCE: Final[Mapping[str, str]] = MappingProxyType(
    {"e6-thin": REAL_ARM, "e6-empty": REAL_ARM, "e6-curated": REAL_ARM}
)
"""Each run family's reference arm, which every other arm's privacy is read against."""

_RATES: Final = {0.01: "tpr01", 0.001: "tpr001"}


def mi(statistic: str, scope: str) -> Measure:
    """The attack's measure of `statistic` (`auc`, `tpr01` or `tpr001`) at `scope`."""
    return Measure(mi_name(statistic, scope))


def mi_name(statistic: str, scope: str) -> str:
    """The name `mi` gives the measure of `statistic` at `scope`, whether or not it is one."""
    return f"mi-{statistic}-{scope}"


MI_MEASURES: Final = tuple(
    mi(statistic, scope)
    for scope in SCOPES
    for statistic in ("auc", *_RATES.values())
    if (scope, statistic) != ("cell", "tpr001")
)
"""The attack's measures, by scope; the cell scope's few records define no rate at 0.001."""

_MI_NAMES: Final = frozenset(m.value for m in MI_MEASURES)

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
    Group(Scope.ALL.value, DISTANCE_MEASURES[:4]),
)
"""The rule's families: each scope's AUC and TPR at 0.01, and the DCR and NNDR quantiles, which
stay descriptive. A family's size is its arms with a defined contrast times its measures."""


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
    distance, every value is NaN.
    """
    distances = np.concatenate([member_d, nonmember_d])
    if not len(member_d) or not len(nonmember_d) or np.isnan(distances).any():
        return Attack(math.nan, dict.fromkeys(fprs, math.nan))
    truth = np.concatenate([np.ones(len(member_d)), np.zeros(len(nonmember_d))])
    fpr, tpr, _ = roc_curve(truth, -distances, drop_intermediate=False)
    auc = float(roc_auc_score(truth, -distances))
    return Attack(auc, {rate: float(np.interp(rate, fpr, tpr)) for rate in fprs})


def exposure(
    records: pd.DataFrame, added: pd.DataFrame, *, train: pd.DataFrame, frame: FrameSpec
) -> np.ndarray:
    """Each record's distance to its nearest added row on `train`'s spans; NaN with no rows."""
    if not len(added):
        return np.full(len(records), np.nan)
    return nearest_from(records, added, frame=frame, spans_from=train)


def scopes(records: pd.DataFrame, cell: Stratum, *, edges: Sequence[int]) -> dict[str, np.ndarray]:
    """Each scope's mask over `records`, by `SCOPES` name."""
    keys = cell_keys(records, edges=edges)
    inside = np.ones(len(records), dtype=bool)
    for column, value in zip(keys.columns, cell, strict=True):
        inside &= (keys[column] == value).to_numpy()
    labelled = (keys[LABEL] == 1).to_numpy()
    return {"outside": ~inside, "cell": inside, "label": ~inside & labelled}


def arm_measures(
    added: pd.DataFrame,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cell: Stratum,
    frame: FrameSpec,
) -> dict[Measure, float]:
    """One arm's `ADDED_MEASURES` on one holdout: its attack by scope, then its distances.

    Every measure of an arm with no added rows is NaN.
    """
    member_d = exposure(train, added, train=train, frame=frame)
    nonmember_d = exposure(test, added, train=train, frame=frame)
    members, nonmembers = (
        scopes(train, cell, edges=frame.edges),
        scopes(test, cell, edges=frame.edges),
    )
    found: dict[Measure, float] = {}
    for scope in SCOPES:
        read = attack(member_d[members[scope]], nonmember_d[nonmembers[scope]])
        named = {"auc": read.auc, **{_RATES[rate]: value for rate, value in read.tpr.items()}}
        for statistic, value in named.items():
            if mi_name(statistic, scope) in _MI_NAMES:
                found[mi(statistic, scope)] = value
    if not len(added):
        return {**found, **dict.fromkeys(DISTANCE_MEASURES, math.nan)}
    return {**found, **distances_and_nndr(added, train, frame=frame)}


def holdout_scores(
    ready: Prepared, fills: Mapping[str, pd.DataFrame], *, cell: Stratum, frame: FrameSpec
) -> list[Score]:
    """One holdout's scores: the held-out rows' own, then each arm's, in `fills`' order."""
    train, test, seed = ready.ablated.train, ready.ablated.test, ready.ablated.seed
    held = {
        Measure.HELD_OUT_RATIO: held_out_ratio(ready.sized, edges=frame.edges),
        **distances_and_nndr(test, train, frame=frame),
    }
    scores = [
        Score(seed, HELD_OUT_ARM, Source.HELD_OUT, measure, Scope.ALL, value)
        for measure, value in held.items()
    ]
    for arm, added in fills.items():
        found = arm_measures(added, train=train, test=test, cell=cell, frame=frame)
        scores += [
            Score(seed, arm, Source.ADDED, measure, Scope.ALL, value)
            for measure, value in found.items()
        ]
    return scores


@dataclass(frozen=True, slots=True)
class Store:
    """A store read back and bound to its folder."""

    path: Path
    rows: Rows
    folder: Folder


def read(
    run: str,
    stores: Sequence[Path],
    data: Data,
    *,
    experiment: Experiment,
    runs: Mapping[str, NamedRun],
    root: Path,
) -> dict[int, Store]:
    """Read and check each store; return the store that holds each of `run`'s holdouts.

    Each store's run is `run` or a slice of it (`slice_of`), its folder is `<root>/<folder>`,
    which it is bound to (`bind`: run, data digest, frame, folder, `log_sha256` and one fill line
    per `arm` line), and the stores' holdouts partition `run`'s. Each holdout's fills are the
    experiment's arms' and their replicas'.

    Raises:
        OSError: If a store or folder cannot be read.
        IdentityError: On any check above.
    """
    found: dict[int, Store] = {}
    for path in stores:
        rows = read_rows(path)
        name, folder_name = str(rows.header.get("run")), rows.header.get("folder")
        if name != run and slice_of(name, runs) != run:
            msg = f"the store {path} is run {name!r}'s, which is not a slice of {run!r}"
            raise IdentityError(msg)
        if not isinstance(folder_name, str) or Path(folder_name).name != folder_name:
            msg = f"the store {path} names the folder {folder_name!r}, not a folder name"
            raise IdentityError(msg)
        folder = read_folder(root / folder_name)
        bind(rows, folder, data.frame, run=name, data=data.digest)
        overlap = sorted(set(folder.splits) & set(found))
        if overlap:
            msg = f"the store {path}'s holdouts {overlap} are another store's too"
            raise IdentityError(msg)
        found |= dict.fromkeys(folder.splits, Store(path, rows, folder))
    if set(found) != set(experiment.holdout_seeds):
        msg = f"the stores' holdouts {sorted(found)} are not run {run!r}'s"
        msg += f" {sorted(experiment.holdout_seeds)}"
        raise IdentityError(msg)
    expected = {*experiment.arms, *(f"{arm}-replica" for arm in experiment.replica_arms)}
    for seed, store in found.items():
        held = {arm for holdout, arm in store.rows.lines if holdout == seed}
        if held != expected:
            msg = f"holdout {seed}'s fills in {store.path} are {sorted(held)}, not run {run!r}'s"
            msg += f" {sorted(expected)}"
            raise IdentityError(msg)
    return found


def reference_of(
    run: str, experiment: Experiment, runs: Mapping[str, NamedRun]
) -> tuple[str, Stratum]:
    """The reference arm of `run`'s family, and the cell its experiment removes.

    The family is `run`'s own `REFERENCE` entry, or else the entry of the run it slices.

    Raises:
        EvaluationError: If the family has no reference arm, the experiment lacks it, or the
            experiment removes no cell, which leaves no cell scope.
    """
    family = run if run in REFERENCE else slice_of(run, runs)
    reference = REFERENCE.get(family or run)
    if reference is None:
        msg = f"run {run!r} has no reference arm; the families read are {sorted(REFERENCE)}"
        raise EvaluationError(msg)
    if reference not in experiment.arms or experiment.removal is None:
        msg = f"run {run!r} needs its reference arm {reference!r} and a removed cell"
        raise EvaluationError(msg)
    return reference, experiment.removal.cell


def privacy(
    run: str,
    stores: Sequence[Path],
    *,
    experiment: Experiment,
    data: Data,
    runs: Mapping[str, NamedRun],
    root: Path,
) -> str:
    """Read `run`'s privacy from its `stores`; return the scores CSV.

    It logs each store's `privacy store=%s sha256=%s` line, a `privacy holdout=%d done=%d/%d
    seconds=%.1f` line per holdout, each `Reading`'s line, then `privacy rows=%d sha256=%s`.

    Raises:
        EvaluationError: As `reference_of`.
        IdentityError: As `read`, or if a holdout's `split` line is not the one its data
            prepares, or as `frame_of`.
    """
    reference, cell = reference_of(run, experiment, runs)
    for path in stores:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        logger.info("privacy store=%s sha256=%s", path, sha)
    held = read(run, stores, data, experiment=experiment, runs=runs, root=root)
    frame = data.frame
    scores: list[Score] = []
    total = len(experiment.holdout_seeds)
    for done, seed in enumerate(experiment.holdout_seeds, start=1):
        started, store = time.perf_counter(), held[seed]
        parts = split(data.records, seed=seed, edges=frame.edges)
        ready = prepared(parts, experiment, edges=frame.edges)
        if split_line(ready.ablated) != store.folder.splits[seed]:
            msg = f"holdout {seed}'s split {split_line(ready.ablated)!r} is not"
            msg += f" {store.folder.path}'s {store.folder.splits[seed]!r}"
            raise IdentityError(msg)
        fills = {
            arm: frame_of(
                store.rows.lines[seed, arm], arm_line=store.folder.arms[seed, arm], frame=frame
            )
            for arm in experiment.arms
        }
        scores += holdout_scores(ready, fills, cell=cell, frame=frame)
        elapsed = time.perf_counter() - started
        logger.info("privacy holdout=%d done=%d/%d seconds=%.1f", seed, done, total, elapsed)
    log_readings(readings(scores, GROUPS, arms=experiment.arms, reference=reference))
    text = render(scores)
    sha = hashlib.sha256(text.encode()).hexdigest()
    logger.info("privacy rows=%d sha256=%s", len(scores), sha)
    return text


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
