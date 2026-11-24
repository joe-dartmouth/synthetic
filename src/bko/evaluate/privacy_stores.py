"""The `rows` stores a privacy or added-rows reading reads, each read back, checked and bound.

`read` returns the store that holds each holdout of a run, refusing stores of another run, a
folder outside the runs directory, overlapping or missing holdouts, and fills that are not the
experiment's arms'.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bko.datasets.split import split
from bko.evaluate.data import Data
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import Experiment
from bko.evaluate.named_runs import NamedRun, wholes_of
from bko.evaluate.prepare import Prepared, prepared, split_line
from bko.evaluate.rows_read import Rows, bind, read_rows
from bko.evaluate.run_folder import Folder, read_folder


logger = logging.getLogger("bko.evaluate")


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
    verb: str = "privacy",
) -> dict[int, Store]:
    """Read and check each store; return the store that holds each of `run`'s holdouts.

    Each store's run is `run` or a slice of it (`wholes_of`), its folder is `<root>/<folder>`, a
    direct child of `root`, which it is bound to (`bind`: run, data digest, frame, folder,
    `log_sha256` and one fill line per `arm` line), and the stores' holdouts partition `run`'s.
    Each holdout's fills are the experiment's arms' and their replicas'. Each bound store logs
    `<verb> store=%s run=%s folder=%s log_sha256=%s holdouts=%s sha256=%s`, the sha256 of the
    bytes it was parsed from, `verb` the reading's: `privacy` or `added`. An error echoes at most
    `SHOWN` characters of a header value.

    Raises:
        OSError: If a store or folder cannot be read.
        IdentityError: On any check above.
    """
    found: dict[int, Store] = {}
    for path in stores:
        rows = read_rows(path)
        name, folder_name = str(rows.header.get("run")), rows.header.get("folder")
        if name != run and run not in wholes_of(name, runs):
            msg = f"the store {path} is run {_shown(name)}'s, which is not a slice of {run!r}"
            raise IdentityError(msg)
        if not isinstance(folder_name, str) or not _child(root, folder_name):
            msg = f"the store {path} names the folder {_shown(folder_name)}, not a folder"
            msg += f" directly under {root}"
            raise IdentityError(msg)
        folder = read_folder(root / folder_name)
        bind(rows, folder, data.frame, run=name, data=data.digest)
        logger.info(
            "%s store=%s run=%s folder=%s log_sha256=%s holdouts=%s sha256=%s",
            verb,
            path,
            name,
            folder_name,
            folder.log_sha256,
            ",".join(str(seed) for seed in sorted(folder.splits)),
            rows.sha256,
        )
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


def checked(seed: int, store: Store, data: Data, experiment: Experiment) -> Prepared:
    """Holdout `seed`'s prepared split, once its split line matches the one `store` is bound to.

    Raises:
        IdentityError: If the split line differs, naming the folder and the store.
    """
    edges = data.frame.edges
    ready = prepared(split(data.records, seed=seed, edges=edges), experiment, edges=edges)
    if split_line(ready.ablated) != store.folder.splits[seed]:
        msg = f"holdout {seed}'s split {split_line(ready.ablated)!r} is not"
        msg += f" {store.folder.path}'s {store.folder.splits[seed]!r}, which the store"
        msg += f" {store.path} is bound to"
        raise IdentityError(msg)
    return ready


SHOWN: Final = 200
"""The most characters of a store's header value an error echoes."""


def _shown(value: object) -> str:
    """`value`'s repr, cut to `SHOWN` characters."""
    text = repr(value)
    return text if len(text) <= SHOWN else f"{text[:SHOWN]}..."


def _child(root: Path, name: str) -> bool:
    """Whether the folder name `name` resolves to a direct child of `root`."""
    if name in {"", ".", ".."}:
        return False
    return (root / name).resolve().parent == root.resolve()
