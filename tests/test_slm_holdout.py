"""`read_holdout` reads one arm's training set from a scored folder and its `rows` store.

The folder and the store are built in `tmp_path` by the harness's own `score` and `rows`, over
`UCI_E1`'s served archive, so no test reads `data/` or the network.
"""

from __future__ import annotations

import dataclasses
import functools
import io
from pathlib import Path

import pandas as pd
import pytest
import torch

from bko.evaluate.context import Context as EvaluateContext
from bko.evaluate.data import Dataset, held_split, load
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import REAL_ARM, Experiment
from bko.evaluate.named_run import NamedRun
from bko.evaluate.prepare import Prepared, prepared
from bko.evaluate.rows_read import frame_of, read_rows
from bko.evaluate.run_folder import read_folder
from bko.slm.__main__ import Context, Request, main, read_holdout
from tests._evaluate_fixtures import FAST_ARMS
from tests.test_evaluate_rows import REAL, stored
from tests.test_evaluate_shared import rescorer, run_folder


THREE_ARMS = dataclasses.replace(REAL, arms=("none", REAL_ARM, "smote"))
"""`REAL` with a baseline whose fill the store holds beside the `real` arm's."""

RUNNERS = {"none": FAST_ARMS["none"], "smote": FAST_ARMS["smote"]}


def _scored(tmp_path: Path, caplog, experiment: Experiment) -> tuple[Request, EvaluateContext]:
    """A folder scored as run `e1` = `experiment`, its store, and the context that reaches both."""
    folder = run_folder(tmp_path, "e1-0", caplog, experiment=experiment, arms=RUNNERS)
    code, _ = stored(tmp_path, folder, experiment=experiment, arms=RUNNERS)
    assert code == 0
    reach = rescorer(experiment=experiment, arms=RUNNERS)
    reach = dataclasses.replace(reach, runs={"e1": NamedRun(experiment)})
    seed = min(read_folder(folder).splits)
    return Request("e1", folder, tmp_path / "rows.jsonl", seed, None), reach


def _prepared(tmp_path: Path, request: Request, reach: EvaluateContext) -> Prepared:
    data = load(Dataset.UCI, tmp_path / "data", reach)
    return prepared(held_split(data, request.holdout), THREE_ARMS, edges=data.frame.edges)


@pytest.fixture
def scored(tmp_path: Path, caplog) -> tuple[Request, EvaluateContext]:
    return _scored(tmp_path, caplog, THREE_ARMS)


def _read(tmp_path: Path, request: Request, reach: EvaluateContext, arm: str | None):
    return read_holdout(dataclasses.replace(request, arm=arm), tmp_path / "data", reach=reach)


def test_none_trains_on_the_ablated_split_and_real_on_the_whole(tmp_path, scored) -> None:
    request, reach = scored
    ready = _prepared(tmp_path, request, reach)
    for arm in (None, "none"):
        held = _read(tmp_path, request, reach, arm)
        pd.testing.assert_frame_equal(held.train, ready.ablated.train)
        pd.testing.assert_frame_equal(held.test, ready.ablated.test)
    real = _read(tmp_path, request, reach, REAL_ARM)
    pd.testing.assert_frame_equal(real.train, ready.sized.train)
    assert len(real.train) > len(ready.ablated.train)
    assert real.store == read_rows(request.rows).sha256


def test_an_arm_trains_on_the_ablated_split_plus_its_stored_fill(tmp_path, scored) -> None:
    request, reach = scored
    ready = _prepared(tmp_path, request, reach)
    held = _read(tmp_path, request, reach, "smote")
    rows, folder = read_rows(request.rows), read_folder(request.folder)
    key = (request.holdout, "smote")
    fill = frame_of(rows.lines[key], arm_line=folder.arms[key], frame=held.frame)
    assert len(fill) > 0
    pd.testing.assert_frame_equal(held.train, pd.concat([ready.ablated.train, fill]))


def test_an_arm_the_store_lacks_is_an_identity_error(tmp_path, scored) -> None:
    request, reach = scored
    with pytest.raises(IdentityError, match="holds no fill of ctgan"):
        _read(tmp_path, request, reach, "ctgan")


def test_an_arm_the_store_lacks_exits_3(tmp_path, scored) -> None:
    request, reach = scored
    context = Context(
        device=lambda: torch.device("cpu"),
        holdout=functools.partial(read_holdout, reach=reach),
    )
    argv = ["--data-dir", str(tmp_path / "data"), "train", request.run]
    argv += ["--folder", str(request.folder), "--rows", str(request.rows)]
    argv += ["--holdout", str(request.holdout), "--arm", "ctgan"]
    assert main(argv, context=context, out=io.StringIO()) == 3


def test_a_run_that_decontaminates_is_refused(tmp_path, scored) -> None:
    request, reach = scored
    cleaned = dataclasses.replace(
        THREE_ARMS, arms=("none", "smote"), candidate="smote", decontaminate=True
    )
    reach = dataclasses.replace(reach, runs={"e1": NamedRun(cleaned)})
    with pytest.raises(EvaluationError, match="'e1' decontaminates") as caught:
        _read(tmp_path, request, reach, "none")
    assert type(caught.value) is EvaluationError


def test_a_run_with_no_name_is_refused(tmp_path, scored) -> None:
    request, reach = scored
    with pytest.raises(EvaluationError, match="no run named 'e9'"):
        _read(tmp_path, dataclasses.replace(request, run="e9"), reach, "none")
