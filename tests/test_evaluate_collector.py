"""`rescore` checks the `split` and `arm` lines the harness puts in a `Logged`, not its logger."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from bko.baselines.records import Baseline
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.data import load
from bko.evaluate.errors import IdentityError
from bko.evaluate.filling import Logged, log_arm
from bko.evaluate.run import score_holdout
from bko.evaluate.run_folder import read_folder
from bko.evaluate.shared import rescore
from tests._evaluate_fixtures import FAST_ARMS, GRADED
from tests.test_evaluate_shared import PAIR, edit_log, rescorer, run_folder, shared


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

SPLIT_LINE = (
    "split seed=0 train=146 test=35"
    " fingerprint=e3d80ad32465506fda2b3936ed7df7bbbafbe537a779c5e91b5ab007a7319266"
)
"""`PAIR`'s holdout 0 `split` line on `GRADED`."""

ARM_LINES = (
    (
        "arm holdout=0 arm=none attempt=1 seed=0 added=0"
        " sha256=eea72aa554ac6c4302c79a1ade2258a0ac787b9d12233bbe64ef33c6e7e402e0 params={}"
    ),
    (
        "arm holdout=0 arm=smote attempt=1 seed=0 added=20"
        " sha256=f7065bccb6fc97a059de46c234a162b2ac484e5ca8aeba6e2f6bf3ec22a6420b"
        """ params={"neighbors": {"('<55', 'female', 1)": 1, "('>=55', 'female', 1)": 5}}"""
    ),
)
"""`PAIR`'s holdout 0 `arm` lines on `GRADED`, in fill order."""


def test_two_rescores_in_one_process_both_pass(tmp_path, caplog):
    first = run_folder(tmp_path, "first", caplog)
    second = run_folder(tmp_path, "second", caplog, replicas=True)
    assert shared(tmp_path, first) == (0, (first / "scores.csv").read_text(encoding="utf-8"))
    code, text = shared(tmp_path, second, replicas=True)
    assert (code, text) == (0, (second / "scores.csv").read_text(encoding="utf-8"))


def test_a_disabled_harness_logger_still_passes(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    harness = logging.getLogger("bko.evaluate.run")
    harness.disabled = True
    try:
        code, text = shared(tmp_path, folder)
    finally:
        harness.disabled = False
    assert (code, text) == (0, (folder / "scores.csv").read_text(encoding="utf-8"))


def test_the_harness_still_logs_its_arm_and_split_lines(caplog):
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    logged = Logged()
    with caplog.at_level(logging.INFO, logger="bko.evaluate.run"):
        score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME, logged=logged)
    messages = [r.getMessage() for r in caplog.records if r.name == "bko.evaluate.run"]
    assert [m for m in messages if m.startswith("split ")] == [SPLIT_LINE]
    assert tuple(m for m in messages if m.startswith("arm ")) == ARM_LINES
    assert logged.split == SPLIT_LINE
    assert tuple(fill.line for fill in logged.fills) == ARM_LINES
    assert [(fill.name, fill.kind, fill.attempt) for fill in logged.fills] == [
        ("none", "gap", 1),
        ("smote", "gap", 1),
    ]


def test_a_reused_collector_holds_only_its_last_holdouts_fills():
    logged = Logged()
    for seed in (1, 0):
        parts = split(GRADED, seed=seed, edges=UCI_FRAME.edges)
        score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME, logged=logged)
    assert logged.split == SPLIT_LINE
    assert tuple(fill.line for fill in logged.fills) == ARM_LINES


def test_log_arm_returns_the_line_it_logs(caplog):
    added = pd.DataFrame({"age": [61]})
    baseline = Baseline(added=added, weights=pd.Series([1.0]), params={"k": 3})
    with caplog.at_level(logging.INFO, logger="bko.evaluate.run"):
        line = log_arm(baseline, holdout=2, name="smote", attempt=1, seed=7)
    assert [r.getMessage() for r in caplog.records] == [line]
    assert line.startswith("arm holdout=2 arm=smote attempt=1 seed=7 added=1 sha256=")
    assert line.endswith(' params={"k": 3}')


def test_after_is_called_once_per_holdout_after_its_checks(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    ctx = rescorer()
    data = load(ctx.dataset, tmp_path / "data", ctx)
    calls: list[tuple[int, str | None, int, set[int]]] = []

    def after(holdout, logged, scored):
        calls.append((holdout, logged.split, len(logged.fills), {s.holdout for s in scored}))

    def rescored() -> str:
        return rescore(
            read_folder(folder),
            [],
            run="e1",
            experiment=ctx.experiment,
            data=data,
            arms=ctx.arms,
            models=ctx.models,
            uci=True,
            after=after,
        )

    assert rescored() == (folder / "scores.csv").read_text(encoding="utf-8")
    splits = read_folder(folder).splits
    assert calls == [(seed, splits[seed], 2, {seed}) for seed in (0, 1)]
    calls.clear()
    edit_log(folder, "arm holdout=1 arm=bko ", lambda line: line.replace("sha256=", "sha256=0"))
    with pytest.raises(IdentityError, match="holdout 1 arm bko"):
        rescored()
    assert [call[0] for call in calls] == [0]
