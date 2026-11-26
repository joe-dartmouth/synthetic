"""The removed-truth reading: each arm's rows in the removed cell against the removed records."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.cells import cell_keys
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import truth as truth_module
from bko.evaluate.__main__ import main
from bko.evaluate.data import held_split, load
from bko.evaluate.experiment import REAL_ARM
from bko.evaluate.fidelity import ReportScores
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.prepare import prepared
from bko.evaluate.truth import TRUTH_COLUMNS, truth_scores
from tests._evaluate_bko_fixtures import NARROW
from tests.test_evaluate_membership import PRIVATE, _context, _header, _rewrite, _stored
from tests.test_evaluate_privacy import random_records


def _truth(
    tmp_path: Path, *stores: Path, run: str = "e1", runs: dict | None = None
) -> tuple[int, str]:
    argv = ["truth", run, "--data-dir", str(tmp_path / "data")]
    for store in stores:
        argv += ["--rows", str(store)]
    out = io.StringIO()
    code = main(argv, context=_context(tmp_path, runs=runs), out=out)
    return code, out.getvalue()


def test_identical_rows_score_one():
    rows = random_records(UCI_FRAME, 20, seed=0)
    found = truth_scores(rows, rows, frame=UCI_FRAME)
    assert found.shapes == pytest.approx(1.0)
    assert found.pair_trends == pytest.approx(1.0)
    assert list(found.columns) == list(UCI_FRAME.columns)
    assert list(found.columns.values()) == pytest.approx([1.0] * len(UCI_FRAME.columns))


def test_a_shifted_column_scores_below_one_on_that_column():
    removed = random_records(UCI_FRAME, 10, seed=1).assign(fbs=[0] * 6 + [1] * 4)
    added = removed.assign(fbs=[0] * 2 + [1] * 8)
    found = truth_scores(added, removed, frame=UCI_FRAME)
    # Shares 0.6/0.4 against 0.2/0.8: total variation 0.5 * (0.4 + 0.4) = 0.4.
    assert found.columns["fbs"] == pytest.approx(1 - 0.4)
    others = [value for name, value in found.columns.items() if name != "fbs"]
    assert others == pytest.approx([1.0] * len(others))
    assert found.shapes < 1.0


def test_fewer_than_two_rows_read_nan():
    rows = random_records(UCI_FRAME, 5, seed=2)
    for added, removed in ((rows.iloc[:1], rows), (rows, rows.iloc[:1]), (rows.iloc[:0], rows)):
        found = truth_scores(added, removed, frame=UCI_FRAME)
        assert math.isnan(found.shapes)
        assert math.isnan(found.pair_trends)
        assert all(math.isnan(value) for value in found.columns.values())


def test_the_dropped_records_are_the_removals(tmp_path, caplog, monkeypatch):
    _, store = _stored(tmp_path, caplog)
    seen: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    real = truth_module.truth_scores

    def spying(added: pd.DataFrame, removed: pd.DataFrame, *, frame: FrameSpec) -> ReportScores:
        seen.append((added, removed))
        return real(added, removed, frame=frame)

    monkeypatch.setattr(truth_module, "truth_scores", spying)
    assert _truth(tmp_path, store)[0] == 0
    ctx = _context(tmp_path)
    data = load(ctx.dataset, tmp_path / "data", ctx)
    arms = [arm for arm in PRIVATE.arms if arm != REAL_ARM]
    assert len(seen) == len(PRIVATE.holdout_seeds) * len(arms)
    cell = PRIVATE.removal.cell if PRIVATE.removal else None
    for index, seed in enumerate(PRIVATE.holdout_seeds):
        removals = prepared(held_split(data, seed), PRIVATE, edges=UCI_FRAME.edges).removed
        assert len(removals)
        for added, removed in seen[index * len(arms) : (index + 1) * len(arms)]:
            pd.testing.assert_frame_equal(removed, removals)
            keys = cell_keys(added, edges=UCI_FRAME.edges)
            assert all(tuple(key) == cell for key in keys.itertuples(index=False))


def test_a_run_with_no_removal_is_refused(tmp_path, caplog):
    ctx = _context(tmp_path)
    runs = {**ctx.runs, "plain": NamedRun(NARROW, ctx.settings, ctx.dataset)}
    assert _truth(tmp_path, tmp_path / "absent.jsonl", run="plain", runs=runs) == (2, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "removes no cell" in errors[-1]


def test_a_store_of_another_run_is_refused(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, _header("run", "b1-uci"))
    assert _truth(tmp_path, store) == (3, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "not a slice of" in errors[-1]


def test_real_is_left_out(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    caplog.set_level(logging.INFO)
    code, text = _truth(tmp_path, store)
    assert code == 0
    table = pd.read_csv(io.StringIO(text), keep_default_na=False)
    assert tuple(table.columns) == TRUTH_COLUMNS
    assert set(table["arm"]) == {"none", "smote"}
    measures = ["column-shapes", "column-pair-trends", *["column-shape"] * len(UCI_FRAME.columns)]
    columns = ["", "", *UCI_FRAME.columns]
    for (seed, arm), rows in table.groupby(["holdout", "arm"], sort=False):
        assert list(rows["measure"]) == measures
        assert list(rows["column"]) == columns
        if arm == "none":
            assert set(rows["rows"]) == {0}
            assert set(rows["value"]) == {""}
        else:
            assert rows["rows"].iloc[0] > 1
            assert all(0 <= float(value) <= 1 for value in rows["value"]), seed
    logged = [r.getMessage() for r in caplog.records]
    assert logged[0].startswith("versions commit=")
    header = f"truth run=e1 stores=1 holdouts=2 cell=<55/female/1 fraction=0.5 root={tmp_path}"
    assert header in logged
    assert f"truth store={store} run=e1 folder=run log_sha256=" in caplog.text
    assert not any(f"arm={REAL_ARM} " in line for line in logged if line.startswith("truth "))
    assert re.search(r"truth holdout=0 removed=\d+$", caplog.text, re.MULTILINE)
    (empty,) = [r for r in caplog.records if r.getMessage().startswith("truth holdout=0 arm=none")]
    assert empty.levelno == logging.WARNING
    assert re.search(r"truth holdout=1 done=2/2 seconds=\S+ elapsed=\S+", caplog.text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert f"truth rows={len(table)} sha256={sha} seconds=" in logged[-1]
