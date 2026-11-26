"""The removed-truth reading: each arm's rows in the removed cell against the removed records."""

from __future__ import annotations

import dataclasses
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
from tests.test_evaluate_shared import edit_log


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


def test_two_rows_on_each_side_are_scored():
    rows = random_records(UCI_FRAME, 4, seed=3)
    found = truth_scores(rows.iloc[:2], rows.iloc[2:], frame=UCI_FRAME)
    assert 0 <= found.shapes <= 1
    assert 0 <= found.pair_trends <= 1
    assert all(0 <= value <= 1 for value in found.columns.values())


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


def _messages(caplog, prefix: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith(prefix)]


def test_real_is_left_out(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    code, text = _truth(tmp_path, store)
    assert code == 0
    table = pd.read_csv(io.StringIO(text), keep_default_na=False)
    assert set(table["arm"]) == {"none", "smote"}
    logged = [r.getMessage() for r in caplog.records]
    assert not any(f"arm={REAL_ARM} " in line for line in logged if line.startswith("truth "))


def test_the_csv_and_its_log_follow_the_contract(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    code, text = _truth(tmp_path, store)
    assert code == 0
    table = pd.read_csv(io.StringIO(text), keep_default_na=False)
    assert tuple(table.columns) == TRUTH_COLUMNS
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
    header = (
        f"truth run=e1 stores=1 holdouts=2 cell=<55/female/1 fraction=0.5 min_rows=2 "
        f"root={tmp_path}"
    )
    assert header in logged
    assert f"truth store={store} run=e1 folder=run log_sha256=" in caplog.text
    assert re.search(r"truth holdout=0 removed=\d+$", caplog.text, re.MULTILINE)
    assert re.search(r"truth holdout=1 done=2/2 seconds=\S+ elapsed=\S+", caplog.text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert f"truth lines={len(table)} sha256={sha} seconds=" in logged[-1]


def test_an_arm_with_too_few_added_rows_logs_at_info_with_its_reason(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    assert _truth(tmp_path, store)[0] == 0
    (empty,) = _messages(caplog, "truth holdout=0 arm=none ")
    assert empty.levelno == logging.INFO
    assert empty.getMessage().endswith(" short=added")
    (scored,) = _messages(caplog, "truth holdout=0 arm=smote ")
    assert scored.levelno == logging.INFO
    assert "short=" not in scored.getMessage()


def test_too_few_removed_records_log_at_info_with_their_reason(tmp_path, caplog, monkeypatch):
    _, store = _stored(tmp_path, caplog)
    real = truth_module.checked

    def one_removed(*args, **kwargs):
        found = real(*args, **kwargs)
        return dataclasses.replace(found, removed=found.removed.iloc[:1])

    monkeypatch.setattr(truth_module, "checked", one_removed)
    assert _truth(tmp_path, store)[0] == 0
    (scored,) = _messages(caplog, "truth holdout=0 arm=smote ")
    assert scored.levelno == logging.INFO
    assert scored.getMessage().endswith(" short=removed")
    (empty,) = _messages(caplog, "truth holdout=0 arm=none ")
    assert empty.getMessage().endswith(" short=added,removed")


def test_a_nan_score_with_enough_rows_logs_a_warning(tmp_path, caplog, monkeypatch):
    _, store = _stored(tmp_path, caplog)
    real = truth_module.truth_scores

    def pairless(added: pd.DataFrame, removed: pd.DataFrame, *, frame: FrameSpec) -> ReportScores:
        found = real(added, removed, frame=frame)
        return dataclasses.replace(found, pair_trends=math.nan)

    monkeypatch.setattr(truth_module, "truth_scores", pairless)
    assert _truth(tmp_path, store)[0] == 0
    (scored,) = _messages(caplog, "truth holdout=0 arm=smote ")
    assert scored.levelno == logging.WARNING
    assert "short=" not in scored.getMessage()
    (empty,) = _messages(caplog, "truth holdout=0 arm=none ")
    assert empty.levelno == logging.INFO


def test_the_run_is_required(tmp_path, capsys):
    argv = ["truth", "--rows", str(tmp_path / "run.jsonl"), "--data-dir", str(tmp_path / "data")]
    with pytest.raises(SystemExit) as exited:
        main(argv, context=_context(tmp_path), out=io.StringIO())
    assert exited.value.code == 2
    assert "run" in capsys.readouterr().err


def test_column_shapes_is_the_mean_of_its_column_shapes(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    code, text = _truth(tmp_path, store)
    assert code == 0
    table = pd.read_csv(io.StringIO(text))
    scored = table[table["arm"] == "smote"]
    assert len(scored)
    for _, rows in scored.groupby("holdout"):
        (shapes,) = rows.loc[rows["measure"] == "column-shapes", "value"]
        assert shapes == pytest.approx(rows.loc[rows["measure"] == "column-shape", "value"].mean())


def test_rows_counts_the_added_rows_in_the_removed_cell(tmp_path, caplog, monkeypatch):
    _, store = _stored(tmp_path, caplog)
    scored: list[int] = []
    filled: list[int] = []
    real_scores, real_fills = truth_module.truth_scores, truth_module.read_fills

    def spying(added: pd.DataFrame, removed: pd.DataFrame, *, frame: FrameSpec) -> ReportScores:
        scored.append(len(added))
        return real_scores(added, removed, frame=frame)

    def counting(*args, **kwargs) -> dict[str, pd.DataFrame]:
        fills = real_fills(*args, **kwargs)
        filled.extend(len(rows) for rows in fills.values())
        return fills

    monkeypatch.setattr(truth_module, "truth_scores", spying)
    monkeypatch.setattr(truth_module, "read_fills", counting)
    code, text = _truth(tmp_path, store)
    assert code == 0
    table = pd.read_csv(io.StringIO(text))
    counts = table.groupby(["holdout", "arm"], sort=False)["rows"].first()
    assert list(counts) == scored
    assert any(inside < added for inside, added in zip(scored, filled, strict=True))


def test_a_split_line_the_data_does_not_reproduce_exits_3(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog)
    edit_log(folder, "split seed=1", lambda line: re.sub(r"fingerprint=\w+", "fingerprint=0", line))
    log = hashlib.sha256((folder / "scores.log").read_bytes()).hexdigest()
    _rewrite(store, _header("log_sha256", log))
    assert _truth(tmp_path, store) == (3, "")
    assert f"which the store {store} is bound to" in caplog.text
