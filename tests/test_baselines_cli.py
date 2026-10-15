"""The `python -m bko.baselines` command: its training set, its streams, its exit codes."""

from __future__ import annotations

import io
import logging
import subprocess
import sys
import urllib.error

import pandas as pd
import pytest

from bko.baselines.__main__ import Context, configure_logging, main
from bko.baselines.errors import BaselineError
from bko.baselines.methods import METHODS
from bko.baselines.records import Baseline, no_rows, unit_weights
from bko.baselines.request import Method
from bko.datasets.split import split
from bko.datasets.uci import complete_records, parse_cleveland
from tests._baselines_fixtures import CLI_TEXT, EVALUATION_COLUMNS, UNBALANCED_TEXT
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import FakeOpener, archive, pin


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ARGS = {Method.NONE: [], Method.CLASS_WEIGHT: []}
"""Each built method's arguments in these tests."""

FAST = METHODS
"""The methods the in-process tests run."""

TRAIN_SPLIT = split(complete_records(parse_cleveland(CLI_TEXT))).train


def run(tmp_path, *argv, text=CLI_TEXT, methods=None, opener=None):
    data = archive(text)
    out = io.StringIO()
    context = Context(open_url=opener or FakeOpener(data), uci=pin(data), methods=methods or FAST)
    code = main([*argv, "--data-dir", str(tmp_path)], context=context, out=out)
    return code, out.getvalue()


def read(out: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(out), index_col="record_id")


@pytest.mark.parametrize("method", list(ARGS), ids=[str(method) for method in ARGS])
def test_each_method_writes_the_training_set(tmp_path, method):
    code, out = run(tmp_path, method.value, *ARGS[method])
    assert code == 0
    frame = read(out)
    assert list(frame.columns) == [*EVALUATION_COLUMNS, "weight"]
    real = frame.loc[frame.index > 0, EVALUATION_COLUMNS]
    pd.testing.assert_frame_equal(real, TRAIN_SPLIT.loc[:, EVALUATION_COLUMNS])
    assert int((frame.index < 0).sum()) == (5 if method.adds_rows else 0)


def test_class_weighting_writes_balanced_weights(tmp_path):
    code, out = run(tmp_path, "class-weight", text=UNBALANCED_TEXT)
    assert code == 0
    frame = read(out)
    assert frame.loc[frame["target"] == 0, "weight"].tolist() == pytest.approx([36 / 24] * 12)
    assert frame.loc[frame["target"] == 1, "weight"].tolist() == pytest.approx([36 / 48] * 24)


def test_stderr_carries_the_row_counts(tmp_path, caplog):
    n = len(TRAIN_SPLIT)
    with caplog.at_level(logging.INFO):
        run(tmp_path, "none")
    assert f"method=none seed=0 strata=cell real={n} synthetic=0 total={n}" in caplog.text
    assert f"split seed=0 train={n} test={8 * 8 - n}" in caplog.text


def test_a_control_takes_no_rows_flag():
    with pytest.raises(SystemExit) as excinfo:
        main(["none", "--rows", "5"])
    assert excinfo.value.code == 2


def test_a_method_not_built_is_not_offered():
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code == 2


def refuse(_train, _request):
    msg = "refused"
    raise BaselineError(msg)


def broken(train, _request):
    return Baseline(added=no_rows(), weights=unit_weights(train).iloc[1:], params={})


def test_an_infeasible_request_exits_two_and_writes_nothing(tmp_path):
    assert run(tmp_path, "none", methods={**FAST, Method.NONE: refuse}) == (2, "")


def test_a_broken_output_exits_three_and_writes_nothing(tmp_path):
    assert run(tmp_path, "none", methods={**FAST, Method.NONE: broken}) == (3, "")


def test_an_unreachable_archive_exits_two(tmp_path):
    opener = FakeOpener(error=urllib.error.URLError("down"))
    assert run(tmp_path, "none", opener=opener) == (2, "")


def test_an_unwritable_data_root_exits_two(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    assert run(blocker, "none")[0] == 2


def test_configure_logging_quiets_sdv():
    loggers = [logging.getLogger(name) for name in ("sdv", "SingleTableSynthesizer")]
    levels = [logger.level for logger in loggers]
    try:
        configure_logging()
        assert [logger.level for logger in loggers] == [logging.WARNING, logging.WARNING]
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


def test_the_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "bko.baselines", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert all(method.value in result.stdout for method in METHODS)
