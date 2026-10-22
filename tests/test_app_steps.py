"""`bko.app.steps`: each action over fakes; no test reaches the API or the network."""

from __future__ import annotations

import logging
import sys

from bko.app.context import Context, configure_logging
from bko.app.runs import runs_root
from bko.app.steps import analyze
from bko.baselines.synthesizers import QUIET_LOGGERS
from bko.datasets.cells import cell_counts
from bko.datasets.fetch import default_data_dir
from bko.gaps.definitions import Definition
from bko.generate.api import HTTP_LOGGERS
from tests._app_fixtures import app_context, never
from tests._guided_fixtures import YOUNG_SICK, YOUNG_WELL


def test_analyze_lists_the_flagged_cells_and_each_definitions_gaps(tmp_path):
    context, _ = app_context(tmp_path, send=never)
    analysis = analyze(context)
    assert {gap.cell for gap in analysis.counted} == {YOUNG_WELL, YOUNG_SICK}
    assert set(analysis.found) == set(Definition)
    assert analysis.counts.equals(cell_counts(analysis.parts.train))
    assert analysis.parts.seed == 0


def test_analyze_fetches_nothing_once_the_data_is_present(tmp_path):
    context, opener = app_context(tmp_path, send=never)
    analyze(context)
    fetched = list(opener.calls)
    analyze(context)
    assert opener.calls == fetched


def test_the_default_roots_sit_at_the_checkout():
    context = Context()
    assert context.data_root() == default_data_dir()
    assert context.runs_root() == runs_root() == default_data_dir().parent / "runs"
    assert context.now().tzinfo is not None


def test_configure_logging_quiets_the_http_and_sdv_loggers(monkeypatch):
    """`captureWarnings` is stubbed: the real call is process-wide and outlives the test."""
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: recorded.append(kwargs))
    monkeypatch.setattr(logging, "captureWarnings", lambda _: None)
    loggers = [logging.getLogger(name) for name in (*QUIET_LOGGERS, *HTTP_LOGGERS)]
    for logger in loggers:
        monkeypatch.setattr(logger, "level", logger.level)
    configure_logging()
    [kwargs] = recorded
    assert kwargs["stream"] is sys.stderr
    assert kwargs["level"] == logging.INFO
    assert [logger.level for logger in loggers] == [logging.WARNING] * len(loggers)
