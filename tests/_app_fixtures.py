"""The app over a small UCI archive and the ladder releases under `tmp_path`, through AppTest."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pandas as pd
from streamlit.testing.v1 import AppTest

from bko.app.context import Context
from bko.app.page import CONTEXT_KEY
from bko.evaluate.experiment import E1
from bko.evaluate.run import method_arms
from bko.generate.messages import Send
from tests._corpus import REPO_ROOT
from tests._datasets_fixtures import pin
from tests._evaluate_fixtures import FAST_METHODS
from tests._fusion_fixtures import young
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._generate_fixtures import SIZES, ScriptedSend, reply, uci
from tests._ontology_fixtures import RoutingOpener


ENTRY: Final = REPO_ROOT / "src" / "bko" / "app" / "streamlit_app.py"
TIMEOUT: Final = 60.0
"""Seconds an AppTest run may take under `pytest -n auto`."""
NOW: Final = datetime(2026, 11, 1, tzinfo=UTC)

ARCHIVE: Final = uci({(40, 0): "4.0"}, "1.0", {**SIZES, (40, 0, 1): 5})
"""`UCI_X` with five young women with the condition. The default split's gap request is then 12
rows of `("<55", "female", 0)` and 16 of `("<55", "female", 1)`, from 8 and 4 training records."""

PIN: Final = pin(ARCHIVE)

# Streamlit sets its own loggers' and uvicorn's levels on its first run in a process. One empty
# run while this module is imported, at collection, does that before any test's setup, so the
# root conftest's logging guard never charges it to whichever app test runs first in a worker.
AppTest.from_string("").run()

EXPERIMENT: Final = dataclasses.replace(
    E1, arms=("none", "smote", "bko"), comparators=("none", "smote"), replica_arms=()
)
"""E1 with one control, one baseline and the candidate."""


def one_fill() -> ScriptedSend:
    """The default split's two calls: 12 young women without the condition, then 16 with it."""
    return ScriptedSend(reply([young(0)] * 12), reply([young(1)] * 16))


def never() -> Send:
    """A send factory for a test that makes no call: building the send fails it."""
    msg = "this test builds no send"
    raise AssertionError(msg)


def ticking() -> Callable[[], datetime]:
    """A clock one second later on every read, so each fill gets its own run folder."""
    ticks = itertools.count()
    return lambda: NOW + timedelta(seconds=next(ticks))


def app_context(
    tmp_path: Path, *, send: Callable[[], Send], **changes: object
) -> tuple[Context, RoutingOpener]:
    """A context over `ARCHIVE` and the ladder releases with fast arms, and its opener."""
    pins, served = ladder_releases()
    opener = RoutingOpener({**served, PIN.url: ARCHIVE})
    built = Context(
        open_url=opener,
        uci=PIN,
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
        arms=method_arms(FAST_METHODS),
        experiment=EXPERIMENT,
        send=send,
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        now=ticking(),
        setup_logging=lambda: None,
    )
    return dataclasses.replace(built, **changes), opener


def settled(at: AppTest) -> AppTest:
    """`at`, after asserting its run raised nothing: AppTest renders an exception, never raises."""
    assert not at.exception, [element.value for element in at.exception]
    return at


def launch(context: Context) -> AppTest:
    """The real entry script's first run with `context` injected."""
    at = AppTest.from_file(str(ENTRY), default_timeout=TIMEOUT)
    at.session_state[CONTEXT_KEY] = context
    return settled(at.run())


def analyzed(context: Context) -> AppTest:
    """The page after Analyze gaps."""
    return settled(launch(context).button(key="bko.analyze").click().run())


def frame_with(at: AppTest, columns: set[str]) -> pd.DataFrame:
    """The first frame on the page holding every column in `columns`."""
    return next(frame.value for frame in at.dataframe if columns <= set(frame.value.columns))
