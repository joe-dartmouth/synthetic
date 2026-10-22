"""The Streamlit page: pick the dataset, analyze its gaps, fill and score them, download the result.

The only module that imports streamlit. Each action runs with no streamlit call inside it and
writes its result into a `Slot` placed in the session before it starts, so a rerun requested
meanwhile neither aborts it nor loses its result (ADR `research/streamlit-app-design`).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import streamlit as st

from bko.app.context import Context
from bko.app.steps import Analysis, analyze
from bko.app.views import balance, cell_table, gap_table
from bko.baselines.errors import BaselineError
from bko.bkb.errors import BKBError
from bko.datasets.errors import DatasetError
from bko.evaluate.errors import EvaluationError
from bko.gaps.definitions import Definition
from bko.generate.errors import GenerationError


CONTEXT_KEY: Final = "bko.context"
"""The session key holding the `Context`; tests set it before the first run."""

ANALYSIS_KEY: Final = "bko.analysis"
OUTCOME_KEY: Final = "bko.outcome"
BUSY: Final = "Another action is running in this app; try again when it finishes."
STOPS: Final = (DatasetError, GenerationError, BaselineError, EvaluationError, BKBError, OSError)
"""What an action reports as a message, as the commands do; anything else is a defect."""

ACTION_LOCK: Final = threading.Lock()
"""One action per process: two sessions' fills would share numpy's and torch's global state, and
a refreshed tab could start a second paid fill beside a running one."""

DATASETS: Final = ("UCI Heart Disease",)


@dataclass(slots=True)
class Slot[T]:
    """Where an action leaves its result or its message.

    Plain attribute assignment sets both, and it never yields to streamlit.
    """

    value: T | None = None
    error: str | None = None


def page() -> None:
    """Render the page for this run of the script."""
    st.set_page_config(page_title="BKO gap filling", layout="wide")
    context: Context = st.session_state.setdefault(CONTEXT_KEY, Context())
    context.setup_logging()
    st.title("Fill clinical coverage gaps")
    st.header("1. Dataset")
    st.selectbox("Dataset", DATASETS, key="bko.dataset")
    _analysis(context)


def _run[T](key: str, label: str, action: Callable[[], T]) -> None:
    """Run `action` into a new slot under `key`, unless another action holds the process."""
    if not ACTION_LOCK.acquire(blocking=False):
        st.warning(BUSY)
        return
    try:
        slot: Slot[T] = Slot()
        st.session_state[key] = slot
        with st.spinner(label):
            try:
                slot.value = action()
            except STOPS as exc:
                slot.error = str(exc)
    finally:
        ACTION_LOCK.release()


def _shown[T](slot: Slot[T]) -> T | None:
    """`slot`'s value, after showing its message, or that the action did not finish."""
    if slot.error is not None:
        st.error("The action stopped.")
        st.code(slot.error, language=None)
        return None
    if slot.value is None:
        st.info("The last action did not finish; run it again.")
    return slot.value


def _analysis(context: Context) -> Analysis | None:
    st.header("2. Gaps")
    if st.button("Analyze gaps", type="primary", key="bko.analyze"):
        st.session_state.pop(OUTCOME_KEY, None)
        _run(
            ANALYSIS_KEY,
            "Loading UCI Heart Disease and the ontology releases; the first run downloads them",
            lambda: analyze(context),
        )
    slot: Slot[Analysis] | None = st.session_state.get(ANALYSIS_KEY)
    if slot is None:
        st.info("Analyze the gaps to continue.")
        return None
    analysis = _shown(slot)
    if analysis is not None:
        _show_analysis(analysis)
    return analysis


def _show_analysis(analysis: Analysis) -> None:
    records, flagged = st.columns(2)
    records.metric("Training records", len(analysis.parts.train))
    flagged.metric("Underrepresented cells", len(analysis.counted))
    st.dataframe(balance(analysis), hide_index=True)
    st.dataframe(cell_table(analysis), hide_index=True)
    tabs = st.tabs([definition.value for definition in Definition] + ["count-based"])
    for tab, definition in zip(tabs[:-1], Definition, strict=True):
        tab.dataframe(gap_table(analysis, definition), hide_index=True)
    tabs[-1].dataframe(cell_table(analysis).query("underrepresented"), hide_index=True)
