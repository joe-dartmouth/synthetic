"""The Streamlit page: pick the dataset, analyze its gaps, fill and score them, download the result.

With `browse.py`, the only modules that import streamlit. Each action runs with no streamlit call
inside it and writes its result into a `Slot` placed in the session before it starts, so a rerun
requested meanwhile neither aborts it nor loses its result (ADR `research/streamlit-app-design`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import streamlit as st

from bko.app.browse import earlier_runs
from bko.app.context import Context
from bko.app.steps import (
    Analysis,
    Outcome,
    analyze,
    call_logs,
    fill_and_score,
    fillers,
    is_paid,
    scores_csv,
    training_csv,
)
from bko.app.views import (
    balance,
    cell_table,
    closure_pairs,
    closure_summary,
    gap_table,
    prediction_column,
    prediction_table,
    score_table,
)
from bko.baselines.errors import BaselineError
from bko.bkb.errors import BKBError
from bko.datasets.errors import DatasetError
from bko.evaluate.errors import EvaluationError
from bko.evaluate.scores import Measure, Scope
from bko.gaps.definitions import Definition
from bko.generate.errors import GenerationError


logger = logging.getLogger(__name__)

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

UNFINISHED: Final = (
    "The last action did not finish; its traceback is in the terminal running the app. A BKO "
    "fill's call log stays in its run folder, so a replay answers it again without paying."
)

RUNS_KEY: Final = "bko.runs"
"""How many fills this session started; each confirmation box is keyed by it, so none carries."""

DATASETS: Final = ("UCI Heart Disease",)

INTRO: Final = (
    "Clinical models fail on groups with too few training records. This page finds where a "
    "dataset is thin, using both record counts and medical knowledge from the MONDO and HPO "
    "ontologies, fills those groups with synthetic records, and shows whether a heart-disease "
    "classifier trained with them predicts better than one trained with the usual fixes."
)
"""The page's opening paragraph, for a reader who has not seen the project."""

STEPS: Final = {
    "gaps": (
        "A cell is one age band, sex and diagnosis. Below the counts, each tab but the last lists "
        "the statements the ontology implies for a cell that the training records do not "
        "support, under one of the project's gap definitions; the last lists the cells that are "
        "thin by count alone."
    ),
    "fill": (
        "Pick how to add records to the underrepresented cells. The BKO arms ask Claude for "
        "records guided by the ontology; the others are standard oversamplers and generators. "
        "Every method is then trained and scored on the same held-out records."
    ),
}
"""The caption under each step's header."""

RECALL_COLUMNS: Final = tuple(prediction_column(Measure.RECALL, scope) for scope in Scope)
"""The `prediction_table` columns the results chart draws: recall, overall and on the flagged
subgroups, the plan's primary measure."""


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
    st.markdown(INTRO)
    view = st.radio("View", ("Demo", "Earlier runs"), horizontal=True, key="bko.view")
    if view == "Earlier runs":
        earlier_runs(context)
        return
    st.header("1. Dataset")
    st.selectbox("Dataset", DATASETS, key="bko.dataset")
    analysis = _analysis(context)
    if analysis is None:
        return
    outcome = _fill(context, analysis)
    if outcome is not None:
        _results(analysis, outcome)


def _nothing() -> None:
    return None


def _run[T](
    key: str, label: str, action: Callable[[], T], *, started: Callable[[], None] = _nothing
) -> bool:
    """Run `action` into a new slot under `key`; False if another action holds the process.

    `started` runs once the process is held, before `action`, so it runs even if `action` raises.
    """
    if not ACTION_LOCK.acquire(blocking=False):
        st.warning(BUSY)
        return False
    try:
        started()
        slot: Slot[T] = Slot()
        st.session_state[key] = slot
        with st.spinner(label):
            try:
                slot.value = action()
            except STOPS as exc:
                logger.warning("stopped action=%s error=%s: %s", key, type(exc).__name__, exc)
                slot.error = "\n".join(
                    [f"{type(exc).__name__}: {exc}", *getattr(exc, "__notes__", ())]
                )
    finally:
        ACTION_LOCK.release()
    return True


def _shown[T](slot: Slot[T]) -> T | None:
    """`slot`'s value, after showing its message, or that the action did not finish."""
    if slot.error is not None:
        st.error("The action stopped.")
        st.code(slot.error, language=None)
        return None
    if slot.value is None:
        st.info(UNFINISHED)
    return slot.value


def _analysis(context: Context) -> Analysis | None:
    st.header("2. Gaps")
    st.caption(STEPS["gaps"])
    clicked = st.button("Analyze gaps", type="primary", key="bko.analyze")
    if clicked and _run(
        ANALYSIS_KEY,
        "Loading UCI Heart Disease and the ontology releases; the first run downloads them",
        lambda: analyze(context),
    ):
        st.session_state.pop(OUTCOME_KEY, None)
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
    # `status` says the same in words, so colour and a tick never carry it alone.
    st.dataframe(cell_table(analysis), hide_index=True, column_config={"underrepresented": None})
    tabs = st.tabs([definition.value for definition in Definition] + ["count-based"])
    for tab, definition in zip(tabs[:-1], Definition, strict=True):
        tab.dataframe(gap_table(analysis, definition), hide_index=True)
    tabs[-1].dataframe(cell_table(analysis).query("underrepresented"), hide_index=True)


def _forget_outcome() -> None:
    st.session_state.pop(OUTCOME_KEY, None)


def _count_fill() -> None:
    """Move to the next confirmation box, so a ticked one never carries to a second fill."""
    st.session_state[RUNS_KEY] = st.session_state.get(RUNS_KEY, 0) + 1


def _fill(context: Context, analysis: Analysis) -> Outcome | None:
    st.header("3. Fill and score")
    st.caption(STEPS["fill"])
    name: str = st.selectbox(
        "Method",
        fillers(context.experiment),
        key="bko.method",
        on_change=_forget_outcome,
        format_func=lambda arm: f"{arm} (Anthropic API)" if is_paid(arm) else arm,
    )
    replay, ready = _answers(context, name) if is_paid(name) else (None, True)
    clicked = st.button("Fill and score", type="primary", key="bko.fill", disabled=not ready)
    if (
        clicked
        and ready
        and _run(
            OUTCOME_KEY,
            f"Filling with {name}, then scoring every method on the split",
            lambda: fill_and_score(analysis, name, context, replay=replay),
            started=_count_fill,
        )
    ):
        # The finished fill sits in its slot; the rerun draws the next fill's unticked box.
        st.rerun()
    slot: Slot[Outcome] | None = st.session_state.get(OUTCOME_KEY)
    if slot is None or (slot.value is not None and slot.value.name != name):
        # A view switch resets the method box but keeps the outcome; its run folder keeps it too.
        return None
    return _shown(slot)


def _answers(context: Context, name: str) -> tuple[Path | None, bool]:
    """Where a BKO fill's answers come from, and whether the fill may start."""
    mode = st.radio(
        "Answers", ("live", "replay"), horizontal=True, key="bko.answers", on_change=_forget_outcome
    )
    if mode == "replay":
        logs = {path.parent.name: path for path in call_logs(context, name)}
        if not logs:
            st.info("No call log to replay yet.")
            return None, False
        chosen: str = st.selectbox(
            "Call log of an earlier fill", tuple(logs), key="bko.log", on_change=_forget_outcome
        )
        return logs[chosen], True
    runs = st.session_state.get(RUNS_KEY, 0)
    confirmed = st.checkbox(
        f"Pay for Anthropic API calls on this fill; it stops calling once its spend reaches "
        f"{context.ceiling} USD",
        key=f"bko.confirm.{name}.{runs}",
    )
    return None, confirmed


def _results(analysis: Analysis, outcome: Outcome) -> None:
    st.header("4. Results")
    st.caption(
        "One split, E1's first holdout. These are measures, not E1's verdict, which "
        "docs/EVIDENCE.md holds."
    )
    added, calls, spend = st.columns(3)
    added.metric("Rows added", len(outcome.baseline.added))
    if outcome.log is not None:
        calls.metric("API calls", outcome.calls)
        spend.metric("Spend (USD)", "0, replayed" if outcome.replayed else f"{outcome.usd:.6f}")
    st.code(str(outcome.folder), language=None)
    st.subheader("Gaps before and after the fill")
    st.dataframe(closure_summary(outcome), hide_index=True)
    st.dataframe(closure_pairs(analysis, outcome), hide_index=True)
    st.subheader("Prediction on the held-out records")
    st.caption(
        "Each method's classifier, trained on the real training records plus the records it "
        "added; the flagged subgroups are the test records of every age-sex group holding an "
        "underrepresented cell. A blank is a measure this split cannot define."
    )
    prediction = prediction_table(outcome.scores, outcome.name)
    st.bar_chart(
        prediction,
        x="arm",
        y=list(RECALL_COLUMNS),
        x_label="",
        y_label="recall",
        horizontal=True,
        sort=False,
        stack=False,
    )
    st.dataframe(prediction, hide_index=True)
    with st.expander("Every measure"):
        st.dataframe(score_table(outcome.scores))
    st.subheader("Download")
    st.download_button(
        "Augmented training set (CSV)",
        training_csv(analysis, outcome),
        file_name=f"{outcome.folder.name}-training.csv",
        mime="text/csv",
        on_click="ignore",
    )
    st.download_button(
        "Scores (CSV)",
        scores_csv(outcome),
        file_name=f"{outcome.folder.name}-scores.csv",
        mime="text/csv",
        on_click="ignore",
    )
