"""The Earlier runs view: the run folders on this machine, each role's stored artifacts.

`docs/DEVELOPMENT.md` (App) says what it shows and derives; ADR `research/streamlit-app-design`
records why it reads stored artifacts only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pandas as pd
import streamlit as st

from bko.app.context import Context
from bko.app.provenance_frames import (
    Line,
    TraceIndex,
    closed_frame,
    concept_frame,
    origin_frame,
    plain,
    prompt_texts,
    read_index,
    record_frame,
    request_shown,
    snode_frame,
    source_frame,
    statement_frame,
    traced,
    validation_frame,
)
from bko.app.runs import Run, call_totals, discover
from bko.app.views import (
    arm_means,
    candidates_table,
    closure_counts,
    e1_validity,
    gap_counts,
    role_table,
    status_counts,
    totals_table,
)
from bko.datasets.errors import DatasetError
from bko.evaluate.errors import EvaluationError
from bko.evaluate.scores import parse
from bko.generate.errors import CandidatesFormatError


logger = logging.getLogger(__name__)

UNREADABLE: Final = (
    EvaluationError,
    CandidatesFormatError,
    ValueError,
    LookupError,
    TypeError,
    AttributeError,
    RecursionError,
    OSError,
)
"""What reading or showing an artifact raises when the file does not hold what its role names,
shown as a message. `EvaluationError` covers `ScoresFormatError`, `validity`'s refusal of a
partial E1 run and a call log line a trace cannot read; `LookupError` covers a missing key or
index; `TypeError` and `AttributeError` a value of the wrong kind, a column the frame display
cannot convert included; `RecursionError` JSON nested too deep to read.
"""

NO_TRACE: Final = (
    "This record's line names no request: no trace is stored in this file for it, only the "
    "record itself."
)
NO_CALL: Final = "Why the record is not traced:"


def earlier_runs(context: Context) -> None:
    """List the run folders and show the chosen one's artifacts, role by role."""
    st.header("Earlier runs")
    try:
        runs = {run.name: run for run in discover(context.runs_root())}
    except (DatasetError, OSError) as exc:
        st.error("The run folders cannot be read.")
        st.code(str(exc), language=None)
        return
    if not runs:
        st.info(
            "No run folder on this machine; docs/DEVELOPMENT.md (Runs) says how runs are "
            "written and copied."
        )
        return
    name: str = st.selectbox("Run", tuple(runs), key="bko.run")
    run = runs[name]
    if run.entry == "app":
        st.caption("A demo fill: not a run of record.")
    else:
        st.caption(
            f"Register entry {run.entry.upper()}: docs/EVIDENCE.md holds its configuration "
            "and result."
        )
    st.code(str(run.folder), language=None)
    _guarded("the folder", partial(_roles, run))
    for role, path in run.roles.items():
        show = SHOWN.get(role)
        if show is not None:
            _guarded(role, partial(show, path))
    if run.entry == "e1" and "scores.csv" in run.roles:
        _guarded("scores.csv", partial(_validity, run.roles["scores.csv"]))


def _guarded(role: str, show: Callable[[], None]) -> None:
    """Run `show`, turning an artifact that does not hold its role's shape into a message."""
    try:
        show()
    except UNREADABLE as exc:
        logger.warning("unreadable role=%s error=%s: %s", role, type(exc).__name__, exc)
        st.error(f"{role} could not be shown.")
        st.code(f"{type(exc).__name__}: {exc}", language=None)


def _roles(run: Run) -> None:
    st.dataframe(role_table(run), hide_index=True)


def _calls(path: Path) -> None:
    st.subheader("Call log totals")
    totals = call_totals(path)
    if totals is None:
        st.info("The call log has no end line: the run was still going or was stopped.")
    else:
        st.dataframe(totals_table(totals), hide_index=True)


def _criteria(path: Path) -> None:
    st.subheader("E1's criteria")
    st.dataframe(pd.read_csv(path), hide_index=True)


def _means(path: Path) -> None:
    st.subheader("Means over the holdouts")
    st.dataframe(arm_means(parse(path.read_text(encoding="utf-8"))))


def _validity(path: Path) -> None:
    st.subheader("Validity")
    st.dataframe(e1_validity(parse(path.read_text(encoding="utf-8"))), hide_index=True)


def _candidates(path: Path) -> None:
    st.subheader("Candidates")
    st.dataframe(candidates_table(path), hide_index=True)


def _closure(path: Path) -> None:
    closure = pd.read_csv(path)
    st.subheader("Gaps before and after the join")
    st.dataframe(closure_counts(closure), hide_index=True)
    st.dataframe(closure, hide_index=True)


def _gaps(title: str, keys: tuple[str, ...]) -> Callable[[Path], None]:
    def show(path: Path) -> None:
        gaps = pd.read_csv(path)
        st.subheader(title)
        st.dataframe(gap_counts(gaps, keys), hide_index=True)
        st.dataframe(gaps, hide_index=True)

    return show


def _reproduction(path: Path) -> None:
    rows = pd.read_csv(path)
    st.subheader("Reproduction")
    st.dataframe(status_counts(rows), hide_index=True)
    st.dataframe(rows, hide_index=True)


@st.cache_resource(max_entries=16, show_spinner=False)
def _cached_index(path: str, mtime_ns: int, size: int) -> TraceIndex:
    """The trace file at `path`, read once per change of its modification time or size."""
    logger.info("trace index path=%s mtime_ns=%d size=%d", path, mtime_ns, size)
    return read_index(Path(path))


def _trace_view(path: Path) -> None:
    """Pick one record of the trace file at `path` by arm, holdout and position, and show it."""
    status = path.stat()
    index = _cached_index(str(path), status.st_mtime_ns, status.st_size)
    st.subheader("Record trace")
    if not index.arms:
        st.info("The trace file holds no line.")
        return
    run = path.parent.name
    arm: str = st.selectbox("Arm", index.arms, key=f"bko.trace.arm.{run}")
    holdout: int = st.selectbox("Holdout", index.holdouts(arm), key=f"bko.trace.holdout.{run}")
    lines = index.lines(arm, holdout)
    record = st.number_input(
        "Record", min_value=0, max_value=len(lines) - 1, step=1, key=f"bko.trace.record.{run}"
    )
    show_trace(lines[int(record)])


def show_trace(line: Line) -> None:
    """Show one trace line: its record beside the facts that determine it, then the raw line.

    Text read from a run folder reaches the page only through frames, JSON and plain code blocks.
    """
    left, right = st.columns([1, 2])
    left.dataframe(record_frame(line), hide_index=True)
    with right:
        st.subheader("Request")
        st.dataframe(origin_frame(line), hide_index=True)
        if traced(line):
            st.json(plain(request_shown(line)))
        else:
            st.info(NO_TRACE)
            if isinstance(line.get("untraced"), str):
                st.caption(NO_CALL)
                st.code(str(line["untraced"]), language=None)
        statements = statement_frame(line)
        if len(statements):
            st.subheader("Gap statements")
            st.dataframe(statements, hide_index=True)
        concepts = concept_frame(line)
        if len(concepts):
            st.subheader("Concepts")
            st.dataframe(concepts, hide_index=True)
            st.dataframe(snode_frame(line), hide_index=True)
        sources = source_frame(line)
        if len(sources):
            st.subheader("Sources")
            st.dataframe(sources, hide_index=True)
            st.dataframe(closed_frame(line), hide_index=True)
        texts = prompt_texts(line)
        if texts:
            st.subheader("Prompt")
            for text in texts:
                st.code(text, language=None)
        st.subheader("Validation")
        st.dataframe(validation_frame(line), hide_index=True)
    with st.expander("The stored line"):
        st.json(plain(line))


def _training(path: Path) -> None:
    st.subheader("Training set")
    st.dataframe(pd.read_csv(path), hide_index=True)


SHOWN: Final[Mapping[str, Callable[[Path], None]]] = MappingProxyType(
    {
        "calls.jsonl": _calls,
        "criteria.csv": _criteria,
        "scores.csv": _means,
        "candidates.jsonl": _candidates,
        "closure.csv": _closure,
        "gaps-uci.csv": _gaps("UCI gap listing", ("detector",)),
        "gaps-synthea.csv": _gaps("E2 gap readings", ("ablation", "detector")),
        "reproduction.csv": _reproduction,
        "training.csv": _training,
        "trace.jsonl": _trace_view,
    }
)
"""Each role the view reads, and how; every other role, such as a log, `first-flag.csv`,
`contradictions.csv` or a `pytest` directory, is listed, not read."""
