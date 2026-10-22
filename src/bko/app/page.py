"""The Streamlit page: pick the dataset, analyze its gaps, fill and score them, download the result.

The only module that imports streamlit. Each action runs with no streamlit call inside it and
writes its result into a `Slot` placed in the session before it starts, so a rerun requested
meanwhile neither aborts it nor loses its result (ADR `research/streamlit-app-design`).
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from bko.app.context import Context


CONTEXT_KEY: Final = "bko.context"
"""The session key holding the `Context`; tests set it before the first run."""

DATASETS: Final = ("UCI Heart Disease",)


def page() -> None:
    """Render the page for this run of the script."""
    st.set_page_config(page_title="BKO gap filling", layout="wide")
    context: Context = st.session_state.setdefault(CONTEXT_KEY, Context())
    context.setup_logging()
    st.title("Fill clinical coverage gaps")
    st.header("1. Dataset")
    st.selectbox("Dataset", DATASETS, key="bko.dataset")
