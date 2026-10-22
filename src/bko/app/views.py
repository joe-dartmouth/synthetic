"""The frames the page shows, built from what `bko.app.steps` and `bko.app.runs` return.

Nothing here imports streamlit, so tests read the values the page shows.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from bko.app.steps import Analysis
from bko.datasets.cells import CELL_COLUMNS, LABEL
from bko.gaps.definitions import Definition
from bko.gaps.report import STATEMENT_COLUMNS, cell_cells, labeller, statement_cells


GAP_COLUMNS: Final = (*STATEMENT_COLUMNS, *CELL_COLUMNS)


def cell_table(analysis: Analysis) -> pd.DataFrame:
    """The training split's cell counts, with a `status` word, so colour never carries it alone."""
    counts = analysis.counts
    words = counts["underrepresented"].map({True: "underrepresented", False: "covered"})
    return counts.assign(status=words)


def balance(analysis: Analysis) -> pd.DataFrame:
    """The training records per label."""
    return analysis.counts.groupby(LABEL, as_index=False)[["count"]].sum()


def gap_table(analysis: Analysis, definition: Definition) -> pd.DataFrame:
    """One row per gap under `definition`: its statement with labels, then its cell."""
    label = labeller(analysis.knowledge.ontologies)
    rows = [
        (*statement_cells(gap.statement, label), *cell_cells(gap.cell))
        for gap in analysis.found[definition]
    ]
    return pd.DataFrame(rows, columns=list(GAP_COLUMNS))
