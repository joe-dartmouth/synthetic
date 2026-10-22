"""The frames the page shows, built from what `bko.app.steps` and `bko.app.runs` return.

Nothing here imports streamlit, so tests read the values the page shows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pandas as pd

from bko.app.steps import Analysis, Outcome
from bko.datasets.cells import CELL_COLUMNS, LABEL
from bko.evaluate.scores import Score, Source
from bko.gaps.definitions import Definition
from bko.gaps.report import STATEMENT_COLUMNS, cell_cells, labeller, statement_cells


GAP_COLUMNS: Final = (*STATEMENT_COLUMNS, *CELL_COLUMNS)
PAIR_COLUMNS: Final = ("definition", "status", *GAP_COLUMNS)
SUMMARY_COLUMNS: Final = ("definition", "before", "closed", "still_open", "new")
SCORE_INDEX: Final = ("source", "measure", "scope")


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


def closure_summary(outcome: Outcome) -> pd.DataFrame:
    """Each definition's gaps before the fill, and how many closed, stayed open or are new."""
    rows = [
        (c.definition.value, len(c.before), len(c.closed), len(c.still_open), len(c.opened))
        for c in outcome.closures
    ]
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def closure_pairs(analysis: Analysis, outcome: Outcome) -> pd.DataFrame:
    """Every gap listed before or after the fill, with whether it closed, stayed open or is new."""
    label = labeller(analysis.knowledge.ontologies)
    rows = []
    for reading in outcome.closures:
        closed, opened = set(reading.closed), set(reading.opened)
        for gap in reading.listed(analysis.cells.each):
            status = "closed" if gap in closed else "new" if gap in opened else "still open"
            cells = (*statement_cells(gap.statement, label), *cell_cells(gap.cell))
            rows.append((reading.definition.value, status, *cells))
    return pd.DataFrame(rows, columns=list(PAIR_COLUMNS))


def score_table(scores: Sequence[Score], source: Source | None = None) -> pd.DataFrame:
    """One holdout's scores, one row per source, measure and scope and one column per arm."""
    rows = [
        (s.source.value, s.measure.value, s.scope.value, s.arm, s.value)
        for s in scores
        if source is None or s.source is source
    ]
    frame = pd.DataFrame(rows, columns=[*SCORE_INDEX, "arm", "value"])
    return frame.pivot(index=list(SCORE_INDEX), columns="arm", values="value")
