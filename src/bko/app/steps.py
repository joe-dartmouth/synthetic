"""The app's two actions over the stages: analyze the default UCI split, then fill and score it.

Nothing here imports streamlit; `bko.app.page` renders what these return. Each action calls the
stages' public functions and adds no pipeline logic (ADR `research/streamlit-app-design`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from bko.app.context import Context
from bko.datasets.cells import cell_counts
from bko.datasets.fetch import ensure
from bko.datasets.split import Split, split
from bko.datasets.uci import load_uci
from bko.evaluate.guided import Knowledge
from bko.gaps.definitions import Definition, StatementGap, gaps
from bko.gaps.detect import CellGap, count_gaps
from bko.gaps.observe import Cells, record_cells
from bko.gaps.scope import load_scope


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class Analysis:
    """The default split of UCI Heart Disease and its gaps.

    Attributes:
        parts: The default split, E1's first holdout; only `bko.evaluate` reads its test rows.
        knowledge: The releases and field mappings every BKO arm reads.
        cells: The training records observed whole and per cell.
        counts: `cell_counts` of the training split.
        counted: Its count-based gaps.
        found: Each definition's gaps over the training split.
    """

    parts: Split
    knowledge: Knowledge
    cells: Cells
    counts: pd.DataFrame
    counted: tuple[CellGap, ...]
    found: Mapping[Definition, tuple[StatementGap, ...]]


def analyze(context: Context) -> Analysis:
    """Load UCI Heart Disease and the releases, fetching what is absent, and list every gap.

    Raises:
        DatasetError: As `ensure`, `load_uci`, `load_scope` and `record_cells`.
        BKBError: As `gaps`.
        OSError: If the data root cannot be written.
    """
    data_dir = context.data_root()
    ensure(context.uci, data_dir, open_url=context.open_url)
    parts = split(load_uci(data_dir, source=context.uci))
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        parts.seed,
        len(parts.train),
        len(parts.test),
        parts.fingerprint,
    )
    rows = context.field_rows
    ontologies, fragments = load_scope(
        {row.concept for row in rows},
        data_dir,
        releases=context.releases,
        open_url=context.open_url,
    )
    cells = record_cells(parts.train, rows, name=context.uci.name)
    counts = cell_counts(parts.train)
    return Analysis(
        parts=parts,
        knowledge=Knowledge(context.uci.name, ontologies, fragments, rows),
        cells=cells,
        counts=counts,
        counted=count_gaps(counts),
        found={definition: gaps(definition, fragments, cells) for definition in Definition},
    )
