"""One frame's gap listing over its training split: the rows `python -m bko.gaps uci` writes.

`listed` takes what a command holds once it has split a frame's records, loaded the releases its
field rows name and observed the training cells, so any frame whose cells are an age band, a sex
and the label lists its gaps through it. It lists each definition's gaps of
`bko.gaps.definitions`, as (cell, statement) pairs, then the cells holding fewer training records
than the threshold, and logs each input's identity, each cell's records, each flagged concept,
each list's length and D1's sweep (`bko.gaps.sweep`) to the `bko.gaps` logger.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Final

import pandas as pd

from bko.bkb.structure import BKB
from bko.datasets.cells import CELL_COLUMNS, UNDERREPRESENTED_BELOW, cell_counts
from bko.gaps.definitions import Definition, gaps, under_represented
from bko.gaps.detect import count_gaps
from bko.gaps.observe import Cells, Observed, cell_key
from bko.gaps.report import STATEMENT_COLUMNS, Row, cell_cells, labeller, statement_cells
from bko.gaps.sweep import listing, log_listing
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.gaps")

LISTING_COLUMNS: Final[Row] = ("detector", *STATEMENT_COLUMNS, *CELL_COLUMNS, "count")
"""The listing's header row."""


def listed(
    train: pd.DataFrame,
    *,
    edges: Sequence[int],
    ontologies: Mapping[str, Ontology],
    fragments: tuple[BKB, ...],
    cells: Cells,
) -> list[Row]:
    """The header, each definition's gaps, then the count-based gaps of `train`.

    Args:
        train: The training records.
        edges: The frame's age band starts.
        ontologies: Each loaded release, by concept prefix, which names each statement's concept.
        fragments: Each release's fragment over the field rows' concepts.
        cells: `train` observed whole and per cell; its whole name is the dataset's.

    Returns:
        `LISTING_COLUMNS`, then one row per gap of each `Definition`, in definition order, then
        one `count` row per cell below `UNDERREPRESENTED_BELOW`.
    """
    log_observed(cells.whole)
    log_cells(cells, f"dataset={cells.whole.name}")
    label = labeller(ontologies)
    rows: list[Row] = [LISTING_COLUMNS]
    for definition in Definition:
        found = gaps(definition, fragments, cells)
        logger.info(
            "bko definition=%s dataset=%s gaps=%d cells=%d",
            definition,
            cells.whole.name,
            len(found),
            len({gap.cell for gap in found}),
        )
        rows.extend(
            (definition.value, *statement_cells(gap.statement, label), *cell_cells(gap.cell), "")
            for gap in found
        )
    log_listing(cells.whole.name, listing(fragments, cells))
    counted = count_gaps(cell_counts(train, edges=edges))
    logger.info(
        "count dataset=%s flagged=%d below=%d",
        cells.whole.name,
        len(counted),
        UNDERREPRESENTED_BELOW,
    )
    rows.extend(
        ("count", "", "", "", "", "", cell.age_band, cell.sex, str(cell.target), str(cell.count))
        for cell in counted
    )
    return rows


def log_observed(observed: Observed) -> None:
    """Log one `observed` line: the records read and how many show each mapped concept."""
    logger.info(
        "observed dataset=%s records=%d held=%d of=%d %s",
        observed.name,
        observed.records,
        len(observed.held),
        len(observed.holders),
        " ".join(f"{concept}={count}" for concept, count in observed.holders.items()),
    )


def log_cells(cells: Cells, where: str) -> None:
    """Log one `cell` line per cell, then one `flagged` line per under-represented concept."""
    for cell, records in cells.each.items():
        logger.info(
            "cell %s key=%s records=%d held=%d of=%d assumed=%s",
            where,
            cell_key(cell),
            records.observed.records,
            len(records.observed.held),
            len(records.observed.holders),
            ",".join(sorted(records.assumed)),
        )
    for flag in under_represented(cells):
        logger.info(
            "flagged %s key=%s concept=%s shown=%d records=%d reference_shown=%d "
            "reference_records=%d mass=%.4f",
            where,
            cell_key(flag.cell),
            flag.concept,
            flag.shown,
            flag.records,
            flag.reference_shown,
            flag.reference_records,
            flag.mass,
        )
