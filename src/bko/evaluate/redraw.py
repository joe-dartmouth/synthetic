"""A `bko-direct` fill re-drawn at its logged seed, and the facts behind each draw's concepts.

`redraw` draws a request again through `bko.generate.direct.sample` and takes the draws through an
`bko.evaluate.attempts.Attempt` as `bko.evaluate.direct.DirectArm` does, so its rows are the fill's
rows byte for byte when the code and the data are the run's. `Shares` holds each asked cell's
records and its label group's, from which `concept_facts` reads one concept's share, its parts,
its S-nodes and its label. `bko.evaluate.trace` writes both (ADR `research/record-explainability`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import pandas as pd

from bko.baselines.request import Request
from bko.bkb.fusion import source_component
from bko.evaluate.attempts import Attempt
from bko.evaluate.data import Data
from bko.evaluate.holdout_candidates import KEPT
from bko.gaps.observe import Cell, Observed, record_cells
from bko.gaps.shares import GROUP_WEIGHT, SharePart, share_parts
from bko.generate.direct import Draw, label_group, sample
from bko.ontology.fragments import INDIVIDUAL
from bko.ontology.terms import Ontology


type Json = Mapping[str, object]


@dataclass(frozen=True, slots=True, eq=False)
class Redrawn:
    """A fill drawn again.

    Attributes:
        rows: The rows the arm returns.
        kept: The draws those rows came from, in generation order.
        drawn: Every draw, in generation order, as the attempt's candidate lines hold them.
    """

    rows: pd.DataFrame
    kept: tuple[Draw, ...]
    drawn: tuple[Draw, ...] = ()


def redraw(
    train: pd.DataFrame,
    request: Request,
    *,
    data: Data,
    ontologies: Mapping[str, Ontology],
    budget: int,
    tag: tuple[int, str],
) -> Redrawn:
    """Draw `request` at its seed as the direct arm does, with no other random draw.

    Raises:
        SamplingError: As `sample`.
    """
    drawn = sample(
        train,
        request.quota,
        rows=data.field_rows,
        ontologies=ontologies,
        name=data.name,
        seed=request.seed,
        frame=data.frame,
        tag=tag,
    )
    attempt = Attempt(train, request, budget * request.rows)
    draws: list[Draw] = []
    for cell, each in drawn.items():
        attempt.add(None, cell, len(each), [dict(draw.record) for draw in each])
        draws += each
    kept = tuple(draws[j] for j, seen in enumerate(attempt.seen) if seen.outcome == KEPT)
    return Redrawn(attempt.rows(), kept, tuple(draws))


class Shares:
    """Each asked cell's records and its label group's, as the direct sampler reads them."""

    def __init__(
        self,
        train: pd.DataFrame,
        cells: Collection[Cell],
        *,
        data: Data,
        ontologies: Mapping[str, Ontology],
    ) -> None:
        """Observe `train`'s cells and the label groups of `cells`.

        Raises:
            GapError: As `record_cells` and `label_group`.
        """
        rows, frame = data.field_rows, data.frame
        self._cells = record_cells(train, rows, name=data.name, edges=frame.edges).each
        self._groups = {
            label: label_group(train, label, rows, name=data.name, frame=frame)
            for label in {int(cell[2]) for cell in cells}
        }
        self._ontologies = ontologies

    def concepts(self, draw: Draw) -> tuple[Json, ...]:
        """`concept_facts` of every mapped concept at `draw`'s cell, in sorted order."""
        records, group = self._cells[draw.cell], self._groups[int(draw.cell[2])]
        return tuple(
            concept_facts(concept, part, draw.concepts, records.observed, group, self._ontologies)
            for concept, part in share_parts(records, group).items()
        )


def concept_facts(
    concept: str,
    part: SharePart,
    present: Collection[str],
    cell: Observed,
    group: Observed,
    ontologies: Mapping[str, Ontology],
) -> Json:
    """One concept's share at a cell, its parts with their counts, its S-nodes and its label.

    The label is the term's name in its pinned release, None when the release lacks it. Each
    fragment that holds a prior for the concept names one S-node: head `(concept, INDIVIDUAL)`,
    tail `(__Source__[concept], <fragment>)`. `dataset_fragment` gives a prior to each concept
    its records hold (`Observed.held`), so the S-node is named by that presence, whatever its
    probability.
    """
    release = ontologies.get(concept.partition(":")[0])
    term = None if release is None else release.terms.get(concept)
    snodes = [
        {"head": [concept, INDIVIDUAL], "tail": [[source_component(concept), seen.name]], "prob": p}
        for seen, p in ((cell, part.p_cell), (group, part.p_group))
        if concept in seen.held
    ]
    return {
        "concept": concept,
        "label": None if term is None else term.name,
        "present": concept in present,
        "share": part.share,
        "cell_prior": part.p_cell,
        "group_prior": part.p_group,
        "cell_records": part.n_cell,
        "cell_holders": cell.holders.get(concept, 0),
        "group_records": group.records,
        "group_holders": group.holders.get(concept, 0),
        "group_weight": GROUP_WEIGHT,
        "snodes": snodes,
    }
