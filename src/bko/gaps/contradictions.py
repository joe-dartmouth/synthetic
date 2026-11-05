"""Where a mapped frame contradicts its ontology or the structure its stages assume.

ADR `research/data-contradiction-handling` names five kinds, each a property of a mapped frame:

- `narrow-without-broad`: per cell and `is_a` pair among the mapped concepts, the records that
  show the narrow concept and not the broad one.
- `child-share-above-parent`: per cell and `is_a` pair, the narrow concept's share above the
  broad one's.
- `empty-cell`: a cell of the grid no record falls in.
- `all-show`: a concept every record of a cell shows, of the concepts that do not follow the
  label (`bko.gaps.shares.label_only`), which every record of the label shows by construction.
- `code-overlap`: two concepts whose codes on one field intersect, neither inside the other; a
  pair one inside the other is `bko.gaps.shares.nested`'s, which the sampler draws.

`contradictions` counts them over one frame's records; `bko.evaluate.contradictions` runs it on
each evaluation frame's training split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import numpy as np
import pandas as pd

from bko.datasets.cells import CELL_COLUMNS, cell_counts, cell_keys
from bko.datasets.frames import FrameSpec
from bko.gaps.observe import Cell, by_concept, cell_key, record_shows
from bko.gaps.shares import label_only
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology, ancestors


KINDS: Final = (
    "narrow-without-broad",
    "child-share-above-parent",
    "empty-cell",
    "all-show",
    "code-overlap",
)
"""The kinds, in the order a census lists them."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One contradiction of one kind.

    Attributes:
        kind: One of `KINDS`.
        cell: The cell's key (`bko.gaps.observe.cell_key`); empty for `code-overlap`, a property
            of the rows.
        concept: The narrow concept of an `is_a` pair, the concept every record shows, or the
            first of two overlapping concepts.
        parent: The broad concept of an `is_a` pair, or the second of two overlapping concepts.
        records: The cell's record count.
        shown: How many of the cell's records show `concept`.
        with_parent: How many of those show `parent` too.
        without_parent: How many do not.
        child_share: `shown` over `records`.
        parent_share: The share of the cell's records showing `parent`.
        field: The field two overlapping concepts' codes share.
        codes: The codes they share, in ascending order.
    """

    kind: str
    cell: str = ""
    concept: str = ""
    parent: str = ""
    records: int | None = None
    shown: int | None = None
    with_parent: int | None = None
    without_parent: int | None = None
    child_share: float | None = None
    parent_share: float | None = None
    field: str = ""
    codes: tuple[int, ...] = ()


def is_a_pairs(
    concepts: Sequence[str], ontologies: Mapping[str, Ontology]
) -> tuple[tuple[str, str], ...]:
    """Each `(narrow, broad)` pair of `concepts` that `is_a` joins, directly or through others.

    A concept's ancestors are followed within its own prefix's release, as
    `bko.generate.direct`'s closure follows them; a concept whose prefix has no release has none.
    """
    mapped = frozenset(concepts)
    pairs = set()
    for concept in mapped:
        prefix = concept.partition(":")[0]
        ontology = ontologies.get(prefix)
        if ontology is not None:
            above = ancestors(ontology, concept, prefix=prefix)
            pairs.update((concept, broad) for broad in above & mapped)
    return tuple(sorted(pairs))


def code_overlaps(rows: Sequence[FieldMapping]) -> tuple[Finding, ...]:
    """Each pair of concepts whose codes on one field intersect, neither holding the other's.

    A concept's codes on a field are the union of every `Codes` clause its rows read there.
    """
    codes: dict[str, dict[str, frozenset[int]]] = {}
    for concept, mapped in by_concept(rows).items():
        for row in mapped:
            for clause in row.when:
                if isinstance(clause, Codes):
                    fields = codes.setdefault(clause.field, {})
                    fields[concept] = fields.get(concept, frozenset()) | clause.codes
    found = []
    for field in sorted(codes):
        for first, second in combinations(sorted(codes[field]), 2):
            mine, theirs = codes[field][first], codes[field][second]
            shared = mine & theirs
            if shared and not (mine <= theirs or theirs <= mine):
                found.append(
                    Finding(
                        "code-overlap", "", first, second, field=field, codes=tuple(sorted(shared))
                    )
                )
    return tuple(found)


def contradictions(
    train: pd.DataFrame,
    rows: Sequence[FieldMapping],
    ontologies: Mapping[str, Ontology],
    *,
    frame: FrameSpec,
    name: str,
) -> tuple[Finding, ...]:
    """Every contradiction of `train` read through `rows` and `ontologies`, in `KINDS` order.

    Within a kind, cells come in the grid's order (`bko.datasets.cells.cell_counts`), every cell
    listed, an empty one included, and pairs and concepts in sorted order.

    Args:
        train: The records, with `age`, `sex` and the label beside the fields the rows read.
        rows: The field mappings.
        ontologies: The loaded releases, by concept prefix.
        frame: The records' frame: its age band edges and its label.
        name: The records' name, for an error.

    Raises:
        GapError: As `bko.gaps.observe.record_shows`, if a row reads a field `train` lacks.
        ContractError: As `bko.datasets.cells.cell_keys`.
    """
    shows = record_shows(train, rows, name=name)
    keys = cell_keys(train, edges=frame.edges)
    grid = cell_counts(train, edges=frame.edges)[list(CELL_COLUMNS)]
    cells: list[tuple[Cell, pd.DataFrame]] = []
    for band, sex, label in grid.itertuples(index=False, name=None):
        cell: Cell = (str(band), str(sex), int(label))
        inside = np.ones(len(keys), dtype=bool)
        for column, value in zip(keys.columns, cell, strict=True):
            inside &= keys[column].eq(value).to_numpy(dtype=bool)
        cells.append((cell, shows.loc[inside]))
    pairs = is_a_pairs(list(shows.columns), ontologies)
    following = label_only(rows, frame=frame)
    found: dict[str, list[Finding]] = {kind: [] for kind in KINDS}
    for cell, members in cells:
        key, records = cell_key(cell), len(members)
        if records == 0:
            found["empty-cell"].append(Finding("empty-cell", key, records=0))
            continue
        for narrow, broad in pairs:
            _pair(found, key, members, narrow, broad)
        for concept in shows.columns:
            if concept not in following and bool(members[concept].all()):
                found["all-show"].append(
                    Finding("all-show", key, str(concept), records=records, shown=records)
                )
    found["code-overlap"].extend(code_overlaps(rows))
    return tuple(finding for kind in KINDS for finding in found[kind])


def _pair(
    found: dict[str, list[Finding]], key: str, members: pd.DataFrame, narrow: str, broad: str
) -> None:
    """Add the cell's `narrow-without-broad` and `child-share-above-parent` findings for a pair."""
    child = members[narrow].to_numpy(dtype=bool)
    parent = members[broad].to_numpy(dtype=bool)
    records, shown, held = len(members), int(child.sum()), int(parent.sum())
    with_parent = int((child & parent).sum())
    for kind, holds in (
        ("narrow-without-broad", shown > with_parent),
        ("child-share-above-parent", shown > held),
    ):
        if holds:
            found[kind].append(
                Finding(
                    kind,
                    key,
                    narrow,
                    broad,
                    records,
                    shown,
                    with_parent,
                    shown - with_parent,
                    shown / records,
                    held / records,
                )
            )
