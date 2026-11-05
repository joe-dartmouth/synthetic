"""Where a mapped frame contradicts its ontology or the structure its stages assume.

ADR `research/data-contradiction-handling` names five kinds, each a property of a mapped frame:

- `narrow-without-broad`: per cell and `is_a` pair among the mapped concepts, the records that
  show the narrow concept and not the broad one.
- `child-share-above-parent`: per cell and `is_a` pair, the narrow concept's share above the
  broad one's.
- `empty-cell`: a cell of the grid no record falls in.
- `all-show`: a concept every record of a cell shows, of the concepts that do not follow the
  label (`bko.gaps.shares.label_only`), which every record of the label shows by construction.
  A cell of few records shows a concept in every record far more readily than a large one, so
  the kind fires most in the smallest cells; it is counted there, not guarded.
- `code-overlap`: two concepts whose codes on one field intersect, unless
  `bko.gaps.shares.nested` resolves the pair, which the sampler draws. A concept's codes on a
  field are taken per row: the codes every one of the row's `Codes` clauses on that field
  admits, united over the concept's rows. A row's clauses on other fields, and its clauses that
  are not `Codes`, are ignored for this kind, so a row reading codes on one field and a threshold
  on another counts its codes alone.

`survey` counts them over one frame's records, with each `is_a` pair's totals over the whole
frame, and `contradictions` returns its findings; `bko.evaluate.contradictions` runs `survey` on
each evaluation frame's training split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from typing import Final, Literal, get_args

import pandas as pd

from bko.datasets.cells import cell_keys
from bko.datasets.frames import FrameSpec
from bko.gaps.observe import by_concept, cell_grid, cell_key, in_cell, record_shows
from bko.gaps.shares import label_only, nested
from bko.ontology.schema import Codes, FieldMapping
from bko.ontology.terms import Ontology, ancestors


type Kind = Literal[
    "narrow-without-broad",
    "child-share-above-parent",
    "empty-cell",
    "all-show",
    "code-overlap",
]
"""A kind of contradiction."""

KINDS: Final[tuple[Kind, ...]] = get_args(Kind.__value__)
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

    kind: Kind
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


@dataclass(frozen=True, slots=True)
class PairTotal:
    """One `is_a` pair's counts over every record of a frame.

    Attributes:
        concept: The narrow concept.
        parent: The broad concept.
        shown: How many records show `concept`.
        without: How many of those do not show `parent`.
        cells: How many cells hold the pair's `narrow-without-broad` finding.
    """

    concept: str
    parent: str
    shown: int
    without: int
    cells: int


@dataclass(frozen=True, slots=True)
class Survey:
    """Every contradiction of one frame's records and what the census read to find them.

    Attributes:
        findings: Every finding, in `KINDS` order (`contradictions`).
        pairs: Each `is_a` pair's totals, in sorted order.
        cells: The grid's cell count.
        examined: How many of the cells hold a record, so are read for every kind but
            `empty-cell`.
        following: The concepts that follow the label, which `all-show` leaves out.
    """

    findings: tuple[Finding, ...]
    pairs: tuple[PairTotal, ...]
    cells: int
    examined: int
    following: frozenset[str]


def is_a_pairs(
    concepts: Sequence[str], ontologies: Mapping[str, Ontology]
) -> tuple[tuple[str, str], ...]:
    """Each `(narrow, broad)` pair of `concepts` that `is_a` joins, directly or through others.

    A concept's ancestors are followed within its own prefix's release, as
    `bko.generate.direct`'s closure follows them; a concept whose prefix has no release has none.
    A release whose `is_a` links cycle back to a concept pairs it with no copy of itself.
    """
    mapped = frozenset(concepts)
    pairs = set()
    for concept in mapped:
        prefix = concept.partition(":")[0]
        ontology = ontologies.get(prefix)
        if ontology is not None:
            above = ancestors(ontology, concept, prefix=prefix) - {concept}
            pairs.update((concept, broad) for broad in above & mapped)
    return tuple(sorted(pairs))


def code_overlaps(rows: Sequence[FieldMapping]) -> tuple[Finding, ...]:
    """Each pair of concepts whose codes on one field intersect, but those `nested` resolves.

    A concept's codes on a field are the codes every one of a row's `Codes` clauses on that field
    admits, united over its rows; its other clauses are ignored. Pairs come by field, then by
    concept, each in sorted order.

    Raises:
        ValueError: As `bko.gaps.shares.nested`, if the rows nest in a chain.
    """
    resolved = {frozenset(pair) for pair in nested(rows).items()}
    codes: dict[str, dict[str, frozenset[int]]] = {}
    for concept, mapped in by_concept(rows).items():
        for row in mapped:
            for field in sorted({c.field for c in row.when if isinstance(c, Codes)}):
                read = [c.codes for c in row.when if isinstance(c, Codes) and c.field == field]
                fields = codes.setdefault(field, {})
                fields[concept] = fields.get(concept, frozenset()) | reduce(frozenset.__and__, read)
    found = []
    for field in sorted(codes):
        for first, second in combinations(sorted(codes[field]), 2):
            shared = codes[field][first] & codes[field][second]
            if shared and frozenset((first, second)) not in resolved:
                found.append(
                    Finding(
                        "code-overlap",
                        concept=first,
                        parent=second,
                        field=field,
                        codes=tuple(sorted(shared)),
                    )
                )
    return tuple(found)


def survey(
    train: pd.DataFrame,
    rows: Sequence[FieldMapping],
    ontologies: Mapping[str, Ontology],
    *,
    frame: FrameSpec,
    name: str,
) -> Survey:
    """Every contradiction of `train` read through `rows` and `ontologies`, with its pair totals.

    Within a kind, cells come in the grid's order (`bko.datasets.cells.cell_counts`), and pairs
    and concepts in sorted order. Every cell is examined, an empty one for `empty-cell` alone;
    only a cell with a finding is listed.

    Args:
        train: The records, with `age`, `sex` and the label beside the fields the rows read.
        rows: The field mappings.
        ontologies: The loaded releases, by concept prefix.
        frame: The records' frame: its age band edges and its label.
        name: The records' name, for an error.

    Raises:
        GapError: As `bko.gaps.observe.record_shows`, if a row reads a field `train` lacks.
        ContractError: As `bko.datasets.cells.cell_keys`.
        ValueError: As `code_overlaps`.
    """
    shows = record_shows(train, rows, name=name)
    keys = cell_keys(train, edges=frame.edges)
    grid = cell_grid(train, frame.edges)
    pairs = is_a_pairs(list(shows.columns), ontologies)
    following = label_only(rows, frame=frame)
    found: dict[Kind, list[Finding]] = {kind: [] for kind in KINDS}
    for cell in grid:
        members = shows.loc[in_cell(keys, cell)]
        key, records = cell_key(cell), len(members)
        if records == 0:
            found["empty-cell"].append(Finding("empty-cell", key, records=0))
            continue
        for narrow, broad in pairs:
            _pair(found, key, members, narrow, broad)
        for concept in shows.columns:
            if concept not in following and bool(members[concept].all()):
                found["all-show"].append(
                    Finding(
                        "all-show",
                        key,
                        str(concept),
                        records=records,
                        shown=records,
                        child_share=1.0,
                    )
                )
    found["code-overlap"].extend(code_overlaps(rows))
    totals = tuple(
        PairTotal(
            narrow,
            broad,
            int(shows[narrow].sum()),
            int((shows[narrow] & ~shows[broad]).sum()),
            sum(f.concept == narrow and f.parent == broad for f in found["narrow-without-broad"]),
        )
        for narrow, broad in pairs
    )
    return Survey(
        tuple(finding for kind in KINDS for finding in found[kind]),
        totals,
        len(grid),
        len(grid) - len(found["empty-cell"]),
        following,
    )


def contradictions(
    train: pd.DataFrame,
    rows: Sequence[FieldMapping],
    ontologies: Mapping[str, Ontology],
    *,
    frame: FrameSpec,
    name: str,
) -> tuple[Finding, ...]:
    """Every contradiction of `train` read through `rows` and `ontologies`, in `KINDS` order.

    `survey`'s findings; its Args and Raises hold here too.
    """
    return survey(train, rows, ontologies, frame=frame, name=name).findings


def _pair(
    found: dict[Kind, list[Finding]], key: str, members: pd.DataFrame, narrow: str, broad: str
) -> None:
    """Add the cell's `narrow-without-broad` and `child-share-above-parent` findings for a pair.

    Args:
        found: The findings so far, by kind; each holding kind gains one.
        key: The cell's key.
        members: Whether each of the cell's records shows each concept.
        narrow: The pair's narrow concept.
        broad: The pair's broad concept.
    """
    child = members[narrow].to_numpy(dtype=bool)
    parent = members[broad].to_numpy(dtype=bool)
    records, shown, held = len(members), int(child.sum()), int(parent.sum())
    with_parent = int((child & parent).sum())
    holding: tuple[tuple[Kind, bool], ...] = (
        ("narrow-without-broad", shown > with_parent),
        ("child-share-above-parent", shown > held),
    )
    for kind, holds in holding:
        if holds:
            found[kind].append(
                Finding(
                    kind,
                    cell=key,
                    concept=narrow,
                    parent=broad,
                    records=records,
                    shown=shown,
                    with_parent=with_parent,
                    without_parent=shown - with_parent,
                    child_share=shown / records,
                    parent_share=held / records,
                )
            )
