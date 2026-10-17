"""The two gap lists: ontology statements a dataset does not ground, and underrepresented cells.

A BKO gap is an ontology statement that instantiation from the concepts a dataset holds never adds
(ADR `research/bko-formal-gap-definition`): Santos et al.'s algorithm (`santos-2024`, section 5.3)
run from the dataset fragment's priors, the only priors off the source components. Each dataset's
BKO holds the classes its own rows map and their ancestors (ADR `research/ontology-mapping-design`).
A count-based gap is a cell `bko.datasets.cells.cell_counts` flags.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from bko.bkb.fusion import fuse, is_source
from bko.bkb.inference import instantiate
from bko.bkb.structure import BKB, SNode
from bko.datasets.cells import COUNT_COLUMNS
from bko.ontology.fragments import INDIVIDUAL, is_a_fragment, scope
from bko.ontology.terms import Ontology


@dataclass(frozen=True, slots=True, order=True)
class Statement:
    """An ontology statement P(consequent | antecedent) = 1, named by its classes.

    Attributes:
        source: The fragment holding it: its release's source name, such as `mondo`.
        antecedent: The class it reads from: an `is_a` edge's child.
        consequent: The class it concludes: the edge's parent.
    """

    source: str
    antecedent: str
    consequent: str


@dataclass(frozen=True, slots=True)
class CellGap:
    """An age-sex subgroup-label cell holding fewer records than the threshold.

    Attributes:
        age_band: The band's label.
        sex: `female` or `male`.
        target: The label, 0 or 1.
        count: The records it holds.
    """

    age_band: str
    sex: str
    target: int
    count: int

    @property
    def cell(self) -> tuple[str, str, int]:
        """The cell without its count: what two lists compare it by."""
        return (self.age_band, self.sex, self.target)


def ontology_fragments(
    ontologies: Mapping[str, Ontology], concepts: Collection[str]
) -> tuple[BKB, ...]:
    """The `is_a` fragment of each release over `concepts` and their ancestors, in prefix order.

    Args:
        ontologies: The loaded release for each concept prefix.
        concepts: The concepts one dataset's rows map.

    Returns:
        One fragment per release some concept belongs to, named by the release.

    Raises:
        ScopeError: As `is_a_fragment`, for a concept absent from or obsolete in its release.
    """
    scoped = (
        (prefix, ontology, scope(ontology, concepts, prefix=prefix))
        for prefix, ontology in sorted(ontologies.items())
    )
    return tuple(
        is_a_fragment(ontology, classes, prefix=prefix)
        for prefix, ontology, classes in scoped
        if classes
    )


def represent(ontology: Sequence[BKB], data: BKB) -> BKB:
    """Fuse the ontology fragments with the dataset fragment, each at weight 1, `data` last."""
    fragments = [*ontology, data]
    return fuse(fragments, [1.0] * len(fragments))


def detect(ontology: Sequence[BKB], data: BKB) -> tuple[Statement, ...]:
    """The statements of `ontology` the dataset fragment `data` leaves ungrounded, sorted."""
    return ungrounded(represent(ontology, data), _held(data))


def grounded(ontology: Sequence[BKB], data: BKB) -> tuple[Statement, ...]:
    """The statements of `ontology` that `data` grounds, sorted."""
    fused = represent(ontology, data)
    missing = set(ungrounded(fused, _held(data)))
    return tuple(statement for statement in statements(fused) if statement not in missing)


def statements(fused: BKB) -> tuple[Statement, ...]:
    """Every ontology statement of `fused`, sorted.

    A statement is an S-node off the source components whose tail names a class.
    """
    return tuple(sorted(_statement(snode) for snode in fused.snodes if _is_statement(snode)))


def ungrounded(fused: BKB, held: Collection[str]) -> tuple[Statement, ...]:
    """The ontology statements of `fused` that instantiation from `held` never adds, sorted.

    With nothing held there is nothing to instantiate from, and every statement is ungrounded;
    `instantiate` refuses an empty anchor, so it is not called.

    Args:
        fused: A fused BKB.
        held: The classes the dataset fragments hold.

    Raises:
        QueryError: If `held` names a class `fused` lacks.
    """
    anchor = sorted({(concept, INDIVIDUAL) for concept in held})
    added: set[int] = set()
    if anchor:
        for step in instantiate(fused, anchor):
            added.update(step)
    return tuple(
        sorted(
            _statement(snode)
            for index, snode in enumerate(fused.snodes)
            if index not in added and _is_statement(snode)
        )
    )


def count_gaps(counts: pd.DataFrame) -> tuple[CellGap, ...]:
    """The cells a `bko.datasets.cells.cell_counts` frame flags underrepresented, in its order."""
    rows = counts[list(COUNT_COLUMNS)].itertuples(index=False, name=None)
    return tuple(
        CellGap(age_band=str(band), sex=str(sex), target=int(target), count=int(count))
        for band, sex, target, count, under in rows
        if under
    )


def _held(data: BKB) -> list[str]:
    return [snode.head[0] for snode in data.snodes]


def _is_statement(snode: SNode) -> bool:
    return not is_source(snode.head[0]) and any(not is_source(c) for c, _ in snode.tail)


def _statement(snode: SNode) -> Statement:
    # Every ontology statement reads from one class (ADR research/bko-fragment-encoding).
    (antecedent,) = (component for component, _ in snode.tail if not is_source(component))
    (state,) = (state for component, state in snode.tail if is_source(component))
    source = state if isinstance(state, str) else "+".join(sorted(state))
    return Statement(source=source, antecedent=antecedent, consequent=snode.head[0])
