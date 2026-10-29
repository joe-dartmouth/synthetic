"""A prescribing request's lines: per asked record, the concepts it must show, and no others.

A `Skeleton` names every mapped concept that is not label-only, by curie, with the rows that map
it, and holds one line per asked record: the named concepts that record must show. The concepts
come from `bko.generate.direct.sample`'s draws (ADR `research/bkb-direct-generation`), and
`bko.generate.prompt.render` writes the lines after the level's components (ADR
`research/record-generation-design`, What every level sends). `Skeleton.shows` reads the named
concepts a decoded record shows, which `bko.baselines.records.assess` matches against the call's
lines by concept set, so a record matching no line left unconsumed is dropped and topped up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.generate.direct import label_only
from bko.ontology.schema import FieldMapping, shown


@dataclass(frozen=True, slots=True)
class Named:
    """One concept a line may hold.

    Attributes:
        concept: The curie.
        label: Its label in the pinned release, as the prompt names it.
        states: The rows mapping it, in the table's order.
    """

    concept: str
    label: str
    states: tuple[FieldMapping, ...]


@dataclass(frozen=True, slots=True)
class Skeleton:
    """One call's prescriptions.

    Attributes:
        named: Every mapped concept that is not label-only, by curie.
        lines: Per asked record, in order: the named concepts it must show.
    """

    named: tuple[Named, ...]
    lines: tuple[frozenset[str], ...]

    def shows(self, record: Mapping[str, float]) -> frozenset[str]:
        """The named concepts `record` shows, a value per field name."""
        return shown(record, (row for one in self.named for row in one.states))


def prescribable(
    rows: Sequence[FieldMapping], *, frame: FrameSpec = UCI_FRAME
) -> tuple[FieldMapping, ...]:
    """The rows of every concept that is not label-only, in the table's order."""
    following = label_only(rows, frame=frame)
    return tuple(row for row in rows if row.concept not in following)


def skeleton(
    lines: Sequence[frozenset[str]], rows: Sequence[FieldMapping], *, frame: FrameSpec = UCI_FRAME
) -> Skeleton:
    """Name `rows`' prescribable concepts and hold `lines`, one per asked record.

    Args:
        lines: Per asked record, the concepts it must show, label-only ones left out.
        rows: The field mappings.
        frame: The frame the records belong to; its label decides which concepts are label-only.

    Raises:
        ValueError: If a line names a concept outside the named ones, or two named concepts share
            a label, so a line could not say which it means.
    """
    by_concept: dict[str, list[FieldMapping]] = {}
    for row in prescribable(rows, frame=frame):
        by_concept.setdefault(row.concept, []).append(row)
    named = tuple(
        Named(concept, states[0].label, tuple(states))
        for concept, states in sorted(by_concept.items())
    )
    labels = [one.label for one in named]
    if len(set(labels)) != len(labels):
        msg = f"two prescribable concepts share a label among {labels}"
        raise ValueError(msg)
    for position, line in enumerate(lines):
        if stray := sorted(line - by_concept.keys()):
            msg = f"line {position + 1} names {stray}, which no prescribable row maps"
            raise ValueError(msg)
    return Skeleton(named, tuple(frozenset(line) for line in lines))
