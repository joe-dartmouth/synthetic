"""Each mapped concept's share under the fused BKB, and the concept structure its draw reads.

The fused BKB holds, per mapped concept, a prior from the cell's dataset fragment and one from its
label group's (`bko.gaps.observe.dataset_fragment`). `shares` reads a concept's share from the two;
`label_only` names the concepts that follow the label and `nested` the concept pairs whose codes lie
one inside the other. The direct sampler (`bko.generate.direct`) and the likelihood score
(`bko.gaps.likelihood`) read them alike (ADR `research/bkb-direct-generation`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Final

from bko.bkb.structure import BKB
from bko.datasets.cells import UNDERREPRESENTED_BELOW
from bko.datasets.frames import FrameSpec
from bko.gaps.observe import CellRecords, Observed, by_concept, dataset_fragment
from bko.ontology.schema import Codes, FieldMapping


GROUP_WEIGHT: Final = UNDERREPRESENTED_BELOW
"""The label group's weight in a share, in records: a cell this small weighs as its group does."""


def shares(cell: CellRecords, group: Observed) -> dict[str, float]:
    """Each mapped concept's share: `(n_c p_c + GROUP_WEIGHT p_g) / (n_c + GROUP_WEIGHT)`.

    `p_c` is the cell's dataset-fragment prior and `p_g` its label group's, each 0 where the
    fragment holds no S-node for the concept; `n_c` is the cell's record count. A cell of no
    record takes `p_g` itself, since multiplying by `GROUP_WEIGHT` and dividing back can move it
    by one ulp.

    Args:
        cell: The cell's records.
        group: The records of the cell's label, observed over the same rows.

    Returns:
        The share of every concept `group` counts, in its sorted order.
    """
    mine = _priors(dataset_fragment(cell.observed))
    theirs = _priors(dataset_fragment(group))
    records = cell.observed.records
    return {
        concept: theirs.get(concept, 0.0)
        if records == 0
        else (records * mine.get(concept, 0.0) + GROUP_WEIGHT * theirs.get(concept, 0.0))
        / (records + GROUP_WEIGHT)
        for concept in group.holders
    }


def label_only(rows: Sequence[FieldMapping], *, frame: FrameSpec) -> frozenset[str]:
    """The concepts every row of which reads only the label: they follow the cell's label."""
    return frozenset(
        concept
        for concept, mapped in by_concept(rows).items()
        if all(clause.field == frame.label for row in mapped for clause in row.when)
    )


def nested(rows: Sequence[FieldMapping]) -> dict[str, str]:
    """Each concept whose one row's codes lie strictly inside another's, on one field, to it.

    On UCI, `HP:0001712` (`restecg` 2) to `HP:0003115` (`restecg` 1 or 2): a record showing the
    first shows the second, so the second is drawn given the first. Only one level is drawn.

    Raises:
        ValueError: If a concept lies inside more than one other, so a chain of nesting, or holds
            more than one.
    """
    single = {
        concept: mapped[0].when[0]
        for concept, mapped in by_concept(rows).items()
        if len(mapped) == 1 and len(mapped[0].when) == 1 and isinstance(mapped[0].when[0], Codes)
    }
    pairs = [
        (inner, outer)
        for inner, small in single.items()
        for outer, large in single.items()
        if isinstance(small, Codes)
        and isinstance(large, Codes)
        and small.field == large.field
        and small.codes < large.codes
    ]
    for side, word in ((0, "lies inside"), (1, "holds")):
        counts = Counter(pair[side] for pair in pairs)
        for concept, times in sorted(counts.items()):
            if times > 1:
                msg = f"{concept} {word} {times} concepts; only one level of nesting is drawn"
                raise ValueError(msg)
    return dict(pairs)


def _priors(fragment: BKB) -> dict[str, float]:
    return {snode.head[0]: snode.prob for snode in fragment.snodes}
