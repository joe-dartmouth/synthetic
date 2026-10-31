"""A record's log-probability under the fused BKB's concept draw: the BKB likelihood of a candidate.

The fused BKB gives every record of a cell the same joint mass, so the score it can give a record is
the probability that the direct sampler (`bko.generate.direct`) draws the record's mapped concepts,
present and absent, at the cell's shares (`bko.gaps.shares.shares`). Each concept that is neither
label-only nor unmapped adds the log of its share when the record shows it, by any of its rows, and
of one less its share when not; the outer concept of a nested pair is drawn given its inner one, at
`(s_o - s_i) / (1 - s_i)`, and is certain once the inner is present. A record the draw cannot make
scores minus infinity. ADR `research/bkb-candidate-curation` fixes the score and the gate that reads
it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from bko.datasets.frames import FrameSpec
from bko.gaps.observe import by_concept
from bko.gaps.shares import label_only, nested
from bko.ontology.schema import FieldMapping


@dataclass(frozen=True, slots=True)
class Likelihood:
    """A record's log-probability under the sampler's concept draw at one cell's shares.

    Attributes:
        shares: Each concept's share in the cell; a concept it lacks has share 0.
        rows: The field mappings the concepts are read by.
        frame: The frame whose label marks the label-only concepts.
    """

    shares: Mapping[str, float]
    rows: tuple[FieldMapping, ...]
    frame: FrameSpec
    _mapped: Mapping[str, list[FieldMapping]] = field(init=False, repr=False)
    _following: frozenset[str] = field(init=False, repr=False)
    _inner_of: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Read the concepts, the label-only ones and the nested pairs from the rows once."""
        nesting = nested(self.rows)
        object.__setattr__(self, "_mapped", by_concept(self.rows))
        object.__setattr__(self, "_following", label_only(self.rows, frame=self.frame))
        object.__setattr__(self, "_inner_of", {outer: inner for inner, outer in nesting.items()})

    def __call__(self, record: Mapping[str, float]) -> float:
        """The log-probability of `record`'s drawn concepts, minus infinity if undrawable."""
        mapped, following, inner_of = self._mapped, self._following, self._inner_of
        shows = {
            concept: any(row.applies(record) for row in concept_rows)
            for concept, concept_rows in mapped.items()
        }
        score = 0.0
        for concept in mapped:
            if concept in following:
                continue
            inner = inner_of.get(concept)
            share = self.shares.get(concept, 0.0)
            if inner is not None:
                inner_share = self.shares.get(inner, 0.0)
                if shows[inner] or inner_share == 1:
                    if not shows[concept]:
                        return -math.inf
                    continue
                share = (share - inner_share) / (1 - inner_share)
            chance = share if shows[concept] else 1 - share
            if chance <= 0:
                return -math.inf
            score += math.log(chance)
        return score
