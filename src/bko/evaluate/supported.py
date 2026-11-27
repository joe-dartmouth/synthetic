"""`supported_frame`: a frame projected onto the fields its rows read.

`bko.evaluate.data` builds each supported variant's frame with it (ADR `research/dataset-variants`).
"""

from __future__ import annotations

from collections.abc import Sequence

from bko.datasets.frames import FrameSpec
from bko.ontology.schema import Codes, FieldMapping


def supported_frame(frame: FrameSpec, rows: Sequence[FieldMapping]) -> FrameSpec:
    """`frame` on `age`, `sex`, the label and every field a `Codes` clause of `rows` reads.

    Its name is `frame`'s with `-supported` after it; every other value is `frame`'s, kept for
    the columns it keeps.
    """
    coded = {c.field for row in rows for c in row.when if isinstance(c, Codes)}
    kept = {"age", "sex", frame.label} | coded

    def keep(names: Sequence[str]) -> tuple[str, ...]:
        return tuple(name for name in names if name in kept)

    return FrameSpec(
        name=f"{frame.name}-supported",
        features=keep(frame.features),
        numeric=keep(frame.numeric),
        fractional=keep(frame.fractional),
        fraction_digits=frame.fraction_digits,
        categorical=keep(frame.categorical),
        codebook={name: codes for name, codes in frame.codebook.items() if name in kept},
        descriptions={name: text for name, text in frame.descriptions.items() if name in kept},
        ranges=tuple(bound for bound in frame.ranges if bound.field in kept),
        label_codes=frame.label_codes,
        label_meanings=frame.label_meanings,
        edges=frame.edges,
        study=frame.study,
        cohort=frame.cohort,
        label=frame.label,
    )
