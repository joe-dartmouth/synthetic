"""Candidate records for `bko.validate`: a valid one, and one planted per check.

Values are written out by hand, not read from the codebook or the mapping table, so a check that
reads the wrong bound fails its planted record.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final


BASE: Final[Mapping[str, int | float]] = MappingProxyType(
    {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
        "target": 0,
    }
)
"""The first real Cleveland record as a decoded candidate, its `num` read as `target`."""


def planted(**changes: object) -> dict[str, object]:
    """`BASE` with `changes` applied."""
    return {**BASE, **changes}


PLANTED: Final[Mapping[str, Mapping[str, object]]] = MappingProxyType(
    {
        "fields": planted(num=0),
        **{f"number:{name}": planted(**{name: "1"}) for name in BASE},
        "range:age": planted(age=121),
        "precision:age": planted(age=63.5),
        "code:sex": planted(sex=2),
        "code:cp": planted(cp=0),
        "range:trestbps": planted(trestbps=251),
        "precision:trestbps": planted(trestbps=145.5),
        "range:chol": planted(chol=601),
        "precision:chol": planted(chol=233.5),
        "code:fbs": planted(fbs=2),
        "code:restecg": planted(restecg=3),
        "range:thalach": planted(thalach=501),
        "precision:thalach": planted(thalach=150.5),
        "code:exang": planted(exang=2),
        "range:oldpeak": planted(oldpeak=10.1),
        "precision:oldpeak": planted(oldpeak=2.35),
        "code:slope": planted(slope=0),
        "code:ca": planted(ca=4),
        "code:thal": planted(thal=4),
        "code:target": planted(target=2),
    }
)
"""One candidate per check, each failing that check alone."""
