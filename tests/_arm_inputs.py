"""Each control arm's declared inputs: what its fill reads, of seven inputs.

ADR `research/matched-knowledge-controls`, Decision 14, defines the inputs and holds this table as
prose; `tests/test_arm_inputs.py` checks the two agree and holds the draws to the table.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final


INPUTS: Final = (
    "cell",
    "label-group",
    "other-label",
    "mapping",
    "hierarchy",
    "codebook",
    "age-range",
)
"""The declared-input names, in the ADR table's column order."""

_RECORDS: Final = frozenset({"cell", "label-group"})
_ALL_RECORDS: Final = _RECORDS | {"other-label"}
_K: Final = frozenset({"mapping", "hierarchy"})

ARM_INPUTS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "none": frozenset(),
        "class-weight": frozenset(),
        "prior-shift": frozenset(),
        "real": frozenset({"cell"}),
        "smote": _ALL_RECORDS,
        "adasyn": _ALL_RECORDS | {"codebook"},
        "ctgan": _ALL_RECORDS,
        "tvae": _ALL_RECORDS,
        "bayes-net": _ALL_RECORDS | {"codebook"},
        "label-bootstrap": _RECORDS,
        "bko-direct": _RECORDS | {"mapping"},
        "bko-nomap": _RECORDS,
        "bko-logic": _RECORDS | _K,
        "bko-tree": _RECORDS | _K,
        "bko-donor": _RECORDS | {"mapping"},
        "bko-pool-random": _RECORDS | {"mapping"},
        "bko-pool-all": _RECORDS,
        "row-resample": _RECORDS,
        "marginals": _RECORDS,
        "eb-marginals": _RECORDS,
        "hb-marginals": _ALL_RECORDS,
        "bko-kb": _K | {"codebook", "age-range"},
    }
)
"""Each arm of E9, E12, E13, E15 and E19 to the inputs of `INPUTS` its fill reads."""
