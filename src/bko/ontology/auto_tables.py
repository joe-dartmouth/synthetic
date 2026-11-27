"""The automatic and permuted mapping tables E22's dataset variants read.

Each `auto-lexical` table is its committed module's `ROWS`; each `auto-local` table is
`bko.ontology.automatic.rows_from_verdicts` of its committed `VERDICTS` and `LABELS`, the two
modules of `diabetes-codes`' split table merged by `merged`, which refuses parts that overlap,
disagree on a label or name another dataset or part; each permuted table is the hand table under
`bko.ontology.automatic.permuted` at `PERMUTED_SEED` (ADR `research/automatic-mapping`).
`diabetes-codes`' frame is built from its records, so its verdict table is read on
`verdict_frame`, the frame of the codes the table names, whose meanings are the records-built
frame's for those codes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Final

import pandas as pd

from bko.datasets.diabetes import DIAGNOSES
from bko.datasets.frames import FrameSpec
from bko.ontology import (
    auto_lexical_codes_rows,
    auto_lexical_diabetes_rows,
    auto_local_codes_rows_high,
    auto_local_codes_rows_low,
    auto_local_diabetes_rows,
)
from bko.ontology.automatic import permuted, rows_from_verdicts
from bko.ontology.code_frames import (
    DIABETES_CODES_FIELD_ROWS,
    DIABETES_CODES_NAME,
    diabetes_codes_frame,
)
from bko.ontology.frames import DIABETES_FIELD_ROWS, DIABETES_FRAME, DIABETES_NAME
from bko.ontology.schema import FieldMapping


PARTS: Final = ("low", "high")
"""A split verdict table's parts, in the order their modules are merged."""


def merged(
    modules: Sequence[ModuleType], *, dataset: str, parts: Sequence[str | None]
) -> tuple[dict[str, str], dict[str, str]]:
    """The `VERDICTS` and `LABELS` of `modules`, one table's parts, merged.

    Args:
        modules: The parts' verdict modules.
        dataset: The `DATASET` each module must name.
        parts: The `PART` each module must name, in `modules`' order; None for an unsplit table,
            whose module names none.

    Raises:
        ValueError: If a module names another dataset or part, two parts share a state, or two
            parts name one concept differently.
    """
    verdicts: dict[str, str] = {}
    labels: dict[str, str] = {}
    for module, part in zip(modules, parts, strict=True):
        named = (getattr(module, "DATASET", None), getattr(module, "PART", None))
        if named != (dataset, part):
            msg = f"{module.__name__} names dataset and part {named}, not {(dataset, part)}"
            raise ValueError(msg)
        shared = sorted(verdicts.keys() & module.VERDICTS.keys())
        clashing = sorted(
            c for c in labels.keys() & module.LABELS.keys() if labels[c] != module.LABELS[c]
        )
        if shared or clashing:
            msg = f"{module.__name__} repeats states {shared[:5]} or relabels {clashing[:5]}"
            raise ValueError(msg)
        verdicts.update(module.VERDICTS)
        labels.update(module.LABELS)
    return verdicts, labels


CODES_VERDICTS, CODES_LABELS = merged(
    (auto_local_codes_rows_low, auto_local_codes_rows_high),
    dataset=DIABETES_CODES_NAME,
    parts=PARTS,
)
"""`diabetes-codes`' verdict table, its `low` and `high` parts merged, and each verdict concept
with its name."""

DIABETES_VERDICTS, DIABETES_LABELS = merged(
    (auto_local_diabetes_rows,), dataset=DIABETES_NAME, parts=(None,)
)
"""`diabetes`' verdict table and each verdict concept with its name."""

PERMUTED_SEED: Final = 0
"""The seed each permuted table's concepts are moved at."""


def verdict_frame(verdicts: Mapping[str, str]) -> FrameSpec:
    """`diabetes_codes_frame` of the diagnosis codes `verdicts`' keys name, in every column."""
    named = sorted(
        {int(key.partition("=")[2]) for key in verdicts if key.partition("=")[0] in DIAGNOSES}
    )
    return diabetes_codes_frame(pd.DataFrame(dict.fromkeys(DIAGNOSES, named)))


AUTO_LEXICAL_DIABETES_ROWS: Final[tuple[FieldMapping, ...]] = auto_lexical_diabetes_rows.ROWS
"""`diabetes`' `auto-lexical` rows."""

AUTO_LOCAL_DIABETES_ROWS: Final[tuple[FieldMapping, ...]] = rows_from_verdicts(
    DIABETES_FRAME, DIABETES_VERDICTS, DIABETES_LABELS
)
"""`diabetes`' `auto-local` rows."""

PERMUTED_DIABETES_ROWS: Final[tuple[FieldMapping, ...]] = permuted(
    DIABETES_FIELD_ROWS, seed=PERMUTED_SEED
)
"""`diabetes`' hand rows with their concepts permuted."""

AUTO_LEXICAL_CODES_ROWS: Final[tuple[FieldMapping, ...]] = auto_lexical_codes_rows.ROWS
"""`diabetes-codes`' `auto-lexical` rows."""

AUTO_LOCAL_CODES_ROWS: Final[tuple[FieldMapping, ...]] = rows_from_verdicts(
    verdict_frame(CODES_VERDICTS), CODES_VERDICTS, CODES_LABELS
)
"""`diabetes-codes`' `auto-local` rows."""

PERMUTED_CODES_ROWS: Final[tuple[FieldMapping, ...]] = permuted(
    DIABETES_CODES_FIELD_ROWS, seed=PERMUTED_SEED
)
"""`diabetes-codes`' hand rows with their concepts permuted."""
