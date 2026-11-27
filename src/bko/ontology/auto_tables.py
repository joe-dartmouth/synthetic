"""The automatic and permuted mapping tables E22's dataset variants read.

Each `auto-lexical` table is its committed module's `ROWS`; each `auto-local` table is
`bko.ontology.automatic.rows_from_verdicts` of its committed `VERDICTS` and `LABELS`, the two
modules of `diabetes-codes`' split table merged; each permuted table is the hand table under
`bko.ontology.automatic.permuted` at `PERMUTED_SEED` (ADR `research/automatic-mapping`).
`diabetes-codes`' frame is built from its records, so its verdict table is read on
`verdict_frame`, the frame of the codes the table names, whose meanings are the records-built
frame's for those codes.
"""

from __future__ import annotations

from collections.abc import Mapping
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
from bko.ontology.code_frames import DIABETES_CODES_FIELD_ROWS, diabetes_codes_frame
from bko.ontology.frames import DIABETES_FIELD_ROWS, DIABETES_FRAME
from bko.ontology.schema import FieldMapping


CODES_VERDICTS: Final[Mapping[str, str]] = {
    **auto_local_codes_rows_low.VERDICTS,
    **auto_local_codes_rows_high.VERDICTS,
}
"""`diabetes-codes`' verdict table: its `low` and `high` parts merged."""

CODES_LABELS: Final[Mapping[str, str]] = {
    **auto_local_codes_rows_low.LABELS,
    **auto_local_codes_rows_high.LABELS,
}
"""Each concept of `CODES_VERDICTS` with its name."""

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
    DIABETES_FRAME, auto_local_diabetes_rows.VERDICTS, auto_local_diabetes_rows.LABELS
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
