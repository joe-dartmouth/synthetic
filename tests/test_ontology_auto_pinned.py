"""The committed automatic mapping modules against the pinned releases under the data root.

Each `auto-lexical` module is a rebuild's text, each module's rows hash to their stated digest,
each `auto-local` table's rows are `rows_from_verdicts` of its committed verdicts on the
records-built frame, and `bko.ontology.check.check` accepts every module's rows (ADR
`research/automatic-mapping`). It is skipped unless `--releases` is passed, since the gate reads no
`data/`; an absent snapshot or release fails it rather than skipping it.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import BinaryIO, Final

import pytest

from bko.datasets.fetch import default_data_dir
from bko.datasets.frames import FrameSpec
from bko.ontology import (
    auto_lexical_codes_rows,
    auto_lexical_diabetes_rows,
    auto_local_diabetes_rows,
)
from bko.ontology.auto_tables import (
    AUTO_LEXICAL_CODES_ROWS,
    AUTO_LEXICAL_DIABETES_ROWS,
    AUTO_LOCAL_CODES_ROWS,
    AUTO_LOCAL_DIABETES_ROWS,
    CODES_LABELS,
    CODES_VERDICTS,
)
from bko.ontology.automatic import build_lexical, render, rows_from_verdicts
from bko.ontology.automatic_inputs import inputs
from bko.ontology.check import check
from bko.ontology.code_frames import DIABETES_CODES_NAME
from bko.ontology.frames import DIABETES_NAME
from bko.ontology.schema import FieldMapping, rows_digest
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]

ROOT: Final = Path(__file__).resolve().parent.parent / "src" / "bko" / "ontology"

LEXICAL: Final = {
    DIABETES_NAME: ROOT / Path(auto_lexical_diabetes_rows.__file__ or "").name,
    DIABETES_CODES_NAME: ROOT / Path(auto_lexical_codes_rows.__file__ or "").name,
}
"""Each dataset's committed `auto-lexical` module."""

DIGESTS: Final = {
    "auto-lexical diabetes": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "auto-lexical diabetes-codes": (
        "bf119a4d05428defc250c070b551d232f6a6d5f8a2421d84d01490fcd606b5c2"
    ),
    "auto-local diabetes": "3b71bbe6e3f5cd8cf15965252a6b3afb60a5dc81022fb5855fa0e57369115e7a",
    "auto-local diabetes-codes": "45caeb10a710164738fb6f28eccbd811dbcd78e991c9fcb0ac8f5da3293a3a63",
}
"""Each table's `rows_digest`, as ADR `research/automatic-mapping` states it."""

TABLES: Final[dict[str, tuple[FieldMapping, ...]]] = {
    "auto-lexical diabetes": AUTO_LEXICAL_DIABETES_ROWS,
    "auto-lexical diabetes-codes": AUTO_LEXICAL_CODES_ROWS,
    "auto-local diabetes": AUTO_LOCAL_DIABETES_ROWS,
    "auto-local diabetes-codes": AUTO_LOCAL_CODES_ROWS,
}


def _no_fetch(url: str, timeout: float) -> BinaryIO:
    msg = f"a release test fetched {url} ({timeout} s)"
    raise AssertionError(msg)


@functools.cache
def _inputs(dataset: str) -> tuple[FrameSpec, dict[str, Ontology], tuple[str, ...]]:
    return inputs(dataset, default_data_dir(), open_url=_no_fetch)


@pytest.mark.parametrize("dataset", [DIABETES_NAME, DIABETES_CODES_NAME])
def test_each_lexical_module_is_a_rebuild(dataset: str):
    frame, ontologies, codes = _inputs(dataset)
    text = render(build_lexical(frame, ontologies, codes=codes))
    assert LEXICAL[dataset].read_text(encoding="utf-8") == text


@pytest.mark.parametrize("table", sorted(TABLES))
def test_each_tables_rows_hash_to_their_stated_digest(table: str):
    assert rows_digest(TABLES[table]) == DIGESTS[table]


def test_each_local_tables_rows_are_its_verdicts_rows_on_the_records_frame():
    frame, _, _ = _inputs(DIABETES_NAME)
    diabetes = auto_local_diabetes_rows
    assert rows_from_verdicts(frame, diabetes.VERDICTS, diabetes.LABELS) == AUTO_LOCAL_DIABETES_ROWS
    frame, _, _ = _inputs(DIABETES_CODES_NAME)
    assert rows_from_verdicts(frame, CODES_VERDICTS, CODES_LABELS) == AUTO_LOCAL_CODES_ROWS


@pytest.mark.parametrize("table", sorted(TABLES))
def test_the_check_accepts_each_tables_rows(table: str):
    _, ontologies, _ = _inputs(DIABETES_NAME)
    verdicts = check(TABLES[table], ontologies)
    assert len(verdicts) == len(TABLES[table])
    assert [f"{v.concept}: {v.problem}" for v in verdicts if v.problem is not None] == []
