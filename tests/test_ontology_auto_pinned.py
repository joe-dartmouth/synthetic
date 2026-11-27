"""The committed automatic mapping modules against the pinned releases under the data root.

Each `auto-lexical` module is a rebuild's text, each `auto-local` table's rows are
`rows_from_verdicts` of its committed verdicts on the records-built frame, each verdict table's
keys are its candidates and cross-referenced states with every verdict one its state could take,
and `bko.ontology.check.check` accepts every module's rows (ADR
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
from bko.ontology import auto_lexical_codes_rows, auto_lexical_diabetes_rows
from bko.ontology.auto_tables import (
    AUTO_LEXICAL_CODES_ROWS,
    AUTO_LEXICAL_DIABETES_ROWS,
    AUTO_LOCAL_CODES_ROWS,
    AUTO_LOCAL_DIABETES_ROWS,
    CODES_LABELS,
    CODES_VERDICTS,
    DIABETES_LABELS,
    DIABETES_VERDICTS,
)
from bko.ontology.automatic import (
    NONE,
    TOP,
    build_lexical,
    render,
    rows_from_verdicts,
    state,
    survey,
)
from bko.ontology.automatic_inputs import inputs
from bko.ontology.check import check
from bko.ontology.code_frames import DIABETES_CODES_NAME
from bko.ontology.frames import DIABETES_NAME
from bko.ontology.schema import FieldMapping
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]

ROOT: Final = Path(__file__).resolve().parent.parent / "src" / "bko" / "ontology"

LEXICAL: Final = {
    DIABETES_NAME: ROOT / Path(auto_lexical_diabetes_rows.__file__ or "").name,
    DIABETES_CODES_NAME: ROOT / Path(auto_lexical_codes_rows.__file__ or "").name,
}
"""Each dataset's committed `auto-lexical` module."""

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


VERDICTS: Final = {
    DIABETES_NAME: (DIABETES_VERDICTS, DIABETES_LABELS),
    DIABETES_CODES_NAME: (CODES_VERDICTS, CODES_LABELS),
}
"""Each dataset's committed verdict table and labels."""


def test_each_local_tables_rows_are_its_verdicts_rows_on_the_records_frame():
    frame, _, _ = _inputs(DIABETES_NAME)
    assert rows_from_verdicts(frame, *VERDICTS[DIABETES_NAME]) == AUTO_LOCAL_DIABETES_ROWS
    frame, _, _ = _inputs(DIABETES_CODES_NAME)
    assert rows_from_verdicts(frame, *VERDICTS[DIABETES_CODES_NAME]) == AUTO_LOCAL_CODES_ROWS


@pytest.mark.parametrize("dataset", sorted(VERDICTS))
def test_each_verdict_is_an_offered_option_or_none_of_a_surveyed_state(dataset: str):
    frame, ontologies, codes = _inputs(dataset)
    found = survey(frame, ontologies, codes)
    verdicts, _ = VERDICTS[dataset]
    crossed = {
        "{}={}".format(*state(row)): row.concept for row in found.rows if state(row)[0] in codes
    }
    offered = {c.key: {o.curie for o in c.options[:TOP]} | {NONE} for c in found.candidates}
    assert crossed.keys().isdisjoint(offered)
    assert set(verdicts) == set(crossed) | set(offered)
    assert {key: verdicts[key] for key in crossed} == crossed
    assert {key: verdicts[key] for key in offered if verdicts[key] not in offered[key]} == {}


@pytest.mark.parametrize("table", sorted(TABLES))
def test_the_check_accepts_each_tables_rows(table: str):
    _, ontologies, _ = _inputs(DIABETES_NAME)
    verdicts = check(TABLES[table], ontologies)
    assert len(verdicts) == len(TABLES[table])
    assert [f"{v.concept}: {v.problem}" for v in verdicts if v.problem is not None] == []
