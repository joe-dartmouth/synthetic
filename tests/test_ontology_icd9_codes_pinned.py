"""The committed ICD-9 code rows against the pinned MONDO release and Diabetes 130 archive.

`tests/test_ontology_codes.py` checks the rows against stanzas built in `tmp_path`, so a new release
or a changed generator passes `make check`. These cases regenerate the rows from the pinned release
and the `diabetes-codes` cohort of the archive under the data root and are skipped unless
`--releases` is passed, since the gate reads no `data/`. An absent release or snapshot fails them
rather than skipping them. `docs/DEVELOPMENT.md` (Ontologies) says when to run them.
"""

from __future__ import annotations

import functools

import pytest

from bko.datasets.diabetes import DIAGNOSES
from bko.datasets.diagnoses import code_text
from bko.datasets.fetch import default_data_dir
from bko.evaluate.context import Context
from bko.evaluate.data import Data, Dataset, load
from bko.ontology.code_frames import DIABETES_CODES_FIELD_ROWS
from bko.ontology.codes import Generated, generate
from bko.ontology.icd9 import EXACT
from bko.ontology.icd9_codes import ICD9_CODES
from bko.ontology.mapping import RELEASES
from bko.ontology.obo import load_ontology
from bko.ontology.schema import Codes
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]

UNMAPPED = {
    "no live xref": 555,
    "obsolete holder only": 2,
    "relatedTo only": 3,
    "unreviewed source only": 170,
    "holders not nested": 0,
    "V code": 47,
    "E code": 80,
}
"""The unmapped tallies ADR `research/diabetes-diagnosis-codes` states for the pinned release."""


@functools.cache
def _mondo() -> Ontology:
    return load_ontology(RELEASES["MONDO"], default_data_dir())


@functools.cache
def _cohort() -> Data:
    return load(Dataset.DIABETES_CODES, default_data_dir(), Context())


@functools.cache
def _generated() -> Generated:
    records = _cohort().records
    held = {int(number) for name in DIAGNOSES for number in records[name].unique()}
    return generate(_mondo(), map(code_text, sorted(held)))


def test_the_committed_rows_equal_the_generator_output_on_the_pinned_release() -> None:
    assert _generated().rows == ICD9_CODES


def test_the_unmapped_tallies_are_the_adrs() -> None:
    assert dict(_generated().unmapped) == UNMAPPED


def test_every_field_row_names_a_code_of_the_frames_codebook() -> None:
    frame = _cohort().frame
    for row in DIABETES_CODES_FIELD_ROWS:
        for clause in row.when:
            if isinstance(clause, Codes) and clause.field in DIAGNOSES:
                assert clause.codes <= set(frame.codebook[clause.field]), row


def test_every_committed_concept_is_live_named_as_labelled_and_holds_its_xref_exactly() -> None:
    terms = _mondo().terms
    for row in ICD9_CODES:
        term = terms[row.concept]
        assert not term.obsolete, row
        assert term.name == row.label, row
        assert EXACT in term.xref_sources(row.xref), row
