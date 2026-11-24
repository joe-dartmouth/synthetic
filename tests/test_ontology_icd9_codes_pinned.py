"""The committed ICD-9 code rows against the pinned MONDO release and Diabetes 130 archive.

`tests/test_ontology_codes.py` checks the rows against stanzas built in `tmp_path`, so a new
release or a changed generator passes `make check`. These cases regenerate the rows from the
pinned release and archive under the data root and are skipped unless `--releases` is passed,
since the gate reads no `data/`. An absent release or snapshot fails them rather than skipping
them. `docs/DEVELOPMENT.md` (Ontologies) says when to run them.
"""

from __future__ import annotations

import functools

import pytest

from bko.datasets.archive import member_text
from bko.datasets.diabetes import DIAGNOSES, MEMBER, parse_diabetic
from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.sources import DIABETES_130
from bko.ontology.codes import generate
from bko.ontology.icd9_codes import ICD9_CODES
from bko.ontology.mapping import RELEASES
from bko.ontology.obo import load_ontology
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _mondo() -> Ontology:
    return load_ontology(RELEASES["MONDO"], default_data_dir())


@functools.cache
def _tokens() -> frozenset[str]:
    data = read_verified(DIABETES_130, default_data_dir())
    raw = parse_diabetic(member_text(data, MEMBER, DIABETES_130))
    return frozenset(token for name in DIAGNOSES for token in raw[name])


def test_the_committed_rows_equal_the_generator_output_on_the_pinned_release() -> None:
    assert generate(_mondo(), _tokens()).rows == ICD9_CODES


def test_every_committed_concept_is_live() -> None:
    terms = _mondo().terms
    for row in ICD9_CODES:
        term = terms[row.concept]
        assert not term.obsolete, row
        assert term.name == row.label, row
        assert row.xref in term.xrefs, row
