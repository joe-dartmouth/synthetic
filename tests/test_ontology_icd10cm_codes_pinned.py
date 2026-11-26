"""The committed ICD-10-CM rows against the pinned MONDO release and NHAMCS archive.

`tests/test_ontology_icd10cm.py` checks the rows against stanzas built in `tmp_path`, so a new
release or a changed generator passes `make check`. These cases regenerate the rows from the
pinned release and the `nhamcs` records under the data root and are skipped unless `--releases`
is passed, since the gate reads no `data/`. An absent release or snapshot fails them rather than
skipping them. ADR `research/nhamcs-frame` records the counts.
"""

from __future__ import annotations

import functools
from collections import Counter
from pathlib import Path

import pytest

from bko.datasets.fetch import default_data_dir
from bko.datasets.icd10_tokens import UNKNOWN, token_text
from bko.datasets.nhamcs import DIAGNOSES, load_nhamcs
from bko.datasets.sources import NHAMCS_ED_2022
from bko.ontology.icd9 import EXACT
from bko.ontology.icd10 import Icd10Code, generate_icd10, render_icd10
from bko.ontology.icd10cm_codes import ICD10CM_CODES
from bko.ontology.mapping import RELEASES
from bko.ontology.obo import load_ontology
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]

UNMAPPED = {
    "no live xref": 1920,
    "obsolete holder only": 0,
    "relatedTo only": 1,
    "unreviewed source only": 11,
    "holders not nested": 0,
    "not ICD-10-CM": 4,
}
"""The unmapped tallies ADR `research/nhamcs-frame` states for the pinned release."""


@functools.cache
def _mondo() -> Ontology:
    return load_ontology(RELEASES["MONDO"], default_data_dir())


@functools.cache
def _tokens() -> tuple[str, ...]:
    records = load_nhamcs(default_data_dir(), sources=(NHAMCS_ED_2022,))
    held = {int(number) for name in DIAGNOSES for number in records[name].unique()}
    return tuple(token_text(number) for number in sorted(held - {UNKNOWN}))


@functools.cache
def _generated() -> tuple[list[Icd10Code], Counter[str]]:
    return generate_icd10(_mondo(), _tokens())


def test_the_committed_module_is_the_generator_output_on_the_pinned_release() -> None:
    rows, _ = _generated()
    assert tuple(rows) == ICD10CM_CODES
    committed = Path(__file__).parents[1] / "src/bko/ontology/icd10cm_codes.py"
    assert render_icd10(rows) == committed.read_text()


def test_the_token_and_unmapped_counts_are_the_adrs() -> None:
    rows, unmapped = _generated()
    assert (len(_tokens()), len(rows)) == (2114, 178)
    assert dict(unmapped) == UNMAPPED


def test_every_committed_concept_is_live_named_as_labelled_and_holds_its_xref_exactly() -> None:
    terms = _mondo().terms
    for row in ICD10CM_CODES:
        term = terms[row.concept]
        assert not term.obsolete, row
        assert term.name == row.label, row
        assert EXACT in term.xref_sources(row.xref), row
