"""The mapping table's invariants against the codebook, the registry and the ADRs; no pronto."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from bko.datasets import sources
from bko.datasets.ablation import DEFAULT_ABLATION
from bko.datasets.cells import LABEL
from bko.datasets.sources import Source
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE
from bko.datasets.uci import CATEGORICAL, CODEBOOK, NUMERIC
from bko.ontology.mapping import (
    CODE_MAPPINGS,
    CONCEPT_ROWS,
    FIELD_MAPPINGS,
    MAPPED,
    OUT_OF_SCOPE,
    RANGES,
    RELEASES,
    UNMAPPED,
)
from bko.ontology.schema import AtLeast, Codes
from tests._corpus import REPO_ROOT
from tests._registry import REGISTRY, read, registry_entries


LABEL_CODES = frozenset({0, 1})


def _codebook(field: str) -> frozenset[int]:
    return LABEL_CODES if field == LABEL else frozenset(CODEBOOK[field])


def _known_state(field: str, codes: frozenset[int]) -> bool:
    return (field in CATEGORICAL or field == LABEL) and bool(codes) and codes <= _codebook(field)


def _codes_clauses() -> list[Codes]:
    return [clause for row in FIELD_MAPPINGS for clause in row.when if isinstance(clause, Codes)]


def test_every_categorical_state_is_accounted_for() -> None:
    states = [(field, code) for field in CATEGORICAL for code in CODEBOOK[field]]
    states += [(LABEL, code) for code in sorted(LABEL_CODES)]
    for field, code in states:
        mapped = any(c.field == field and code in c.codes for c in _codes_clauses())
        unmapped = sum(row.field == field and code in row.codes for row in UNMAPPED)
        assert mapped != (unmapped == 1), (field, code)
        assert unmapped <= 1, (field, code)


def test_clauses_name_known_fields_and_codes() -> None:
    for row in FIELD_MAPPINGS:
        assert row.when, row.concept
        fields = [clause.field for clause in row.when]
        assert len(fields) == len(set(fields)), row.concept
        for clause in row.when:
            if isinstance(clause, Codes):
                assert _known_state(clause.field, clause.codes), (row.concept, clause)
            else:
                assert clause.field in NUMERIC, (row.concept, clause)


def test_unmapped_rows_name_known_states() -> None:
    for row in UNMAPPED:
        assert _known_state(row.field, row.codes), row


def test_every_numeric_field_has_one_range() -> None:
    assert sorted(r.field for r in RANGES) == sorted(NUMERIC)
    for r in RANGES:
        assert math.isfinite(r.low), r
        assert math.isfinite(r.high), r
        assert r.low < r.high, r
        assert r.unit.strip(), r


def test_thresholds_sit_inside_their_ranges() -> None:
    bounds = {r.field: (r.low, r.high) for r in RANGES}
    for row in FIELD_MAPPINGS:
        for clause in row.when:
            if isinstance(clause, AtLeast):
                low, high = bounds[clause.field]
                assert low <= clause.value <= high, (row.concept, clause)


def test_applies_needs_every_clause() -> None:
    [row] = [r for r in FIELD_MAPPINGS if r.label == "ST segment depression"]
    assert row.applies({"oldpeak": 1.0, "slope": 2})
    assert not row.applies({"oldpeak": 0.9, "slope": 2})
    assert not row.applies({"oldpeak": 2.0, "slope": 1})


def test_every_removed_synthea_code_has_a_row() -> None:
    snomed = [c.snomed for c in CODE_MAPPINGS]
    assert DEFAULT_ABLATION.codes | {ISCHEMIC_HEART_DISEASE} <= set(snomed)
    assert len(snomed) == len(set(snomed))
    assert all(c.concept.startswith("MONDO:") for c in CODE_MAPPINGS)


def _well_formed(curie: str) -> bool:
    return (
        re.fullmatch(r"(MONDO|HP):[0-9]{7}", curie, re.ASCII) is not None
        and curie.partition(":")[0] in RELEASES
    )


def test_concept_ids_are_well_formed() -> None:
    for row in CONCEPT_ROWS:
        assert _well_formed(row.concept), row.concept
    for bad in ("MONDO:024644", "mondo:0024644", "MONDO:0024644\n", "DOID:0050000"):
        assert not _well_formed(bad), bad


def test_no_concept_is_both_mapped_and_out_of_scope() -> None:
    assert MAPPED.isdisjoint(r.concept for r in OUT_OF_SCOPE)


def _resolves(source: str) -> bool:
    names = {value.name for value in vars(sources).values() if isinstance(value, Source)}
    keys = {key for _, key, _ in registry_entries(read(REGISTRY))}
    if source in names or source in keys:
        return True
    return bool(source) and (REPO_ROOT / "docs" / "adr" / f"{source}.md").is_file()


def _all_sources() -> Iterable[str]:
    for table in (FIELD_MAPPINGS, UNMAPPED, RANGES, CODE_MAPPINGS, OUT_OF_SCOPE):
        for row in table:
            yield row.source


def test_every_source_resolves() -> None:
    for source in _all_sources():
        assert _resolves(source), source
    assert not _resolves("")
    assert not _resolves("x")


def test_every_row_explains_itself() -> None:
    for row in FIELD_MAPPINGS:
        assert row.label.strip(), row
        assert row.limits.strip(), row
    for row in UNMAPPED:
        assert row.reason.strip(), row
    for row in RANGES:
        assert row.limits.strip(), row
    for row in CODE_MAPPINGS:
        assert row.label.strip(), row
    for row in OUT_OF_SCOPE:
        assert row.label.strip(), row
        assert row.reason.strip(), row
    for table in (FIELD_MAPPINGS, UNMAPPED, RANGES, CODE_MAPPINGS, OUT_OF_SCOPE):
        assert len(table) == len(set(table)), table
