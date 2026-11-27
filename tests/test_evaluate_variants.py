"""Dataset variants: a projection or a row swap of a base dataset under a new dataset name.

ADR `research/dataset-variants` fixes each rule. No test reaches the API or reads `data/`: the
archive is the toy cohort `toy_archive` builds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pytest

from bko.evaluate.context import Context
from bko.evaluate.data import (
    VARIANTS,
    Dataset,
    Variant,
    dataset_releases,
    digest,
    field_rows,
    frame,
    held_split,
    load,
    supported_frame,
)
from bko.evaluate.direct_arms import DIRECT_ARMS, HIER_W_ARM
from bko.evaluate.runs_diabetes import E6_CELL
from bko.generate.direct import plan_hierarchy, sample
from bko.ontology.auto_tables import (
    AUTO_LEXICAL_CODES_ROWS,
    AUTO_LEXICAL_DIABETES_ROWS,
    AUTO_LOCAL_CODES_ROWS,
    AUTO_LOCAL_DIABETES_ROWS,
    PERMUTED_CODES_ROWS,
    PERMUTED_DIABETES_ROWS,
)
from bko.ontology.code_frames import DIABETES_CODES_FIELD_ROWS
from bko.ontology.frames import DIABETES_FIELD_ROWS, DIABETES_FRAME
from bko.ontology.obo import parse_obo
from bko.ontology.schema import Codes, FieldMapping, rows_digest
from bko.ontology.terms import Ontology
from tests._diabetes_fixtures import diabetes_releases, pin, toy_archive
from tests._ontology_fixtures import RoutingOpener
from tests.test_evaluate_hierarchy import MONDO as HIER_MONDO


ARCHIVE: Final = toy_archive()
PIN: Final = pin(ARCHIVE, "uci-diabetes-130")
SUPPORTED_COLUMNS: Final = (
    "age",
    "sex",
    "A1Cresult",
    "max_glu_serum",
    "diabetic_kidney_disease",
    "essential_hypertension",
    "acute_myocardial_infarction",
    "heart_failure",
    "cerebral_artery_occlusion",
    "pneumonia",
    "chronic_bronchitis",
    "chronic_obstructive_pulmonary_disease",
    "chronic_kidney_disease",
    "chronic_ulcer_of_skin",
    "target",
)
"""`age`, `sex`, the label and every field a `Codes` clause of Diabetes 130's rows reads."""


def _context() -> Context:
    return Context(open_url=RoutingOpener({PIN.url: ARCHIVE}), diabetes=PIN)


def test_the_supported_variant_is_diabetes_projected():
    assert Dataset.DIABETES_SUPPORTED.value == "diabetes-supported"
    assert VARIANTS[Dataset.DIABETES_SUPPORTED] == Variant(Dataset.DIABETES, supported=True)


def test_the_supported_frame_keeps_the_axes_the_label_and_the_mapped_fields():
    spec = frame(Dataset.DIABETES_SUPPORTED)
    assert spec.columns == SUPPORTED_COLUMNS
    assert len(spec.columns) == 15
    assert len(DIABETES_FRAME.columns) == 34
    mapped = {c.field for row in DIABETES_FIELD_ROWS for c in row.when if isinstance(c, Codes)}
    assert set(spec.columns) == {"age", "sex", spec.label} | mapped
    assert spec.name == "diabetes-supported"
    assert spec.numeric == ("age",)
    assert spec.ranges == tuple(r for r in DIABETES_FRAME.ranges if r.field == "age")
    assert all(spec.codebook[f] == DIABETES_FRAME.codebook[f] for f in spec.categorical)
    assert (spec.edges, spec.label, spec.study) == (
        DIABETES_FRAME.edges,
        DIABETES_FRAME.label,
        DIABETES_FRAME.study,
    )
    assert supported_frame(DIABETES_FRAME, DIABETES_FIELD_ROWS).columns == SUPPORTED_COLUMNS


def test_the_supported_frame_is_one_object():
    assert frame(Dataset.DIABETES_SUPPORTED) is frame(Dataset.DIABETES_SUPPORTED)
    assert frame(Dataset.DIABETES) is DIABETES_FRAME


def test_the_supported_variant_keeps_its_bases_rows_without_data():
    context = Context(open_url=RoutingOpener({}))
    assert field_rows(Dataset.DIABETES_SUPPORTED, context) == DIABETES_FIELD_ROWS


@pytest.mark.parametrize("seed", [0, 5])
def test_the_supported_split_is_the_base_split(tmp_path: Path, seed: int):
    base = load(Dataset.DIABETES, tmp_path, _context())
    variant = load(Dataset.DIABETES_SUPPORTED, tmp_path, _context())
    one, two = held_split(base, seed), held_split(variant, seed)
    assert set(one.train.index) == set(two.train.index)
    assert set(one.test.index) == set(two.test.index)
    assert two.train.equals(one.train[list(SUPPORTED_COLUMNS)])


def test_a_variants_data_line_names_the_variant_and_its_frame(tmp_path: Path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.DIABETES_SUPPORTED, tmp_path, _context())
    assert data.frame is frame(Dataset.DIABETES_SUPPORTED)
    assert tuple(data.records.columns) == SUPPORTED_COLUMNS
    assert data.digest == digest(data.records)
    line = (
        f"data dataset=diabetes-supported frame=diabetes-supported records={len(data.records)} "
        f"digest={data.digest} agreement=none"
    )
    assert [m for m in caplog.messages if m.startswith("data ")] == [line]


def test_a_variant_keeps_its_bases_source_name_and_releases(tmp_path: Path):
    context = _context()
    base = load(Dataset.DIABETES, tmp_path, context)
    variant = load(Dataset.DIABETES_SUPPORTED, tmp_path, context)
    assert variant.name == base.name == PIN.name
    assert variant.field_rows == base.field_rows == DIABETES_FIELD_ROWS
    assert (variant.agreement, variant.held) == (base.agreement, base.held)
    assert variant.digest != base.digest
    assert dataset_releases(Dataset.DIABETES_SUPPORTED, context) == dataset_releases(
        Dataset.DIABETES, context
    )


def test_loading_a_variant_logs_no_base_line(tmp_path: Path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        load(Dataset.DIABETES_SUPPORTED, tmp_path, _context())
    assert not any(m.startswith("data dataset=diabetes ") for m in caplog.messages)


def test_loading_the_base_logs_its_line_unchanged(tmp_path: Path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.DIABETES, tmp_path, _context())
    line = (
        f"data dataset=diabetes frame=diabetes records={len(data.records)} "
        f"digest={digest(data.records)} agreement=none"
    )
    assert [m for m in caplog.messages if m.startswith("data ")] == [line]


def test_a_row_swap_variant_reads_its_bases_frame_and_its_own_rows(
    tmp_path: Path, monkeypatch, caplog
):
    rows = DIABETES_FIELD_ROWS[:3]
    swapped = {Dataset.DIABETES_SUPPORTED: Variant(Dataset.DIABETES, rows=rows)}
    monkeypatch.setattr("bko.evaluate.data.VARIANTS", swapped)
    context = _context()
    assert frame(Dataset.DIABETES_SUPPORTED) is DIABETES_FRAME
    assert field_rows(Dataset.DIABETES_SUPPORTED, context) == rows
    assert dataset_releases(Dataset.DIABETES_SUPPORTED, context) == context.releases
    base = load(Dataset.DIABETES, tmp_path, context)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.DIABETES_SUPPORTED, tmp_path, context)
    assert data.frame is DIABETES_FRAME
    assert data.field_rows == rows
    assert data.records.equals(base.records)
    assert (data.name, data.digest) == (base.name, base.digest)
    line = (
        f"data dataset=diabetes-supported frame=diabetes records={len(data.records)} "
        f"digest={data.digest} agreement=none"
    )
    assert [m for m in caplog.messages if m.startswith("data ")] == [line]


ROW_SWAPS: Final = {
    Dataset.DIABETES_AUTO_LEXICAL: (Dataset.DIABETES, AUTO_LEXICAL_DIABETES_ROWS),
    Dataset.DIABETES_AUTO_LOCAL: (Dataset.DIABETES, AUTO_LOCAL_DIABETES_ROWS),
    Dataset.DIABETES_PERMUTED: (Dataset.DIABETES, PERMUTED_DIABETES_ROWS),
    Dataset.DIABETES_CODES_AUTO_LEXICAL: (Dataset.DIABETES_CODES, AUTO_LEXICAL_CODES_ROWS),
    Dataset.DIABETES_CODES_AUTO_LOCAL: (Dataset.DIABETES_CODES, AUTO_LOCAL_CODES_ROWS),
    Dataset.DIABETES_CODES_PERMUTED: (Dataset.DIABETES_CODES, PERMUTED_CODES_ROWS),
}
"""E22's row-swap variants, each with its base and its rows (ADR `research/automatic-mapping`)."""

HAND_DIGESTS: Final = {
    Dataset.DIABETES: "62132c26d9ddab1f82257fcda72040425fe99f1f3b681770f01595d73fe63988",
    Dataset.DIABETES_CODES: "a83ae9a4cca6fa641f835910f4729c67e42e2d240fdc620c6f9dfe635d65d920",
}
"""`rows_digest` of each base's hand rows, which no variant changes."""


@pytest.mark.parametrize("variant", sorted(ROW_SWAPS))
def test_a_variant_loads_its_bases_records_with_its_own_rows(tmp_path: Path, variant: Dataset):
    base, rows = ROW_SWAPS[variant]
    assert VARIANTS[variant] == Variant(base, rows)
    context = _context()
    assert field_rows(variant, context) == rows
    assert dataset_releases(variant, context) == dataset_releases(base, context)
    one, two = load(base, tmp_path, context), load(variant, tmp_path, context)
    assert two.records.equals(one.records)
    assert (two.frame.name, two.frame.columns) == (one.frame.name, one.frame.columns)
    assert {f: dict(c) for f, c in two.frame.codebook.items()} == {
        f: dict(c) for f, c in one.frame.codebook.items()
    }
    assert (two.name, two.digest, two.agreement) == (one.name, one.digest, one.agreement)
    assert two.field_rows == rows


def test_the_hand_datasets_rows_are_unchanged():
    context = Context(open_url=RoutingOpener({}))
    assert field_rows(Dataset.DIABETES, context) == DIABETES_FIELD_ROWS
    assert field_rows(Dataset.DIABETES_CODES, context) == DIABETES_CODES_FIELD_ROWS
    assert {name: rows_digest(field_rows(name, context)) for name in HAND_DIGESTS} == HAND_DIGESTS
    assert PERMUTED_DIABETES_ROWS != DIABETES_FIELD_ROWS
    assert PERMUTED_CODES_ROWS != DIABETES_CODES_FIELD_ROWS


@pytest.mark.parametrize("variant", sorted(ROW_SWAPS))
def test_a_variant_logs_its_mapping_line(tmp_path: Path, caplog, variant: Dataset):
    rows = ROW_SWAPS[variant][1]
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        load(variant, tmp_path, _context())
    line = f"mapping dataset={variant} rows={len(rows)} sha256={rows_digest(rows)}"
    assert [m for m in caplog.messages if m.startswith("mapping ")] == [line]


def test_a_base_or_a_supported_variant_logs_no_mapping_line(tmp_path: Path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        load(Dataset.DIABETES, tmp_path, _context())
        load(Dataset.DIABETES_SUPPORTED, tmp_path, _context())
    assert not any(m.startswith("mapping ") for m in caplog.messages)


def _toy_ontologies(tmp_path: Path) -> dict[str, Ontology]:
    pins, bodies = diabetes_releases()
    (tmp_path / "hp.obo").write_bytes(bodies[pins["HP"].url])
    (tmp_path / "mondo.obo").write_text(HIER_MONDO, encoding="utf-8")
    return {
        "HP": parse_obo(tmp_path / "hp.obo", name="hp"),
        "MONDO": parse_obo(tmp_path / "mondo.obo", name="mondo"),
    }


def test_a_permuted_variant_changes_a_hierarchy_reading_arms_fill(tmp_path: Path):
    data = load(Dataset.DIABETES_PERMUTED, tmp_path / "data", _context())
    ontologies = _toy_ontologies(tmp_path)
    spec = DIRECT_ARMS[HIER_W_ARM]

    def fill(rows: tuple[FieldMapping, ...]) -> list[dict[str, float]]:
        assert plan_hierarchy(rows, ontologies, data.frame, spec) is not None
        drawn = sample(
            data.records,
            {E6_CELL: 40},
            rows=rows,
            ontologies=ontologies,
            name=data.name,
            seed=0,
            frame=data.frame,
            tag=(0, "gap"),
            spec=spec,
        )
        return [dict(draw.record) for draw in drawn[E6_CELL]]

    assert data.field_rows == PERMUTED_DIABETES_ROWS
    assert fill(PERMUTED_DIABETES_ROWS) != fill(DIABETES_FIELD_ROWS)
