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
from bko.ontology.frames import DIABETES_FIELD_ROWS, DIABETES_FRAME
from bko.ontology.schema import Codes
from tests._diabetes_fixtures import pin, toy_archive
from tests._ontology_fixtures import RoutingOpener


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
