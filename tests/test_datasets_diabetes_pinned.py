"""The Diabetes 130 loader against the pinned archive under the data root.

Every other Diabetes test reads an archive built in `tmp_path`, so a change that alters how the real
archive parses passes `make check`. These cases load the snapshot from the data root and are
skipped unless `--releases` is passed, since the gate reads no `data/`. An absent snapshot fails
them rather than skipping them. `docs/DEVELOPMENT.md` (Data, The Diabetes frame) says when to run
them and what fetches the snapshot.
"""

from __future__ import annotations

import dataclasses
import functools
import io
import zipfile

import pandas as pd
import pytest

from bko.datasets.cells import LABEL
from bko.datasets.diabetes import (
    COLUMNS,
    DRUGS,
    EXCLUDED_DISCHARGES,
    IDS_MEMBER,
    MEMBER,
    excluded_discharges,
    load_diabetes,
    parse_diabetic,
)
from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.sources import DIABETES_130
from bko.datasets.split import split
from bko.evaluate.diabetes import DIABETES_FRAME, DIABETES_INDICATORS
from bko.evaluate.experiment import E1, Experiment, Removal, flagged, gap_request
from bko.evaluate.named_runs import RUNS
from bko.evaluate.perturb import incoherent
from bko.evaluate.prepare import prepared


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _records() -> pd.DataFrame:
    return load_diabetes(default_data_dir(), indicators=DIABETES_INDICATORS)


@functools.cache
def _member(name: str) -> str:
    data = read_verified(DIABETES_130, default_data_dir())
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        return bundle.read(name).decode("utf-8")


def test_the_pinned_archive_loads_its_cohort() -> None:
    records = _records()
    assert len(records) == 69283
    assert int(records[LABEL].sum()) == 6248
    assert tuple(records.columns) == DIABETES_FRAME.columns


@pytest.mark.parametrize("bound", DIABETES_FRAME.ranges, ids=lambda bound: bound.field)
def test_every_frame_range_holds_the_cohorts_least_and_greatest_value(bound) -> None:
    values = _records()[bound.field]
    assert bound.low <= values.min()
    assert values.max() <= bound.high


def test_the_mappings_expired_and_hospice_discharges_are_the_pinned_set() -> None:
    assert excluded_discharges(_member(IDS_MEMBER)) == EXCLUDED_DISCHARGES


def test_every_real_training_row_keeps_the_three_rules() -> None:
    """The cohort holds every holdout's training split, so no real training row breaks a rule."""
    assert incoherent(_records()) == pytest.approx(0.0)


def test_the_drug_columns_read_are_the_eight_most_often_prescribed() -> None:
    raw = parse_diabetic(_member(MEMBER))
    drugs = COLUMNS[COLUMNS.index("metformin") : COLUMNS.index("change")]
    assert len(drugs) == len(DRUGS) + 15
    share = {name: float((raw[name] != "No").mean()) for name in drugs}
    assert tuple(sorted(drugs, key=lambda name: -share[name])[: len(DRUGS)]) == DRUGS


CELL = ("6-7", "male", 1)
"""E6's removed cell (ADR `research/diabetes-subgroup-removal-design`)."""


def _experiment(fraction: float) -> Experiment:
    """E6's configuration at `fraction`: the design's threshold, fill target, cutoff and copies."""
    return dataclasses.replace(
        E1,
        holdout_seeds=tuple(range(100)),
        threshold=20,
        fill_to=1269,
        cutoff=0.0902,
        decontaminate=False,
        removal=Removal(CELL, fraction),
    )


def test_the_drug_constant_is_the_eight_drugs_the_frame_reads_most_often_first():
    assert DRUGS == (
        "insulin",
        "metformin",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "glimepiride",
        "repaglinide",
    )


def test_the_indicator_set_is_the_ten_icd9_rows_in_mapping_order():
    assert tuple(DIABETES_INDICATORS) == (
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
    )


@pytest.mark.parametrize("fraction", [0.985, 1.0])
def test_the_removed_cell_is_the_only_flagged_cell_on_every_holdout_seed(fraction):
    experiment = _experiment(fraction)
    edges = DIABETES_FRAME.edges
    for seed in experiment.holdout_seeds:
        parts = split(_records(), seed=seed, edges=edges)
        train = prepared(parts, experiment, edges=edges).ablated.train
        assert flagged(train, experiment, edges=edges) == (CELL,), seed


@pytest.mark.parametrize(
    ("name", "asked"),
    [
        ("e10-pilot-thin", 1250),
        ("e10-pilot-empty", 1269),
        ("e10-pram-thin", 1250),
        ("e10-pram-empty", 1269),
    ],
)
def test_each_perturbation_run_requests_only_the_removed_cell(name, asked):
    experiment = RUNS[name].experiment
    edges = DIABETES_FRAME.edges
    for seed in experiment.holdout_seeds:
        parts = split(_records(), seed=seed, edges=edges)
        train = prepared(parts, experiment, edges=edges).ablated.train
        request = gap_request(train, experiment, seed=seed, frame=DIABETES_FRAME)
        assert dict(request.quota) == {CELL: asked}, seed
