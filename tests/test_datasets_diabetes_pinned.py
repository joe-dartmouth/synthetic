"""The Diabetes 130 loader against the pinned archive under the data root.

Every other Diabetes test reads an archive built in `tmp_path`, so a change that alters how the real
archive parses passes `make check`. These cases load the snapshot from the data root and are
skipped unless `--releases` is passed, since the gate reads no `data/`. An absent snapshot fails
them rather than skipping them. `docs/DEVELOPMENT.md` (Data, The Diabetes frame) says when to run
them and what fetches the snapshot.
"""

from __future__ import annotations

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
from bko.evaluate.diabetes import DIABETES_FRAME, DIABETES_INDICATORS


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


def test_the_drug_columns_read_are_the_eight_most_often_prescribed() -> None:
    raw = parse_diabetic(_member(MEMBER))
    drugs = COLUMNS[COLUMNS.index("metformin") : COLUMNS.index("change")]
    assert len(drugs) == len(DRUGS) + 15
    share = {name: float((raw[name] != "No").mean()) for name in drugs}
    assert tuple(sorted(drugs, key=lambda name: -share[name])[: len(DRUGS)]) == DRUGS
