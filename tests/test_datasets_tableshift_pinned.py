"""TableShift's readmission partition on the pinned Diabetes 130 archive under the data root.

The counts are TableShift's Table 12 (ADR `research/tableshift-comparison`). Skipped unless
`--releases` is passed, since the gate reads no `data/`; an absent snapshot fails them.
"""

from __future__ import annotations

import dataclasses
import functools

import pandas as pd
import pytest

from bko.datasets.archive import member_text
from bko.datasets.diabetes import MEMBER, parse_diabetic
from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.sources import DIABETES_130
from bko.datasets.tableshift import Partition, kept_rows, load_tableshift, partition
from bko.evaluate.experiment import E1, flagged


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _raw() -> pd.DataFrame:
    data = read_verified(DIABETES_130, default_data_dir())
    return parse_diabetic(member_text(data, MEMBER, DIABETES_130))


@functools.cache
def _partition() -> Partition:
    return partition(_raw())


def test_the_pinned_archive_partition_counts_equal_table_12() -> None:
    parts = _partition()
    assert len(kept_rows(_raw())) == 99493
    sizes = tuple(len(getattr(parts, field.name)) for field in dataclasses.fields(parts))
    assert sizes == (34288, 4286, 4287, 5664, 50968)


def test_the_flagged_cells_are_the_three_youngest() -> None:
    records, _, spec = load_tableshift(default_data_dir(), domain="id")
    train = records.loc[_partition().train]
    assert flagged(train, E1, edges=spec.edges) == (
        ("<1", "female", 1),
        ("<1", "male", 0),
        ("<1", "male", 1),
    )
