"""SUPPORT2's pinned archive under `data/`: its sha256, the cohort, the label rate, the zeros and
the tertiles.

It is skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot
fails it rather than skipping it. ADR `research/support2-frame` records the figures.
"""

from __future__ import annotations

import functools
import hashlib
import io
import logging
import zipfile

import pandas as pd
import pytest

from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import SUPPORT2
from bko.datasets.split import split
from bko.datasets.states import tertiles
from bko.datasets.support2 import (
    DECIMALS,
    IMPLAUSIBLE_ZERO,
    MEMBER,
    TERTILES,
    frame_records,
    parse_support2,
    tertile_base,
)
from bko.ontology.catalog import CATALOG
from tests._support2_fixtures import REAL_HEADER, REAL_ROW


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _loaded() -> tuple[pd.DataFrame, FrameSpec]:
    return CATALOG["support2"].read(default_data_dir(), open_url=_no_fetch)


def _no_fetch(url: str, timeout: float) -> io.BytesIO:
    msg = f"the pinned test fetches nothing: {url} ({timeout} s)"
    raise AssertionError(msg)


def test_the_archive_is_the_pinned_bytes() -> None:
    data = read_verified(SUPPORT2, default_data_dir())
    assert hashlib.sha256(data).hexdigest() == (
        "8ed43980742a18e1847a8dfc5530bc4b30564ad9e4ad1b1b50bbc5d29d8c86fe"
    )
    assert zipfile.ZipFile(io.BytesIO(data)).namelist() == [MEMBER]


def test_the_fixtures_real_header_and_row_are_the_files() -> None:
    data = read_verified(SUPPORT2, default_data_dir())
    lines = zipfile.ZipFile(io.BytesIO(data)).read(MEMBER).decode("utf-8").splitlines()
    assert (lines[0], lines[1]) == (REAL_HEADER, REAL_ROW)


def test_the_cohort_and_the_training_splits_death_rate() -> None:
    records, frame = _loaded()
    assert len(records) == 9104
    assert int(records["target"].sum()) == 2359
    train = split(records, seed=0, edges=frame.edges).train
    assert (len(train), int(train["target"].sum())) == (7282, 1887)
    assert f"{train['target'].mean():.4f}" == "0.2591"


def _raw() -> pd.DataFrame:
    data = read_verified(SUPPORT2, default_data_dir())
    return parse_support2(zipfile.ZipFile(io.BytesIO(data)).read(MEMBER).decode("utf-8"))


def test_the_tertile_cutoffs_are_the_kept_records() -> None:
    """Each tertile field's cuts over its kept, zero-masked values, and their count."""
    base = tertile_base(_raw())
    for name, cut in TERTILES.items():
        low, high = tertiles(base[name])
        assert (round(low, DECIMALS), round(high, DECIMALS)) == cut.at_or_above, name
    assert {name: len(values) for name, values in base.items()} == {
        "edu": 7470,
        "sod": 9103,
        "alb": 5733,
        "ph": 6821,
        "glucose": 4603,
        "bun": 4752,
        "urine": 4242,
    }
    assert (TERTILES["glucose"].at_or_above, TERTILES["urine"].at_or_above) == (
        (111.0, 166.0),
        (1428.33, 2595.0),
    )


def test_the_zero_lines_count_the_adrs_zeros(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.datasets.support2"):
        frame_records(_raw())
    zeros = {"hrt": 83, "wblc": 10, "meanbp": 51, "glucose": 1}
    assert [m for m in caplog.messages if m.startswith("zeros ")] == [
        f"zeros dataset=support2 field={name} readings={zeros.get(name, 0)} "
        f"records={zeros.get(name, 0)}"
        for name in IMPLAUSIBLE_ZERO
    ]
