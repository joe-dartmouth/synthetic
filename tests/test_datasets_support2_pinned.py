"""SUPPORT2's pinned archive under `data/`: its sha256, the cohort, the label rate and the tertiles.

It is skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot
fails it rather than skipping it. ADR `research/support2-frame` records the figures.
"""

from __future__ import annotations

import functools
import hashlib
import io
import zipfile

import pandas as pd
import pytest

from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import SUPPORT2
from bko.datasets.split import split
from bko.datasets.states import tertiles
from bko.datasets.support2 import DECIMALS, MEMBER, TERTILES, checked, measured, parse_support2
from bko.ontology.catalog import CATALOG


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


def test_the_cohort_and_the_training_splits_death_rate() -> None:
    records, frame = _loaded()
    assert len(records) == 9104
    assert int(records["target"].sum()) == 2359
    train = split(records, seed=0, edges=frame.edges).train
    assert (len(train), int(train["target"].sum())) == (7282, 1887)
    assert f"{train['target'].mean():.4f}" == "0.2591"


def test_the_tertile_cutoffs_are_the_published_files() -> None:
    data = read_verified(SUPPORT2, default_data_dir())
    raw = checked(parse_support2(zipfile.ZipFile(io.BytesIO(data)).read(MEMBER).decode("utf-8")))
    for name, cut in TERTILES.items():
        low, high = tertiles(measured(raw[name]))
        assert (round(low, DECIMALS), round(high, DECIMALS)) == cut.at_or_above, name
