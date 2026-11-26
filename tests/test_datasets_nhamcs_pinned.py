"""NHAMCS's pinned archive under `data/`: its sha256, its real record, its counts and label rate.

It is skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot
fails it rather than skipping it. ADR `research/nhamcs-frame` records the figures.
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
from bko.datasets.nhamcs import MEMBER
from bko.datasets.sources import NHAMCS_ED_2022
from bko.ontology.catalog import CATALOG
from tests._nhamcs_fixtures import REAL_LINE


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _loaded() -> tuple[pd.DataFrame, FrameSpec]:
    return CATALOG["nhamcs"].read(default_data_dir(), open_url=_no_fetch)


def _no_fetch(url: str, timeout: float) -> io.BytesIO:
    msg = f"the pinned test fetches nothing: {url} ({timeout} s)"
    raise AssertionError(msg)


def test_the_archive_is_the_pinned_bytes() -> None:
    data = read_verified(NHAMCS_ED_2022, default_data_dir())
    assert hashlib.sha256(data).hexdigest() == (
        "a513693ed05b98bf94d2ff48b50de85afe48dd63cd7d43fb0930f914b18a9f25"
    )
    assert zipfile.ZipFile(io.BytesIO(data)).namelist() == [MEMBER]


def test_the_fixtures_real_record_is_the_files_first() -> None:
    data = read_verified(NHAMCS_ED_2022, default_data_dir())
    first = zipfile.ZipFile(io.BytesIO(data)).read(MEMBER).split(b"\r\n", 1)[0]
    assert first.decode("ascii") == REAL_LINE


def test_every_visit_is_a_record() -> None:
    records, _ = _loaded()
    assert len(records) == 16025
