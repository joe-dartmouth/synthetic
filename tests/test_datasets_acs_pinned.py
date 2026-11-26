"""ACS's pinned archive and OccO release under `data/`: their bytes, the cohort and the label rate.

It is skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot
fails it rather than skipping it. ADR `research/acs-frame` records the figures.
"""

from __future__ import annotations

import functools
import hashlib
import io
import zipfile

import pandas as pd
import pytest

from bko.datasets.acs import HEADER, MEMBER, occupation_pairs, read_kept
from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import ACS_MA_2024, OCCO
from bko.datasets.split import split
from bko.ontology.catalog import CATALOG
from bko.ontology.elk import IMPORTS
from tests._acs_fixtures import REAL_HEADER, REAL_ROW


pytestmark = [pytest.mark.releases, pytest.mark.integration]


def _no_fetch(url: str, timeout: float) -> io.BytesIO:
    msg = f"the pinned test fetches nothing: {url} ({timeout} s)"
    raise AssertionError(msg)


@functools.cache
def _loaded() -> tuple[pd.DataFrame, FrameSpec]:
    return CATALOG["acs"].read(default_data_dir(), open_url=_no_fetch)


def test_the_archive_is_the_pinned_bytes() -> None:
    data = read_verified(ACS_MA_2024, default_data_dir())
    assert hashlib.sha256(data).hexdigest() == (
        "bb7d4009feaf8d190d687fc091fc7a56e2aa82b9972ae14a16d4ad70a8e0eef7"
    )
    assert zipfile.ZipFile(io.BytesIO(data)).namelist() == [MEMBER, "ACS2024_PUMS_README.pdf"]


def test_the_fixtures_real_header_and_row_are_the_files() -> None:
    data = read_verified(ACS_MA_2024, default_data_dir())
    lines = zipfile.ZipFile(io.BytesIO(data)).read(MEMBER).decode("utf-8").splitlines()
    assert len(lines) == 74_618
    assert (lines[0], lines[5]) == (REAL_HEADER, REAL_ROW)
    assert "RELSHIPP" in HEADER
    assert "RELP" not in HEADER


def test_the_filters_and_the_occupation_codes() -> None:
    rows, counts = read_kept(default_data_dir(), ACS_MA_2024)
    assert counts == (74_617, 63_191, 58_029, 42_543, 42_543)
    pairs = occupation_pairs(rows)
    assert len(pairs) == 517
    assert not (rows == "").any().any()


def test_the_cohort_and_the_training_splits_label_rate() -> None:
    records, frame = _loaded()
    assert len(records) == 42_543
    assert int(records["target"].sum()) == 24_896
    train = split(records, seed=0, edges=frame.edges).train
    assert (len(train), int(train["target"].sum())) == (34_035, 19_918)
    assert f"{train['target'].mean():.4f}" == "0.5852"


def test_the_occo_release_names_no_import() -> None:
    data = read_verified(OCCO, default_data_dir())
    assert hashlib.sha256(data).hexdigest() == (
        "97462cabc824526898e486c6d780d62ab13b81ec98d30d4f05df41178b2d9f37"
    )
    assert not any(marker in data for marker in IMPORTS)
