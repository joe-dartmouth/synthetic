"""NHANES's pinned component files under `data/`: their sha256s, the adult frame, the composite,
the training split and the tertiles.

It is skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot
fails it rather than skipping it. ADR `research/nhanes-frame` records the figures, and the bounds
on the composite are those ADR `research/confirmatory-dataset-selection` derives from the
codebook's whole-file counts.
"""

from __future__ import annotations

import functools
import hashlib
import io

import pandas as pd
import pytest

from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.frames import FrameSpec
from bko.datasets.nhanes import COMPONENTS, DECIMALS, TERTILES, measured, read_component
from bko.datasets.sources import NHANES, NHANES_BMX_L
from bko.datasets.split import split
from bko.datasets.states import tertiles
from bko.ontology.catalog import CATALOG
from tests._nhanes_fixtures import REAL_COLUMNS, REAL_ROWS


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _loaded() -> tuple[pd.DataFrame, FrameSpec]:
    return CATALOG["nhanes"].read(default_data_dir(), open_url=_no_fetch)


def _no_fetch(url: str, timeout: float) -> io.BytesIO:
    msg = f"the pinned test fetches nothing: {url} ({timeout} s)"
    raise AssertionError(msg)


def test_each_component_is_the_pinned_bytes() -> None:
    found = {
        source.filename: hashlib.sha256(read_verified(source, default_data_dir())).hexdigest()
        for source in NHANES
    }
    assert found == {
        "DEMO_L.xpt": "ca4374a158b493b8b0163e1388da21d57a18d1b9cecff2aa4e2fa2bec494fe23",
        "MCQ_L.xpt": "0889bb3eaa5adedf1a11c4af2d4776fc4c4c3cfdeea24fcedb77c4636b478f61",
        "BPXO_L.xpt": "189aa9bed689d2c69cf8d780b01de85073276fea5694dd52cdf696229f9054a6",
        "BMX_L.xpt": "44440c416d9ad709e8b1708a5975378ab4d5b18edc39eb5015c2ae7186500170",
        "TCHOL_L.xpt": "5c3312e555bb0706fd97cbd63fcbc9e78f00bda8d2d149732d9dd368169b25d3",
        "HDL_L.xpt": "c1bb43df5ca366f470b411c043f4aee7ae6552a7f4bca1178bc87b24fb9a7300",
        "GHB_L.xpt": "67aee0353160e2392dc0a33bece99b90764a630c76d82415ac4639105ad9dd03",
    }


@pytest.mark.parametrize(("name", "source"), list(zip(COMPONENTS, NHANES, strict=True)))
def test_the_fixtures_real_columns_and_row_are_the_files(name: str, source) -> None:
    frame = pd.read_sas(io.BytesIO(read_verified(source, default_data_dir())), format="xport")
    assert isinstance(frame, pd.DataFrame)
    assert tuple(frame.columns) == REAL_COLUMNS[name]
    first = tuple(None if pd.isna(value) else value for value in frame.iloc[0].tolist())
    assert first == REAL_ROWS[name]


def test_the_adult_frame_the_composite_and_the_training_split() -> None:
    records, frame = _loaded()
    composite = int(records["target"].sum())
    assert 402 <= composite <= 1444
    train = split(records, seed=0, edges=frame.edges).train
    assert len(train) >= 5000
    assert (len(records), composite) == (7769, 937)
    assert (len(train), int(train["target"].sum())) == (6215, 750)
    assert f"{train['target'].mean():.4f}" == "0.1207"


def test_the_tertile_cutoffs_are_the_published_files() -> None:
    raw = read_component(read_verified(NHANES_BMX_L, default_data_dir()), "BMX_L")
    for name, cut in TERTILES.items():
        low, high = tertiles(measured(raw[name]))
        assert (round(low, DECIMALS), round(high, DECIMALS)) == cut.at_or_above, name
