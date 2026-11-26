"""PhysioNet 2012's pinned files under `data/`: their sha256s, the real stay and the counts.

It is skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot
fails it rather than skipping it. ADR `research/physionet2012-frame` records the figures.
"""

from __future__ import annotations

import functools
import hashlib
import io
import re
import zipfile

import pandas as pd
import pytest

from bko.datasets.archive import members, tar_members
from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import (
    PHYSIONET2012,
    PHYSIONET2012_OUTCOMES_A,
    PHYSIONET2012_SET_A,
    PHYSIONET2012_SET_C,
)
from bko.ontology.catalog import CATALOG
from tests._physionet2012_fixtures import REAL_ID, REAL_OUTCOME, REAL_STAY


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _loaded() -> tuple[pd.DataFrame, FrameSpec]:
    return CATALOG["physionet2012"].read(default_data_dir(), open_url=_no_fetch)


def _no_fetch(url: str, timeout: float) -> io.BytesIO:
    msg = f"the pinned test fetches nothing: {url} ({timeout} s)"
    raise AssertionError(msg)


def test_the_files_are_the_pinned_bytes() -> None:
    digests = [
        hashlib.sha256(read_verified(source, default_data_dir())).hexdigest()
        for source in PHYSIONET2012
    ]
    assert digests == [
        "1433638192fda622e8bb11a4559b8af2dbb1c69f0e1c3b5b606cc12aabd0a4c5",
        "54094c39631797af554ad4233164e004a699521351bd2f2bed50d6564ebc56dc",
        "a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1",
        "2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2",
        "1305f29734da374a809483229a99f33a9230aa8d4fa74b1a2d6ad788136423d5",
        "0b32dcb7753c0f18f2f5fde498a28618b8879c5e527ce0dae8ad120dd7e205c9",
    ]


def test_each_set_holds_4000_stays() -> None:
    zipped = read_verified(PHYSIONET2012_SET_A, default_data_dir())
    names = zipfile.ZipFile(io.BytesIO(zipped)).namelist()
    assert (names[0], len(names)) == ("set-a/", 4001)
    tarred = read_verified(PHYSIONET2012_SET_C, default_data_dir())
    pattern = re.compile(r"set-c/\d+\.txt")
    assert sum(1 for _ in tar_members(tarred, pattern, PHYSIONET2012_SET_C)) == 4000


def test_the_fixtures_real_stay_and_outcome_are_the_files() -> None:
    zipped = read_verified(PHYSIONET2012_SET_A, default_data_dir())
    stays = dict(members(zipped, re.compile(r"set-a/\d+\.txt"), PHYSIONET2012_SET_A))
    assert stays[f"set-a/{REAL_ID}.txt"] == REAL_STAY
    outcomes = read_verified(PHYSIONET2012_OUTCOMES_A, default_data_dir()).decode("ascii")
    assert f"\n{REAL_OUTCOME}\n" in outcomes


def test_the_real_stay_loads_as_its_record() -> None:
    records, _ = _loaded()
    assert records.loc[REAL_ID].to_dict() == {
        "age": 76,
        "sex": 1,
        "ICUType": 3,
        "hr_max": 2,
        "resprate_max": 0,
        "temp_max": 1,
        "temp_min": 1,
        "wbc_max": 1,
        "wbc_min": 2,
        "map_min": 1,
        "creatinine_max": 1,
        "bilirubin_max": 2,
        "platelets_min": 1,
        "gcs_min": 1,
        "mechvent": 1,
        "target": 0,
    }
