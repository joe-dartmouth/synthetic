"""PhysioNet 2012's pinned files under `data/`: their sha256s, the real stay and the counts.

It is skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot
fails it rather than skipping it. ADR `research/physionet2012-frame` records the figures.
"""

from __future__ import annotations

import functools
import hashlib
import io
import logging
import re
import zipfile
from collections import Counter

import pandas as pd
import pytest

from bko.datasets.archive import members, tar_members
from bko.datasets.fetch import default_data_dir, read_verified
from bko.datasets.frames import FrameSpec, digest
from bko.datasets.physionet2012 import (
    SETS,
    TARRED,
    Stay,
    frame_records,
    parse_outcomes,
    parse_stay,
)
from bko.datasets.sources import (
    PHYSIONET2012,
    PHYSIONET2012_OUTCOMES_A,
    PHYSIONET2012_SET_A,
)
from bko.datasets.split import split
from bko.ontology.catalog import CATALOG
from tests._physionet2012_fixtures import (
    REAL_DIED_ID,
    REAL_DIED_OUTCOME,
    REAL_DIED_STAY,
    REAL_ID,
    REAL_OUTCOME,
    REAL_STAY,
)


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _loaded() -> tuple[pd.DataFrame, FrameSpec]:
    return CATALOG["physionet2012"].read(default_data_dir(), open_url=_no_fetch)


@functools.cache
def _texts() -> dict[str, dict[str, str]]:
    """Each set's members by name, read from its pinned archive."""
    texts = {}
    for label, source in zip(SETS, PHYSIONET2012, strict=False):
        read = tar_members if label in TARRED else members
        data = read_verified(source, default_data_dir())
        texts[label] = dict(read(data, re.compile(rf"set-{label}/[0-9]+\.txt"), source))
    return texts


@functools.cache
def _lines() -> tuple[tuple[str, str, str, str], ...]:
    """Every observation line of every stay: its set, its member, its `Parameter` and `Value`."""
    return tuple(
        (label, name, *line.split(",")[1:])
        for label, held in _texts().items()
        for name, text in held.items()
        for line in text.split("\n")[1:-1]
    )


@functools.cache
def _stays() -> tuple[Stay, ...]:
    return tuple(
        parse_stay(name, text) for held in _texts().values() for name, text in held.items()
    )


def _values(parameter: str) -> list[str]:
    return [value for _, _, name, value in _lines() if name == parameter]


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
    assert {label: len(held) for label, held in _texts().items()} == {
        "a": 4000,
        "b": 4000,
        "c": 4000,
    }


def test_the_descriptors_hold_the_adrs_figures() -> None:
    ages = Counter(float(value) for value in _values("Age"))
    assert (min(ages), max(ages), ages[90.0], ages[89.0], ages[-1.0]) == (15, 90, 464, 120, 0)
    assert Counter(_values("Gender"))["-1"] == 12
    assert set(_values("ICUType")) == {"1", "2", "3", "4"}


def test_the_values_hold_the_adrs_figures() -> None:
    assert Counter(_values("MechVent")) == {"1": 91644}
    below = [value for value in _values("Temp") if float(value) < 0 and value != "-1"]
    assert (len(below), below.count("-17.8")) == (59, 55)
    assert sum("e" in value for _, _, _, value in _lines()) == 83
    unnamed = [(label, name) for label, name, parameter, _ in _lines() if not parameter]
    assert Counter(label for label, _ in unnamed) == {"c": 2403}
    assert len(set(unnamed)) == 1594


def test_the_recorded_zeros_are_the_adrs() -> None:
    kept = [stay for stay in _stays() if -1.0 not in {stay.age, stay.gender}]
    zeros = {
        (variable, how): sum(stay.reduced(variable, how) == 0 for stay in kept)
        for variable, how in (
            *(("Temp", "min"), ("MAP", "min"), ("HR", "max")),
            *(("RespRate", "max"), ("WBC", "min"), ("Bilirubin", "max")),
        )
    }
    assert zeros == {
        ("Temp", "min"): 102,
        ("MAP", "min"): 108,
        ("HR", "max"): 0,
        ("RespRate", "max"): 0,
        ("WBC", "min"): 1,
        ("Bilirubin", "max"): 1,
    }


def test_the_cohort_line_and_the_records_are_the_adrs(caplog) -> None:
    deaths: dict[int, int] = {}
    for source in PHYSIONET2012[len(SETS) :]:
        text = read_verified(source, default_data_dir()).decode("ascii")
        deaths.update(parse_outcomes(text, source.name))
    with caplog.at_level(logging.INFO, logger="bko.datasets.physionet2012"):
        records = frame_records(list(_stays()), deaths)
    assert caplog.messages[-1].startswith(
        "cohort dataset=physionet2012 stays=12000 stays_a=4000 stays_b=4000 stays_c=4000 "
        "unknown_age=0 unknown_sex=12 left_out=12 records=11988 died=1706 unnamed=2403 missing=0 "
        "below_zero=59 "
    )
    assert digest(records) == "5ca824f645d6a37af0aa99c29d37556f8994ba3cde2941cdac61b807b0678f48"


def test_the_fixtures_real_stays_and_outcomes_are_the_files() -> None:
    stays = _texts()["a"]
    assert stays[f"set-a/{REAL_ID}.txt"] == REAL_STAY
    assert stays[f"set-a/{REAL_DIED_ID}.txt"] == REAL_DIED_STAY
    outcomes = read_verified(PHYSIONET2012_OUTCOMES_A, default_data_dir()).decode("ascii")
    assert f"\n{REAL_OUTCOME}\n" in outcomes
    assert f"\n{REAL_DIED_OUTCOME}\n" in outcomes


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


def test_the_records_and_the_training_splits_death_rate() -> None:
    records, frame = _loaded()
    assert len(records) == 11988
    assert int(records["target"].sum()) == 1706
    assert int((records["mechvent"] == 1).sum()) == 7576
    train = split(records, seed=0, edges=frame.edges).train
    assert (len(train), int(train["target"].sum())) == (9590, 1364)
    assert f"{train['target'].mean():.4f}" == "0.1422"
    assert digest(train) == "b99bf9acdc084ad965060ed868b24db828fe216bc08260c45544536a203b72e2"
