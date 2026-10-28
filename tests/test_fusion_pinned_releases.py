"""The real-data control: records showing every mapped concept close every per-subgroup gap.

Every other fusion test builds its records and releases in `tmp_path`. This case reads the UCI
snapshot and the pinned releases from the data root and is skipped unless `--releases` is passed.
Nothing is fetched: an absent snapshot or release fails it. No case reaches the API.
"""

from __future__ import annotations

from typing import NoReturn

import pytest

from bko.datasets.fetch import default_data_dir
from bko.datasets.sources import UCI_HEART_DISEASE
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.fusion.closure import closure
from bko.fusion.join import join
from bko.gaps.definitions import Definition, completeness_request
from bko.gaps.observe import record_cells
from bko.gaps.scope import load_scope
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from bko.validate.checks import validate
from tests._validate_fixtures import planted


pytestmark = [pytest.mark.releases, pytest.mark.integration]


def _no_fetch(url: str, timeout: float) -> NoReturn:
    pytest.fail(f"a release is absent from the data root, and this test never fetches: {url}")


def test_records_showing_every_mapped_concept_close_every_per_subgroup_gap() -> None:
    data_dir = default_data_dir()
    train = split(load_uci(data_dir)).train
    _, fragments = load_scope(
        {row.concept for row in FIELD_MAPPINGS},
        data_dir,
        releases=RELEASES,
        open_url=_no_fetch,
    )
    before = record_cells(train, FIELD_MAPPINGS, name=UCI_HEART_DISEASE.name)
    request = completeness_request(fragments, before)
    accepted = {}
    for cell, rows in request.items():
        band, sex, target = cell
        record = planted(
            age=40 if band == "<55" else 63,
            sex=0 if sex == "female" else 1,
            target=target,
            cp=1,
            trestbps=150,
            chol=250,
            fbs=1,
            restecg=2,
            exang=1,
            oldpeak=2.0,
            slope=2,
            ca=1,
            thal=7,
        )
        result = validate([record] * rows, frame=UCI_FRAME)
        assert result.passed == rows
        accepted[cell] = result.accepted
    joined = join(train, accepted, request)
    after = record_cells(joined.frame, FIELD_MAPPINGS, name=f"{UCI_HEART_DISEASE.name}-joined")
    readings = {reading.definition: reading for reading in closure(fragments, before, after)}
    assert readings[Definition.SUBGROUP].before
    assert readings[Definition.SUBGROUP].after == ()
    assert readings[Definition.SUBGROUP].opened == ()
    assert readings[Definition.POPULATION].opened == ()
