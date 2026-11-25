"""The control runs against the pinned data under the data root.

E12's gap requests read the pinned Diabetes 130 archive, and E13's `bko-logic` the pinned
releases over the Synthea field rows. They are skipped unless `--releases` is passed, since the
gate reads no `data/`; an absent snapshot or release fails them rather than skipping them.
"""

from __future__ import annotations

import functools
from typing import NoReturn

import pandas as pd
import pytest

from bko.datasets.diabetes import load_diabetes
from bko.datasets.fetch import default_data_dir
from bko.datasets.split import split
from bko.evaluate.experiment import gap_request
from bko.evaluate.named_runs import RUNS
from bko.evaluate.prepare import prepared
from bko.evaluate.runs_diabetes import E6_CELL
from bko.gaps.scope import load_scope
from bko.generate.direct import logic_components
from bko.ontology.frames import (
    DIABETES_FRAME,
    DIABETES_INDICATORS,
    SYNTHEA_FIELD_ROWS,
    SYNTHEA_FRAME,
)
from bko.ontology.mapping import RELEASES


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _records() -> pd.DataFrame:
    return load_diabetes(default_data_dir(), indicators=DIABETES_INDICATORS)


def _no_fetch(url: str, timeout: float) -> NoReturn:
    pytest.fail(f"a release is absent from the data root, and this test never fetches: {url}")


@pytest.mark.parametrize(("name", "asked"), [("e12-thin", 1250), ("e12-empty", 1269)])
def test_each_e12_run_requests_only_the_removed_cell(name, asked):
    experiment = RUNS[name].experiment
    edges = DIABETES_FRAME.edges
    for seed in experiment.holdout_seeds:
        parts = split(_records(), seed=seed, edges=edges)
        train = prepared(parts, experiment, edges=edges).ablated.train
        request = gap_request(train, experiment, seed=seed, frame=DIABETES_FRAME)
        assert dict(request.quota) == {E6_CELL: asked}, seed


def test_bko_logic_draws_on_the_synthea_frame():
    concepts = {row.concept for row in SYNTHEA_FIELD_ROWS}
    ontologies, _ = load_scope(concepts, default_data_dir(), releases=RELEASES, open_url=_no_fetch)
    found = logic_components(SYNTHEA_FIELD_ROWS, ontologies, SYNTHEA_FRAME)
    assert len(found) == 5
    assert all(len(component.concepts) <= 2 for component in found)
