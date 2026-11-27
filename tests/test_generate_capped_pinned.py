"""`bko-capped`'s gap fill of E6's cell against the pinned data under the data root.

The draw reads the pinned Diabetes 130 archive and the pinned releases over its field rows. It is
skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot or
release fails it rather than skipping it.
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import NoReturn

import pandas as pd
import pytest

from bko.datasets.diabetes import load_diabetes
from bko.datasets.fetch import default_data_dir
from bko.datasets.split import split
from bko.evaluate.direct_arms import CAPPED_ARM, DIRECT_ARMS
from bko.evaluate.experiment import gap_request
from bko.evaluate.named_runs import RUNS
from bko.evaluate.prepare import prepared
from bko.evaluate.runs_diabetes import E6_CELL
from bko.gaps.scope import load_scope
from bko.generate.direct import sample
from bko.ontology.frames import (
    DIABETES_FIELD_ROWS,
    DIABETES_FRAME,
    DIABETES_INDICATORS,
    DIABETES_NAME,
)
from bko.ontology.mapping import RELEASES
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]


@functools.cache
def _loaded() -> pd.DataFrame:
    return load_diabetes(default_data_dir(), indicators=DIABETES_INDICATORS)


def _no_fetch(url: str, timeout: float) -> NoReturn:
    pytest.fail(f"a release is absent from the data root, and this test never fetches: {url}")


@functools.cache
def _ontologies() -> dict[str, Ontology]:
    concepts = {row.concept for row in DIABETES_FIELD_ROWS}
    ontologies, _ = load_scope(concepts, default_data_dir(), releases=RELEASES, open_url=_no_fetch)
    return dict(ontologies)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("name", "asked"), [("e14-thin", 1250), ("e14-empty", 1269)])
def test_a_capped_fill_of_e6s_cell_completes_at_both_fractions(name, asked, seed):
    """At a floor of 2 and a cap of 1 per unit, the gap fill draws every asked row of E6's cell.

    A pool every matching donor of which is at the cap raises `SamplingError`, which fails this
    test, and no training record gives one unit's values twice (ADR
    `research/sampler-donor-limits`). A unit's fields share their source, so a field's reuse is
    its unit's.
    """
    experiment = RUNS[name].experiment
    edges = DIABETES_FRAME.edges
    parts = split(_loaded().copy(), seed=seed, edges=edges)
    train = prepared(parts, experiment, edges=edges).ablated.train
    request = gap_request(
        train, experiment, seed=seed, frame=DIABETES_FRAME, rows=(), source_name=""
    )
    assert dict(request.quota) == {E6_CELL: asked}
    drawn = sample(
        train,
        request.quota,
        rows=DIABETES_FIELD_ROWS,
        ontologies=_ontologies(),
        name=DIABETES_NAME,
        seed=request.seed,
        frame=DIABETES_FRAME,
        tag=(seed, "gap"),
        spec=DIRECT_ARMS[CAPPED_ARM],
    )
    assert len(drawn[E6_CELL]) == asked
    reuse = Counter((field, source) for d in drawn[E6_CELL] for field, source in d.sources.items())
    assert max(reuse.values()) == 1
