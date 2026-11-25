"""The `bko-hier` arms' gap fill of E6's cell against the pinned data under the data root.

The draw reads the pinned Diabetes 130 archive and the pinned releases over its field rows. It is
skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot or
release fails it rather than skipping it.
"""

from __future__ import annotations

import functools
from typing import NoReturn

import pandas as pd
import pytest

from bko.datasets.diabetes import load_diabetes
from bko.datasets.fetch import default_data_dir
from bko.datasets.split import split
from bko.evaluate.added_contradictions import added_counts
from bko.evaluate.direct_arms import DIRECT_ARMS, HIER_0_ARM, HIER_CLOSED_ARM, HIER_W_ARM
from bko.evaluate.experiment import gap_request
from bko.evaluate.named_runs import RUNS
from bko.evaluate.prepare import prepared
from bko.evaluate.runs_diabetes import E6_CELL
from bko.gaps.observe import Cell
from bko.gaps.scope import load_scope
from bko.generate.direct import Draw, plan_hierarchy, sample
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
from bko.ontology.frames import (
    DIABETES_FIELD_ROWS,
    DIABETES_FRAME,
    DIABETES_INDICATORS,
    DIABETES_NAME,
)
from bko.ontology.mapping import RELEASES
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]

PAIRS = (("MONDO:0005016", "MONDO:0005300"), ("MONDO:0005607", "MONDO:0005002"))
"""The frame's two `is_a` pairs (`docs/EVIDENCE.md`, E16)."""


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


@functools.cache
def _fill(name: str, seed: int, spec: DrawSpec) -> tuple[dict[Cell, int], tuple[Draw, ...]]:
    experiment = RUNS[name].experiment
    edges = DIABETES_FRAME.edges
    parts = split(_loaded().copy(), seed=seed, edges=edges)
    train = prepared(parts, experiment, edges=edges).ablated.train
    request = gap_request(train, experiment, seed=seed, frame=DIABETES_FRAME)
    drawn = sample(
        train,
        request.quota,
        rows=DIABETES_FIELD_ROWS,
        ontologies=_ontologies(),
        name=DIABETES_NAME,
        seed=request.seed,
        frame=DIABETES_FRAME,
        tag=(seed, "gap"),
        spec=spec,
    )
    return dict(request.quota), drawn[E6_CELL]


def _narrow_without_broad(draws: tuple[Draw, ...]) -> int:
    added = pd.DataFrame([dict(draw.record) for draw in draws])
    found = added_counts(
        added,
        DIABETES_FIELD_ROWS,
        _ontologies(),
        frame=DIABETES_FRAME,
        asked=None,
        name="added",
    )["narrow-without-broad"]
    assert found is not None
    return found


def test_the_frames_pairs_are_the_two_the_census_names():
    """A broad concept with two narrow ones is refused here, before any draw; it parks the unit."""
    for arm in (HIER_0_ARM, HIER_W_ARM, HIER_CLOSED_ARM):
        hierarchy = plan_hierarchy(
            DIABETES_FIELD_ROWS, _ontologies(), DIABETES_FRAME, DIRECT_ARMS[arm]
        )
        assert hierarchy is not None
        assert hierarchy.pairs == PAIRS


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("name", "asked"), [("e16-thin", 1250), ("e16-empty", 1269)])
@pytest.mark.parametrize("arm", [HIER_0_ARM, HIER_W_ARM, HIER_CLOSED_ARM])
def test_a_hier_fill_of_e6s_cell_completes_at_both_fractions(arm, name, asked, seed):
    """Narrow-without-broad rows fall from `bko-hier-0` to `bko-hier-w` to the closed fill's 0.

    `bko-hier-0` reads the data's conditional, which leaves most records' narrow concept without
    its broad one, so its count stays near `bko-direct`'s; `c_w >= c_0` lowers it at `bko-hier-w`.
    """
    quota, draws = _fill(name, seed, DIRECT_ARMS[arm])
    assert quota == {E6_CELL: asked}
    assert len(draws) == asked
    found = _narrow_without_broad(draws)
    if arm == HIER_CLOSED_ARM:
        assert found == 0
    elif arm == HIER_0_ARM:
        assert found > 0
    else:
        assert found < _narrow_without_broad(_fill(name, seed, DIRECT_ARMS[HIER_0_ARM])[1])


@pytest.mark.parametrize(("name", "asked"), [("e16-thin", 1250), ("e16-empty", 1269)])
def test_the_closed_fill_holds_no_narrow_without_broad_row(name, asked):
    _, closed = _fill(name, 0, DIRECT_ARMS[HIER_CLOSED_ARM])
    _, direct = _fill(name, 0, DEFAULT_SPEC)
    assert len(closed) == len(direct) == asked
    assert _narrow_without_broad(closed) == 0
    assert _narrow_without_broad(direct) > 0
    for draw in closed:
        for narrow, broad in PAIRS:
            assert narrow not in draw.concepts or broad in draw.concepts
