"""`bko-coherent`'s gap fill of E6's cell against the pinned data under the data root.

The draw reads the pinned Diabetes 130 archive and the pinned releases over its field rows. It is
skipped unless `--releases` is passed, since the gate reads no `data/`; an absent snapshot or
release fails it rather than skipping it. Each fill logs its three coherence counts beside
`bko-direct`'s at the same seed, and each rule's breaks under `bko-coherent` are at most
`bko-direct`'s.
"""

from __future__ import annotations

import functools
import logging
from typing import NoReturn

import pandas as pd
import pytest

from bko.baselines.request import Request
from bko.datasets.diabetes import load_diabetes
from bko.datasets.fetch import default_data_dir
from bko.datasets.split import split
from bko.evaluate.coherent import CoherentArm, coherent_draws
from bko.evaluate.direct import DirectArm
from bko.evaluate.direct_arms import COHERENT_ARM
from bko.evaluate.experiment import gap_request
from bko.evaluate.named_runs import RUNS
from bko.evaluate.perturb import DIABETES_UNMAPPED, RULES, rule_breaks
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

logger = logging.getLogger(__name__)


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


def _request(name: str, seed: int) -> tuple[pd.DataFrame, Request, int]:
    """The training split, the gap request and the attempt budget of `name` at `seed`."""
    experiment = RUNS[name].experiment
    edges = DIABETES_FRAME.edges
    parts = split(_loaded().copy(), seed=seed, edges=edges)
    train = prepared(parts, experiment, edges=edges).ablated.train
    return (
        train,
        gap_request(train, experiment, seed=seed, frame=DIABETES_FRAME, rows=(), source_name=""),
        experiment.budget,
    )


def _breaks(rows: pd.DataFrame) -> dict[str, int]:
    return {rule: int(broken.sum()) for rule, broken in rule_breaks(rows).items()}


@pytest.mark.parametrize(("name", "asked"), [("e17-thin", 1250), ("e17-empty", 1269)])
def test_a_coherent_fill_of_e6s_cell_completes_at_both_fractions(name, asked):
    train, request, budget = _request(name, 0)
    assert dict(request.quota) == {E6_CELL: asked}
    inner = DirectArm(
        DIABETES_FIELD_ROWS,
        _ontologies(),
        name=COHERENT_ARM,
        budget=budget,
        sink=None,
        dataset=DIABETES_NAME,
    )
    added = CoherentArm(inner, fields=DIABETES_UNMAPPED)(train, request).added
    assert len(added) == asked
    drawn = sample(
        train,
        request.quota,
        rows=DIABETES_FIELD_ROWS,
        ontologies=_ontologies(),
        name=DIABETES_NAME,
        seed=request.seed,
        frame=DIABETES_FRAME,
        tag=(0, "gap"),
    )[E6_CELL]
    changed = coherent_draws(train, request, drawn, fields=DIABETES_UNMAPPED)
    rows = pd.DataFrame([dict(draw.record) for draw in changed], columns=list(added.columns))
    pd.testing.assert_frame_equal(
        rows.astype(dict(DIABETES_FRAME.dtypes)), added.reset_index(drop=True)
    )
    for draw in changed:
        assert set(DIABETES_UNMAPPED) <= set(draw.sources)
        for field in DIABETES_UNMAPPED:
            assert train.loc[draw.sources[field], field] == draw.record[field]
    direct = pd.DataFrame([dict(draw.record) for draw in drawn])
    ours, theirs = _breaks(rows), _breaks(direct)
    logger.warning(
        "coherence %s seed=0 rows=%d %s",
        name,
        asked,
        " ".join(f"{rule}={ours[rule]}/{theirs[rule]}" for rule in RULES),
    )
    for rule in RULES:
        assert ours[rule] <= theirs[rule], (rule, ours[rule], theirs[rule])
