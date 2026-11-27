"""E22's decision gate: which mapping variants a hierarchy- or tree-reading arm can draw.

`plan_hierarchy` at `bko-hier-w`'s spec is built on each `diabetes` variant's committed rows, and
`column_trees` on each `diabetes-codes` variant's, against the pinned releases under the data
root. A variant whose build fails has its E22 family registered without that arm (ADR
`research/automatic-mapping`, Consequences). It is skipped unless `--releases` is passed, since the
gate reads no `data/`; an absent snapshot or release fails it rather than skipping it.
"""

from __future__ import annotations

import functools
from typing import BinaryIO, Final

import pytest

from bko.datasets.fetch import default_data_dir
from bko.datasets.frames import FrameSpec
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, field_rows
from bko.evaluate.direct_arms import DIRECT_ARMS, HIER_W_ARM
from bko.generate.direct import plan_hierarchy
from bko.generate.tree import column_trees
from bko.ontology.automatic_inputs import inputs
from bko.ontology.code_frames import DIABETES_CODES_NAME
from bko.ontology.frames import DIABETES_NAME
from bko.ontology.terms import Ontology


pytestmark = [pytest.mark.releases, pytest.mark.integration]

HIERARCHY: Final = {
    Dataset.DIABETES: True,
    Dataset.DIABETES_AUTO_LEXICAL: False,
    Dataset.DIABETES_AUTO_LOCAL: True,
    Dataset.DIABETES_PERMUTED: True,
}
"""Whether `bko-hier-w` can draw each `diabetes` variant: `auto-lexical` maps no state."""

TREE: Final = {
    Dataset.DIABETES_CODES: True,
    Dataset.DIABETES_CODES_AUTO_LEXICAL: True,
    Dataset.DIABETES_CODES_AUTO_LOCAL: True,
    Dataset.DIABETES_CODES_PERMUTED: True,
}
"""Whether `bko-tree` can draw each `diabetes-codes` variant."""


def _no_fetch(url: str, timeout: float) -> BinaryIO:
    msg = f"a release test fetched {url} ({timeout} s)"
    raise AssertionError(msg)


@functools.cache
def _inputs(dataset: str) -> tuple[FrameSpec, dict[str, Ontology]]:
    frame, ontologies, _ = inputs(dataset, default_data_dir(), open_url=_no_fetch)
    return frame, ontologies


def _rows(dataset: Dataset) -> tuple:
    return field_rows(dataset, Context(open_url=_no_fetch))


@pytest.mark.parametrize("dataset", sorted(HIERARCHY))
def test_bko_hier_w_builds_its_hierarchy_on_each_diabetes_variant(dataset: Dataset):
    frame, ontologies = _inputs(DIABETES_NAME)
    spec = DIRECT_ARMS[HIER_W_ARM]
    if HIERARCHY[dataset]:
        hierarchy = plan_hierarchy(_rows(dataset), ontologies, frame, spec)
        assert hierarchy is not None
        assert hierarchy.pairs
    else:
        with pytest.raises(ValueError, match="needs an is_a pair among the drawn concepts"):
            plan_hierarchy(_rows(dataset), ontologies, frame, spec)


@pytest.mark.parametrize("dataset", sorted(TREE))
def test_bko_tree_finds_a_tree_field_on_each_codes_variant(dataset: Dataset):
    frame, ontologies = _inputs(DIABETES_CODES_NAME)
    assert bool(column_trees(_rows(dataset), frame, ontologies)) is TREE[dataset]
