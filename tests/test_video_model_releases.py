"""The demo video's example model against UCI Heart Disease and the pinned HPO release in `data/`.

`video.model` holds its cell counts, edges and holders as literals, so a change to the split, the
mapping table or the release would leave the video showing figures the pipeline no longer computes.
These cases load the data root and are skipped unless `--releases` is passed, since the gate reads
no `data/`. A failure means a model literal is wrong: fix the model.
"""

from __future__ import annotations

import functools
from typing import NoReturn

import pytest

from bko.datasets.fetch import default_data_dir
from bko.datasets.sources import UCI_HEART_DISEASE
from bko.datasets.uci import UCI_FRAME
from bko.gaps.detect import detect
from bko.gaps.observe import CellRecords, cell_key, dataset_fragment
from bko.gaps.scope import UciScope, load_uci_scope
from bko.gaps.shares import share_parts
from bko.generate.direct import label_group
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from video import model
from video.captions import CELL_GAPS


pytestmark = [pytest.mark.releases, pytest.mark.integration]


def _refuse(*_: object, **__: object) -> NoReturn:
    msg = "the release tests read data/ and never fetch"
    raise AssertionError(msg)


@functools.cache
def _scope() -> UciScope:
    return load_uci_scope(
        default_data_dir(),
        uci=UCI_HEART_DISEASE,
        releases=RELEASES,
        field_rows=FIELD_MAPPINGS,
        open_url=_refuse,
    )


def _cell() -> CellRecords:
    (records,) = [r for c, r in _scope().cells.each.items() if cell_key(c) == model.CELL]
    return records


def test_the_models_cell_counts_are_the_training_splits() -> None:
    counts = {cell_key(c): r.observed.records for c, r in _scope().cells.each.items()}
    assert counts == dict(model.CELL_RECORDS)


def test_each_model_edge_is_an_hpo_is_a_edge() -> None:
    terms = _scope().ontologies["HP"].terms
    for child, parent in model.EDGES:
        assert parent in terms[child].parents, (child, parent)


def test_the_models_holders_are_the_cells() -> None:
    holders = _cell().observed.holders
    assert {c: holders.get(c, 0) for c in model.HOLDERS} == dict(model.HOLDERS)


def test_the_models_share_is_the_samplers() -> None:
    group = label_group(
        _scope().parts.train, 1, FIELD_MAPPINGS, name=UCI_HEART_DISEASE.name, frame=UCI_FRAME
    )
    assert group.records == model.GROUP_RECORDS
    assert group.holders["HP:0001717"] == model.GROUP_HOLDERS["HP:0001717"]
    parts = share_parts(_cell(), group)
    assert parts["HP:0001717"].share == pytest.approx(model.share("HP:0001717"))


def test_the_cell_holds_the_captions_gap_count_including_the_models() -> None:
    found = detect(_scope().fragments, dataset_fragment(_cell().observed))
    assert len(found) == CELL_GAPS
    assert set(model.gaps()) <= {(s.antecedent, s.consequent) for s in found}
