"""Guidance built and rendered over the real mapping table, UCI snapshot and pinned releases.

Every other generation test builds its records and releases in `tmp_path`, so a change that
breaks guidance over the real table passes `make check`. These cases read the UCI snapshot and
the pinned releases from the data root and are skipped unless `--releases` is passed. Nothing is
fetched: an absent snapshot or release fails them. No case reaches the API.
"""

from __future__ import annotations

import logging
from typing import NoReturn

import pytest

from bko.datasets.fetch import default_data_dir
from bko.datasets.sources import UCI_HEART_DISEASE
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.gaps.definitions import Definition, gaps
from bko.gaps.observe import record_cells
from bko.gaps.scope import load_scope
from bko.generate.guidance import Basis, guide
from bko.generate.messages import LEVELS, Level
from bko.generate.prompt import BASES, render
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES


pytestmark = [pytest.mark.releases, pytest.mark.integration]


def _no_fetch(url: str, timeout: float) -> NoReturn:
    pytest.fail(f"a release is absent from the data root, and this test never fetches: {url}")


@pytest.mark.parametrize("definition", list(Definition))
def test_every_cell_renders_at_every_level_and_every_statement_reaches_a_row(
    definition: Definition, caplog: pytest.LogCaptureFixture
) -> None:
    data_dir = default_data_dir()
    train = split(load_uci(data_dir), edges=UCI_FRAME.edges).train
    ontologies, fragments = load_scope(
        {row.concept for row in FIELD_MAPPINGS},
        data_dir,
        releases=RELEASES,
        open_url=_no_fetch,
    )
    cells = record_cells(train, FIELD_MAPPINGS, name=UCI_HEART_DISEASE.name, edges=UCI_FRAME.edges)
    with caplog.at_level(logging.WARNING, logger="bko.generate"):
        guided = guide(
            train,
            cells,
            gaps(definition, fragments, cells),
            definition=definition,
            fragments=fragments,
            ontologies=ontologies,
            frame=UCI_FRAME,
        )
    assert [r.getMessage() for r in caplog.records if r.getMessage().startswith("unreached")] == []
    assert len(guided) == len(cells.each)
    chained = 0
    for guidance in guided.values():
        for level in LEVELS:
            system, user = render(guidance, 2, level, frame=UCI_FRAME)
            assert system
            assert user.startswith("Write exactly 2 records")
        _, user = render(guidance, 2, Level.ONTOLOGY, frame=UCI_FRAME)
        if guidance.basis is not Basis.NONE and guidance.findings:
            assert BASES[guidance.basis] in user
            chained += " is a kind of " in user
    assert chained > 0
