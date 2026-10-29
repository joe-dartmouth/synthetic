"""A free fill through the page over the real UCI snapshot and the pinned releases in `data/`.

It never fetches and never calls the API; run it with `--releases` once `make cells` and
`python -m bko.ontology check` have fetched the data (docs/DEVELOPMENT.md, App).
"""

from __future__ import annotations

from typing import NoReturn

import pytest

from bko.app.context import Context
from bko.app.page import OUTCOME_KEY
from bko.datasets.cells import cell_counts
from bko.datasets.fetch import default_data_dir
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.gaps.definitions import Definition
from bko.gaps.detect import count_gaps
from tests._app_fixtures import frame_with, launch, never, settled


pytestmark = [pytest.mark.releases, pytest.mark.integration]


def _no_fetch(url: str, timeout: float) -> NoReturn:
    pytest.fail(f"a release is absent from the data root, and this test never fetches: {url}")


def test_a_free_fill_runs_through_the_page_over_the_pinned_releases(tmp_path):
    data_dir = default_data_dir()
    context = Context(
        open_url=_no_fetch, send=never, runs_dir=tmp_path / "runs", setup_logging=lambda: None
    )
    at = launch(context)
    at = settled(at.button(key="bko.analyze").click().run(timeout=300))
    cells = frame_with(at, {"count", "status"})
    flagged = cells.loc[cells["status"] == "underrepresented", ["age_band", "sex", "target"]]
    expected = {
        gap.cell
        for gap in count_gaps(
            cell_counts(
                split(load_uci(data_dir), edges=UCI_FRAME.edges).train, edges=UCI_FRAME.edges
            )
        )
    }
    assert set(flagged.itertuples(index=False, name=None)) == expected
    at = settled(at.selectbox(key="bko.method").select("smote").run())
    at = settled(at.button(key="bko.fill").click().run(timeout=600))
    outcome = at.session_state[OUTCOME_KEY].value
    assert len(outcome.baseline.added) == outcome.request.rows
    assert len(frame_with(at, {"closed", "still_open"})) == len(Definition)
