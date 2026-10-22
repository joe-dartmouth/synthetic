"""The page fills a gap through the real Messages API: one paid fill, run only under `--live`.

It reads the data root, fetching UCI Heart Disease and the releases if absent, and makes the
default split's BKO gap fill through a client with few retries, so a stalled call cannot hold the
run for long (docs/DEVELOPMENT.md, App).
"""

from __future__ import annotations

import dataclasses

import anthropic
import pytest

from bko.app.context import Context
from bko.app.page import OUTCOME_KEY
from bko.evaluate.experiment import E1
from bko.generate.api import send_with
from tests._app_fixtures import launch, settled, ticking


pytestmark = pytest.mark.live


def test_the_page_fills_a_gap_through_the_real_api(tmp_path):
    client = anthropic.Anthropic(max_retries=2, timeout=anthropic.Timeout(120.0, connect=30.0))
    context = Context(
        experiment=dataclasses.replace(
            E1, arms=("none", "bko"), comparators=("none",), replica_arms=()
        ),
        send=lambda: send_with(client),
        runs_dir=tmp_path / "runs",
        now=ticking(),
        setup_logging=lambda: None,
    )
    at = launch(context)
    at = settled(at.button(key="bko.analyze").click().run(timeout=300))
    at = settled(at.selectbox(key="bko.method").select("bko").run())
    at = settled(at.checkbox(key="bko.confirm.bko.0").check().run())
    at = settled(at.button(key="bko.fill").click().run(timeout=600))
    assert not at.error, [element.value for element in at.error]
    outcome = at.session_state[OUTCOME_KEY].value
    assert outcome.calls >= 2
    assert 0 < outcome.usd < context.ceiling
    assert len(outcome.baseline.added) == outcome.request.rows
