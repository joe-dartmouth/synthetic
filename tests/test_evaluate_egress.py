"""Every Messages arm refuses data held under an agreement that grants no permission to send it."""

from __future__ import annotations

import dataclasses

import pytest

from bko.datasets.sources import Agreement
from bko.evaluate.errors import EvaluationError
from bko.evaluate.guided import SEQUENTIAL, Knowledge, guided_arms
from bko.evaluate.holdout_candidates import Origin
from bko.ontology.mapping import RELEASES
from tests._evaluate_fixtures import GRADED_DATA, SMALL
from tests._generate_fixtures import ScriptedSend


def refusing():
    pytest.fail("a send was built for data its agreement keeps here")


def open_arms(tmp_path, agreement: Agreement | None, send=refusing):
    data = dataclasses.replace(GRADED_DATA, agreement=agreement)
    return guided_arms(
        ("bko",),
        Knowledge("x", {}, (), ()),
        SMALL,
        log=tmp_path / "calls.jsonl",
        candidates=tmp_path / "holdout-candidates.jsonl",
        origin=Origin("e1", data, RELEASES),
        send=send,
        run="e1",
        limits=SEQUENTIAL,
    )


def test_a_messages_arm_refuses_data_under_an_agreement_without_permission(tmp_path):
    with (
        pytest.raises(EvaluationError, match="agreement"),
        open_arms(tmp_path, Agreement("toy DUA", sending=None)),
    ):
        pass
    assert not (tmp_path / "calls.jsonl").exists()
    assert not (tmp_path / "holdout-candidates.jsonl").exists()


@pytest.mark.parametrize(
    "agreement", [Agreement("toy DUA", sending="clause 4.2"), None], ids=["permitted", "open"]
)
def test_a_messages_arm_opens_on_permitted_or_open_data(tmp_path, agreement):
    with open_arms(tmp_path, agreement, ScriptedSend) as run:
        assert set(run.arms) == {"bko"}
    assert (tmp_path / "calls.jsonl").exists()
