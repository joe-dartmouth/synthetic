"""Every Messages arm refuses data held under an agreement that grants no permission to send it."""

from __future__ import annotations

import dataclasses

import pytest

from bko.datasets import sources as pinned
from bko.datasets.sources import Agreement, Source, permits_sending
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import MESSAGES_ARMS, PAID_ARMS, PERTURBED_ARM
from bko.evaluate.guided import SEQUENTIAL, Knowledge
from bko.evaluate.guided_run import guided_arms
from bko.evaluate.holdout_candidates import Origin
from bko.evaluate.perturbed import PerturbedArm
from bko.ontology.mapping import RELEASES
from tests._diabetes_fixtures import pin, toy_archive
from tests._evaluate_bko_fixtures import PIN as UCI_PIN
from tests._evaluate_bko_fixtures import UCI_E1
from tests._evaluate_fixtures import GRADED_DATA, SMALL
from tests._generate_fixtures import ScriptedSend
from tests._ontology_fixtures import RoutingOpener


def refusing():
    pytest.fail("a send was built for data its agreement keeps here")


def open_arms(tmp_path, agreement: Agreement | None, send=refusing, arm: str = "bko"):
    data = dataclasses.replace(GRADED_DATA, agreement=agreement)
    return guided_arms(
        (arm,),
        Knowledge("x", {}, (), ()),
        SMALL,
        log=tmp_path / "calls.jsonl",
        candidates=tmp_path / "holdout-candidates.jsonl",
        origin=Origin("e1", data, RELEASES),
        send=send,
        run="e1",
        limits=SEQUENTIAL,
    )


@pytest.mark.parametrize("arm", ["bko", PERTURBED_ARM])
def test_a_paid_arm_refuses_data_under_an_agreement_without_permission(tmp_path, arm):
    with (
        pytest.raises(EvaluationError, match="agreement"),
        open_arms(tmp_path, Agreement("toy DUA", sending=None), arm=arm),
    ):
        pass
    assert not (tmp_path / "calls.jsonl").exists()
    assert not (tmp_path / "holdout-candidates.jsonl").exists()


def test_the_perturbed_arm_is_a_paid_arm_built_over_the_shared_flight(tmp_path):
    assert PERTURBED_ARM in PAID_ARMS
    assert PERTURBED_ARM not in MESSAGES_ARMS
    with open_arms(tmp_path, None, ScriptedSend, arm=PERTURBED_ARM) as run:
        assert set(run.arms) == {PERTURBED_ARM}
        assert isinstance(run.arms[PERTURBED_ARM], PerturbedArm)
    assert (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize(
    "agreement", [Agreement("toy DUA", sending="clause 4.2"), None], ids=["permitted", "open"]
)
def test_a_messages_arm_opens_on_permitted_or_open_data(tmp_path, agreement):
    with open_arms(tmp_path, agreement, ScriptedSend) as run:
        assert set(run.arms) == {"bko"}
    assert (tmp_path / "calls.jsonl").exists()


def test_every_pinned_source_is_open_data():
    sources = tuple(value for value in vars(pinned).values() if isinstance(value, Source))
    assert {source.name for source in sources} >= {"uci-heart-disease", "uci-diabetes-130"}
    assert all(source.agreement is None for source in sources)
    assert all(permits_sending(source.agreement) for source in sources)


@pytest.mark.parametrize("dataset", [Dataset.UCI, Dataset.DIABETES])
def test_load_carries_each_sources_agreement(tmp_path, dataset):
    agreement = Agreement("toy DUA", sending="ok")
    archive = toy_archive()
    uci = dataclasses.replace(UCI_PIN, agreement=agreement)
    diabetes = dataclasses.replace(pin(archive, "uci-diabetes-130"), agreement=agreement)
    context = Context(
        open_url=RoutingOpener({uci.url: UCI_E1, diabetes.url: archive}),
        uci=uci,
        diabetes=diabetes,
    )
    data = load(dataset, tmp_path, context)
    assert data.agreement == agreement
    assert data.name == (uci if dataset is Dataset.UCI else diabetes).name
