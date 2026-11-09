"""The app's `Context` refuses an experiment whose training split its analysis would not read."""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from bko.app.context import Context
from bko.evaluate.experiment import Removal
from tests._app_fixtures import EXPERIMENT
from tests._guided_fixtures import YOUNG_SICK


def test_the_app_accepts_an_experiment_that_reads_the_raw_split():
    assert Context(experiment=EXPERIMENT).experiment is EXPERIMENT


def test_the_app_refuses_an_experiment_that_subsamples():
    experiment = dataclasses.replace(EXPERIMENT, training_fraction=Fraction(1, 2))
    with pytest.raises(ValueError, match="raw training split") as caught:
        Context(experiment=experiment)
    assert "training_fraction=1/2" in str(caught.value)


def test_the_app_refuses_an_experiment_that_removes():
    removal = Removal(YOUNG_SICK, 0.5)
    experiment = dataclasses.replace(EXPERIMENT, removal=removal)
    with pytest.raises(ValueError, match="raw training split") as caught:
        Context(experiment=experiment)
    assert str(removal) in str(caught.value)
