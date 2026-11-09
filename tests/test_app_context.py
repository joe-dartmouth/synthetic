"""The app's `Context` refuses an experiment whose training split its analysis would not read."""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from bko.evaluate.experiment import Removal
from tests._app_fixtures import EXPERIMENT, app_context, never
from tests._guided_fixtures import YOUNG_SICK


def test_the_app_refuses_an_experiment_that_subsamples(tmp_path):
    with pytest.raises(ValueError, match="raw training split"):
        app_context(
            tmp_path,
            send=never,
            experiment=dataclasses.replace(EXPERIMENT, training_fraction=Fraction(1, 2)),
        )


def test_the_app_refuses_an_experiment_that_removes(tmp_path):
    with pytest.raises(ValueError, match="raw training split"):
        app_context(
            tmp_path,
            send=never,
            experiment=dataclasses.replace(EXPERIMENT, removal=Removal(YOUNG_SICK, 0.5)),
        )
