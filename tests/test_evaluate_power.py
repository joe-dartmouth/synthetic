"""`bko.evaluate.power.mde`: the smallest mean contrast a corrected interval reads at a power."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import t as student

from bko.evaluate.criteria import corrected_interval
from bko.evaluate.power import mde


def test_at_power_one_half_the_mde_is_the_intervals_half_width():
    contrast = np.random.default_rng(3).normal(0.05, 0.2, size=40).tolist()
    sd = float(np.std(contrast, ddof=1))
    interval = corrected_interval(contrast, ratio=0.25, confidence=0.95, family=1)
    half = interval.high - interval.mean
    assert mde(sd, holdouts=40, ratio=0.25, power=0.5) == pytest.approx(half, rel=1e-12)


def test_the_power_term_adds_its_t_quantile():
    sd, holdouts, ratio = 0.3, 100, 0.25
    scale = math.sqrt((1 / holdouts + ratio) * sd**2)
    gap = mde(sd, holdouts=holdouts, ratio=ratio) - mde(
        sd, holdouts=holdouts, ratio=ratio, power=0.5
    )
    assert gap == pytest.approx(float(student.ppf(0.8, holdouts - 1)) * scale, rel=1e-12)
    assert gap > 0


def test_the_mde_falls_with_holdouts_and_rises_with_ratio():
    few = mde(0.2, holdouts=10, ratio=0.25)
    many = mde(0.2, holdouts=100, ratio=0.25)
    assert many < few
    assert mde(0.2, holdouts=100, ratio=0.5) > many
    assert mde(0.4, holdouts=100, ratio=0.25) == pytest.approx(2 * many, rel=1e-12)
