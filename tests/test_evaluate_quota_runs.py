"""E21: the quota sweep, E12's runs with each share-based quota rule, each in ten slices."""

from __future__ import annotations

import dataclasses

import pytest

from bko.evaluate.data import Dataset
from bko.evaluate.experiment import PAID_ARMS, REAL_ARM
from bko.evaluate.named_runs import RUNS, wholes_of
from bko.evaluate.reference import REFERENCE, reference_of
from bko.evaluate.runs_diabetes import E6_CELL, E6_SETTINGS
from bko.evaluate.runs_direct import E12_EMPTY, E12_THIN, E21, E21_NAMES
from bko.gaps.quota import QuotaRule


WORDS = {
    "coverage": QuotaRule.COVERAGE,
    "coverage-two": QuotaRule.COVERAGE_2,
    "coverage-four": QuotaRule.COVERAGE_4,
    "coverage-eight": QuotaRule.COVERAGE_8,
    "expected-size": QuotaRule.EXPECTED,
    "recommended": QuotaRule.RECOMMENDED,
}
"""Each run name's word with the rule it sets."""

NAMES = tuple(f"{word}-{fraction}" for word in WORDS for fraction in ("thin", "empty"))


def test_the_sweep_is_every_rule_but_the_fixed_fill_at_both_fractions():
    assert tuple(E21) == NAMES
    assert {word: rule for rule, word in E21_NAMES.items()} == WORDS
    assert set(QuotaRule) - set(WORDS.values()) == {QuotaRule.FILL_TO}


@pytest.mark.parametrize("name", NAMES)
def test_each_sweep_run_differs_from_e12_only_in_its_rule(name):
    word, fraction = name.rsplit("-", 1)
    e12 = {"thin": E12_THIN, "empty": E12_EMPTY}[fraction]
    experiment = E21[name]
    assert experiment.quota_rule is WORDS[word]
    assert experiment == dataclasses.replace(e12, quota_rule=WORDS[word])
    assert e12.quota_rule is QuotaRule.FILL_TO
    whole = RUNS[f"e21-{name}"]
    assert (whole.experiment, whole.settings, whole.dataset) == (
        experiment,
        E6_SETTINGS,
        Dataset.DIABETES,
    )
    held: list[int] = []
    for letter in "abcdefghij":
        assert wholes_of(f"e21-{name}{letter}") == {f"e21-{name}"}
        held += RUNS[f"e21-{name}{letter}"].experiment.holdout_seeds
    assert held == list(whole.experiment.holdout_seeds)
    assert wholes_of(f"e21-{name}") == frozenset()


def test_no_sweep_arm_is_paid():
    names = [name for name in RUNS if name.startswith("e21-")]
    assert len(names) == len(NAMES) * 11
    for name in names:
        assert not set(RUNS[name].experiment.arms) & PAID_ARMS


@pytest.mark.parametrize("name", NAMES)
def test_the_reference_of_a_sweep_run_is_real_and_e6s_cell(name):
    assert REFERENCE[f"e21-{name}"] == REAL_ARM
    for run in (f"e21-{name}", f"e21-{name}c"):
        assert reference_of(run, RUNS[run].experiment, RUNS) == (REAL_ARM, E6_CELL)
