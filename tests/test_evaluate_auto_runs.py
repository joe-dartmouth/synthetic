"""E22: eight mapping families, each `E9_THIN` and `E9_EMPTY` on its own dataset, in ten slices."""

from __future__ import annotations

import dataclasses

import pytest

from bko.evaluate.data import Dataset
from bko.evaluate.direct_arms import HIER_W_ARM, NOMAP_ARM, TREE_ARM
from bko.evaluate.experiment import DIRECT_ARM, PAID_ARMS, REAL_ARM
from bko.evaluate.named_runs import RUNS, wholes_of
from bko.evaluate.reference import REFERENCE, reference_of
from bko.evaluate.runs_diabetes import E6_SETTINGS, E9_EMPTY, E9_THIN
from bko.evaluate.runs_direct import E22


DATASETS = {
    "hand": Dataset.DIABETES,
    "auto-lexical": Dataset.DIABETES_AUTO_LEXICAL,
    "auto-local": Dataset.DIABETES_AUTO_LOCAL,
    "permuted": Dataset.DIABETES_PERMUTED,
    "codes-hand": Dataset.DIABETES_CODES,
    "codes-auto-lexical": Dataset.DIABETES_CODES_AUTO_LEXICAL,
    "codes-auto-local": Dataset.DIABETES_CODES_AUTO_LOCAL,
    "codes-permuted": Dataset.DIABETES_CODES_PERMUTED,
}
"""Each E22 family by its run-name word, with the dataset ADR `research/automatic-mapping` gives."""

FRACTIONS = (("thin", E9_THIN), ("empty", E9_EMPTY))

WHOLES = [f"{family}-{fraction}" for family in DATASETS for fraction, _ in FRACTIONS]


def test_the_e22_names_are_the_wholes_and_their_slices():
    names = [name for name in RUNS if name.startswith("e22-")]
    assert names == [f"e22-{whole}{suffix}" for whole in WHOLES for suffix in ("", *"abcdefghij")]
    assert list(E22) == WHOLES


@pytest.mark.parametrize("family", list(DATASETS))
@pytest.mark.parametrize("fraction", ["thin", "empty"])
def test_each_e22_family_reads_its_dataset(family, fraction):
    for suffix in ("", *"abcdefghij"):
        run = RUNS[f"e22-{family}-{fraction}{suffix}"]
        assert (run.settings, run.dataset) == (E6_SETTINGS, DATASETS[family])
    assert E22[f"{family}-{fraction}"][1] == DATASETS[family]


@pytest.mark.parametrize(("fraction", "e9"), FRACTIONS)
def test_the_families_of_one_frame_differ_only_in_dataset_and_gated_arms(fraction, e9):
    hier = ("none", REAL_ARM, DIRECT_ARM, NOMAP_ARM, HIER_W_ARM)
    tree = ("none", REAL_ARM, DIRECT_ARM, NOMAP_ARM, TREE_ARM)
    arms = {
        "hand": hier,
        "auto-lexical": ("none", REAL_ARM, DIRECT_ARM, NOMAP_ARM),
        "auto-local": hier,
        "permuted": hier,
        "codes-hand": tree,
        "codes-auto-lexical": tree,
        "codes-auto-local": tree,
        "codes-permuted": tree,
    }
    for family, family_arms in arms.items():
        experiment = RUNS[f"e22-{family}-{fraction}"].experiment
        assert experiment == dataclasses.replace(
            e9, arms=family_arms, candidate=DIRECT_ARM, comparators=(NOMAP_ARM, "none")
        )
        assert experiment.replica_arms == ()
        assert not set(experiment.arms) & PAID_ARMS


@pytest.mark.parametrize("whole", WHOLES)
def test_each_e22_runs_slices_cover_its_holdouts_once(whole):
    run = RUNS[f"e22-{whole}"]
    held: list[int] = []
    for letter in "abcdefghij":
        assert wholes_of(f"e22-{whole}{letter}") == {f"e22-{whole}"}
        held += RUNS[f"e22-{whole}{letter}"].experiment.holdout_seeds
    assert held == list(run.experiment.holdout_seeds)
    assert wholes_of(f"e22-{whole}") == frozenset()


@pytest.mark.parametrize("whole", WHOLES)
def test_the_reference_of_an_e22_run_is_bko_direct(whole):
    assert REFERENCE[f"e22-{whole}"] == DIRECT_ARM
    removal = RUNS[f"e22-{whole}"].experiment.removal
    assert removal is not None
    for run in (f"e22-{whole}", f"e22-{whole}c"):
        assert reference_of(run, RUNS[run].experiment, RUNS) == (DIRECT_ARM, removal.cell)
