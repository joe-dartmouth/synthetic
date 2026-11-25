"""The fixed-partition seam: a dataset with held ids scores the same test records at every seed.

`bko.datasets.split.fixed` holds a published partition's test records out, and
`bko.evaluate.data.held_split` chooses it over the seed-based `split` when the data holds ids
(ADR `research/tableshift-comparison`, Decision 7).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
from fractions import Fraction
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student

import bko.evaluate.likelihood as likelihood_module
import bko.evaluate.privacy_stores as privacy_stores_module
import bko.evaluate.scoring as scoring_module
import bko.evaluate.shared as shared_module
import bko.evaluate.trace as trace_module
from bko.baselines.request import Request
from bko.datasets.diabetes import parse_diabetic
from bko.datasets.errors import ContractError
from bko.datasets.split import fixed, split
from bko.datasets.tableshift import kept_rows, partition
from bko.datasets.tableshift_frame import encode, tableshift_frame
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.classifier import MODELS
from bko.evaluate.context import Context
from bko.evaluate.criteria import corrected_interval, held_out_ratio_of, series_of
from bko.evaluate.data import Data, Dataset, held_split, load
from bko.evaluate.direct import DirectArm, run_direct
from bko.evaluate.experiment import DIRECT_ARM, E1, Removal, gap_request
from bko.evaluate.guided import Knowledge
from bko.evaluate.holdout_candidates import CandidateFile
from bko.evaluate.named_runs import RUNS
from bko.evaluate.prepare import held_out_ratio, prepared
from bko.evaluate.rehearse import safe_ranges
from bko.evaluate.run import run
from bko.evaluate.runs_published import P1
from bko.evaluate.scores import Measure, Score
from bko.ontology.frames import LAB_FIELD_ROWS
from tests._diabetes_fixtures import pin, text
from tests._evaluate_bko_fixtures import context, scripted
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL
from tests._frame_fixtures import TOY, toy_records
from tests._ontology_fixtures import RoutingOpener
from tests._tableshift_fixtures import tableshift_archive, tableshift_lines


ONLY_SMOTE = dataclasses.replace(
    SMALL, arms=("smote",), candidate="smote", comparators=(), replica_arms=()
)
"""E1 on two holdouts with SMOTE alone."""


def _tableshift() -> Data:
    raw = parse_diabetic(text(*tableshift_lines()))
    kept = kept_rows(raw)
    parts = partition(raw)
    records = encode(kept).loc[sorted([*parts.train, *parts.test])]
    held = frozenset(int(record) for record in parts.test)
    return Data(records, tableshift_frame(kept), LAB_FIELD_ROWS, "toy", None, held=held)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_a_fixed_partition_gives_the_same_test_ids_at_every_seed(seed):
    data = _tableshift()
    assert data.held is not None
    parts = held_split(data, seed)
    assert set(parts.test.index) == data.held
    assert set(parts.train.index) == set(data.records.index) - data.held
    assert parts.seed == seed
    assert parts.fingerprint == held_split(data, 0).fingerprint


@pytest.mark.parametrize("seed", [0, 3])
def test_a_dataset_without_held_ids_splits_as_before(seed):
    data = Data(toy_records(), TOY, (), "toy", None)
    assert data.held is None
    expected = split(data.records, seed=seed, edges=TOY.edges)
    found = held_split(data, seed)
    pd.testing.assert_frame_equal(found.train, expected.train)
    pd.testing.assert_frame_equal(found.test, expected.test)
    assert found.seed == seed


def test_a_held_id_missing_from_the_records_is_refused():
    records = toy_records()
    missing = int(records.index.max()) + 1
    with pytest.raises(ContractError, match=f"held record {missing}"):
        fixed(records, held=frozenset({int(records.index[0]), missing}), seed=0)


def test_bko_direct_draws_on_the_tableshift_fixture():
    data = _tableshift()
    train = held_split(data, 0).train
    request: Request = gap_request(train, E1, seed=0, frame=data.frame)
    arm = DirectArm(LAB_FIELD_ROWS, {}, name=DIRECT_ARM, budget=3, sink=None, dataset="toy")
    added = arm(train, request).added
    assert len(added) == sum(request.quota.values())
    assert tuple(added.columns) == data.frame.columns


def test_p1_names_the_stated_configuration():
    arms = (
        "none",
        "class-weight",
        "smote",
        "adasyn",
        "ctgan",
        "tvae",
        "gaussian-copula",
        "cart",
        "bayes-net",
        "tabddpm",
        "label-bootstrap",
        "bko-direct",
    )
    assert P1.arms == arms
    assert P1.holdout_seeds == tuple(range(10))
    assert P1.candidate == "bko-direct"
    assert P1.comparators == arms[:-1]
    assert P1.replica_arms == (
        "smote",
        "adasyn",
        "ctgan",
        "tvae",
        "gaussian-copula",
        "cart",
        "bayes-net",
        "tabddpm",
        "label-bootstrap",
        "bko-direct",
    )
    assert P1.criteria == ()
    assert P1.decontaminate is False
    assert P1.removal is None
    assert (
        dataclasses.replace(
            P1,
            holdout_seeds=E1.holdout_seeds,
            arms=E1.arms,
            candidate=E1.candidate,
            comparators=E1.comparators,
            replica_arms=E1.replica_arms,
            criteria=E1.criteria,
            decontaminate=E1.decontaminate,
        )
        == E1
    )
    assert RUNS["p1-id"].experiment == P1
    assert RUNS["p1-ood"].experiment == P1
    assert RUNS["p1-id"].dataset is Dataset.TABLESHIFT_ID
    assert RUNS["p1-ood"].dataset is Dataset.TABLESHIFT_OOD
    assert list(RUNS)[-2:] == ["p1-id", "p1-ood"]


def _held(records: pd.DataFrame) -> frozenset[int]:
    """A held set of `records`: seed 5's test records, so every cell holds some out."""
    return frozenset(int(r) for r in split(records, seed=5, edges=UCI_FRAME.edges).test.index)


def test_a_fixed_split_refuses_no_held_record():
    with pytest.raises(ContractError, match="one or more held records"):
        fixed(toy_records(), held=frozenset(), seed=0)


def test_a_fixed_split_refuses_a_duplicate_index():
    records = toy_records()
    doubled = pd.concat([records, records.iloc[:1]])
    with pytest.raises(ContractError, match="unique record index"):
        fixed(doubled, held=frozenset({int(records.index[1])}), seed=0)


def test_the_digest_covers_the_held_ids_only_when_there_are_some():
    records = toy_records()
    csv = records.to_csv(lineterminator="\n")
    plain = Data(records, TOY, (), "toy", None)
    assert plain.digest == hashlib.sha256(csv.encode()).hexdigest()
    first, second = sorted(int(record) for record in records.index[:2])
    both = Data(records, TOY, (), "toy", None, held=frozenset({second, first}))
    assert both.digest == hashlib.sha256(f"{csv}held={first},{second}\n".encode()).hexdigest()
    one = Data(records, TOY, (), "toy", None, held=frozenset({first}))
    assert len({plain.digest, both.digest, one.digest}) == 3


@pytest.fixture(scope="module")
def fixed_run() -> tuple[frozenset[int], list[set[int]], list[Score]]:
    held = _held(GRADED)
    seen: list[set[int]] = []

    def spying(train, request):
        seen.append(set(train.index))
        return FAST_ARMS["smote"](train, request)

    scores = run(GRADED, ONLY_SMOTE, {"smote": spying}, frame=UCI_FRAME, held=held)
    return held, seen, scores


def test_run_holds_the_same_test_records_out_at_every_holdout(fixed_run):
    held, seen, _ = fixed_run
    assert len(seen) == len(ONLY_SMOTE.holdout_seeds)
    assert all(indexes == set(GRADED.index) - held for indexes in seen)


def test_a_fixed_run_logs_ratio_0_and_reads_the_plain_t_interval(fixed_run):
    _, _, scores = fixed_run
    ratios = {s.holdout: s.value for s in scores if s.measure is Measure.HELD_OUT_RATIO}
    assert ratios == {0: 0.0, 1: 0.0}
    ratio = held_out_ratio_of(series_of(scores))
    assert ratio == pytest.approx(0.0, abs=0)
    contrast = [0.1, 0.4, 0.25]
    found = corrected_interval(contrast, ratio=ratio, confidence=0.95, family=1)
    half = student.ppf(0.975, 2) * np.std(contrast, ddof=1) / math.sqrt(3)
    assert found.mean == pytest.approx(0.25)
    assert (found.low, found.high) == pytest.approx((0.25 - half, 0.25 + half))


def test_a_seed_split_keeps_its_held_out_ratio():
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    assert not parts.fixed
    assert held_out_ratio(parts, edges=UCI_FRAME.edges) > 0


@pytest.mark.parametrize(
    "change",
    [
        {"training_fraction": Fraction(1, 2)},
        {"removal": Removal((">=55", "female", 1), 0.5)},
    ],
)
def test_a_fixed_split_stays_fixed_through_the_fraction_and_the_removal(change):
    held = _held(GRADED)
    experiment = dataclasses.replace(SMALL, **change)
    ready = prepared(fixed(GRADED, held=held, seed=0), experiment, edges=UCI_FRAME.edges)
    assert ready.sized.fixed
    assert ready.ablated.fixed
    assert held_out_ratio(ready.sized, edges=UCI_FRAME.edges) == pytest.approx(0.0, abs=0)
    assert set(ready.ablated.test.index) == held


def test_run_direct_passes_the_held_ids_to_the_harness():
    held = _held(GRADED)
    seen: list[set[int]] = []

    def spying(train, request):
        seen.append(set(train.index))
        return FAST_ARMS["smote"](train, request)

    run_direct(
        GRADED,
        ONLY_SMOTE,
        {"smote": spying},
        Knowledge("toy", {}, (), ()),
        names=(),
        sink=cast("CandidateFile", None),
        models=MODELS,
        frame=UCI_FRAME,
        held=held,
    )
    assert seen
    assert all(indexes == set(GRADED.index) - held for indexes in seen)


def test_score_runs_the_loaded_datas_held_ids(tmp_path, monkeypatch):
    loaded, held_seen = scoring_module.load, []

    def holding(*args):
        data = loaded(*args)
        return dataclasses.replace(data, held=_held(data.records))

    def spying(*args, **kwargs):
        held_seen.append(kwargs["held"])
        return run(*args, **kwargs)

    monkeypatch.setattr(scoring_module, "load", holding)
    monkeypatch.setattr(scoring_module, "run", spying)
    ctx, _ = context(scripted(), arms={"none": FAST_ARMS["none"], "bko": FAST_ARMS["bko"]})
    argv = ["score", "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    assert len(held_seen) == 1
    assert held_seen[0] is not None
    assert held_seen[0] == _held(loaded(Dataset.UCI, tmp_path / "data", ctx).records)


def test_the_rehearsals_safe_ranges_read_the_fixed_training_split():
    held = _held(GRADED)
    train = GRADED.loc[~GRADED.index.isin(held)]
    expected = {
        name: (float(train[name].min()), float(train[name].max())) for name in UCI_FRAME.numeric
    }
    assert safe_ranges(GRADED, SMALL, frame=UCI_FRAME, held=held) == expected


@pytest.mark.parametrize(
    "module", [shared_module, trace_module, privacy_stores_module, likelihood_module]
)
def test_every_split_site_splits_through_held_split(module):
    assert module.held_split is held_split
    assert not hasattr(module, "split")


@pytest.mark.parametrize(
    ("dataset", "part"), [(Dataset.TABLESHIFT_ID, "test"), (Dataset.TABLESHIFT_OOD, "ood_test")]
)
def test_load_holds_the_domains_test_split(tmp_path, dataset, part):
    archive = tableshift_archive()
    source = pin(archive, "uci-diabetes-130")
    data = load(
        dataset, tmp_path, Context(open_url=RoutingOpener({source.url: archive}), diabetes=source)
    )
    parts = partition(parse_diabetic(text(*tableshift_lines())))
    test = frozenset(int(record) for record in getattr(parts, part))
    assert data.held == test
    assert set(data.records.index) == set(parts.train) | test
    assert data.field_rows == LAB_FIELD_ROWS
    assert data.frame.name == "tableshift"
    assert data.name == source.name
    assert set(held_split(data, 3).test.index) == test
