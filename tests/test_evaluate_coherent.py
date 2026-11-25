"""`bko-coherent`: `bko-direct`'s draw with its free fields redrawn by conditional CART."""

from __future__ import annotations

import dataclasses
import io
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pytest

from bko.datasets.cells import cell_keys
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import coherent as coherent_module
from bko.evaluate import donor_attacks, trace_perturb
from bko.evaluate.__main__ import Context, _named, main
from bko.evaluate.attribution import Attribution, arm_kind, classify
from bko.evaluate.coherent import CoherentArm, coherent_draws, coherent_records
from bko.evaluate.data import digest, load
from bko.evaluate.direct import WRAPPERS, DirectArm
from bko.evaluate.direct_arms import COHERENT_ARM
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import DIRECT_ARM, DIRECT_ARMS, PRAM_ARM
from bko.evaluate.holdout_candidates import CHANGED, KEPT, PERTURBS, UNCHANGED, read
from bko.evaluate.named_runs import RUNS, NamedRun
from bko.evaluate.pram import PramArm
from bko.evaluate.reference import REFERENCE
from bko.evaluate.run_folder import BKO
from bko.evaluate.runs_diabetes import E9_EMPTY, E9_THIN
from bko.generate.direct import Draw, sample
from bko.generate.draw_spec import DEFAULT_SPEC
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests.test_evaluate_attribution import MAPPED, _direct
from tests.test_evaluate_direct import ADDED_DIGEST, SMALL_G2, g2_context, varied_arm
from tests.test_evaluate_donor_attacks import _donors_of, _sourced
from tests.test_evaluate_pram import FIELDS
from tests.test_evaluate_shared import FORMAT
from tests.test_evaluate_trace_perturb import _trace


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

COHERED = dataclasses.replace(
    SMALL_G2,
    arms=("none", DIRECT_ARM, COHERENT_ARM),
    candidate=COHERENT_ARM,
    comparators=("none", DIRECT_ARM),
    replica_arms=(),
)
"""Two holdouts, one baseline, `bko-direct` and `bko-coherent`, the coherent arm the candidate."""

COHERENT_DIGEST: Final = "d3babb0bd042a6ea3621baeca356c225cb2002aab215232d96ee33c5a51dc55d"
"""`digest` of `bko-coherent`'s rows on `varied_arm`'s request at seed 0: pins the trees, the
leaf draws and their stream."""


def coherent_arm() -> CoherentArm:
    """The coherent arm over `varied_arm`'s field rows, redrawing `FIELDS`."""
    inner = DirectArm(
        (ANGINA, STENOSIS), {}, name=COHERENT_ARM, budget=3, sink=None, dataset="uci-test"
    )
    return CoherentArm(inner, fields=FIELDS)


def _cohered(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> tuple[pd.DataFrame, Context]:
    """A finished `g2` run of `COHERED` in `tmp_path / "run"`, its data root beside it."""
    caplog.set_level(logging.INFO)
    ctx, send = g2_context()
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(COHERED)})
    folder = tmp_path / "run"
    folder.mkdir()
    out = io.StringIO()
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    caplog.clear()
    assert main([*argv, "--data-dir", str(tmp_path / "data")], context=ctx, out=out) == 0
    assert send.calls == []
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    return pd.read_csv(io.StringIO(out.getvalue())), ctx


@pytest.fixture
def cohered(tmp_path, caplog) -> tuple[Path, Context]:
    """A finished `g2` folder holding `bko-coherent`, its data root beside it; its context."""
    _, ctx = _cohered(tmp_path, caplog)
    return tmp_path / "run", ctx


def _training(folder: Path, ctx: Context) -> pd.DataFrame:
    """The `g2` records under `folder`'s data root, indexed by record id."""
    named = _named("g2", ctx)
    return load(named.dataset, folder.parent / "data", named).records


def test_the_coherent_arm_is_a_direct_arm_at_the_default_spec_a_bko_arm_and_a_perturbing_one():
    assert DIRECT_ARMS[COHERENT_ARM] == DEFAULT_SPEC
    assert COHERENT_ARM in BKO
    assert PERTURBS[COHERENT_ARM] == frozenset({CHANGED, UNCHANGED})
    inner = coherent_arm().inner
    assert isinstance(WRAPPERS[COHERENT_ARM](inner, FIELDS), CoherentArm)
    assert isinstance(WRAPPERS[PRAM_ARM](inner, FIELDS), PramArm)


def test_mapped_fields_and_the_label_equal_bko_directs_at_the_same_seed():
    train, request, direct = varied_arm()
    ours = coherent_arm()(train, request)
    theirs = direct(train, request).added
    assert digest(theirs) == ADDED_DIGEST
    assert list(ours.added.index) == list(theirs.index)
    kept = [column for column in theirs.columns if column not in FIELDS]
    pd.testing.assert_frame_equal(ours.added[kept], theirs[kept])
    assert (ours.added[list(FIELDS)] != theirs[list(FIELDS)]).to_numpy().any()
    assert ours.params == {"generator": "direct", "seed": 0, "perturb": "coherent"}


def test_the_coherent_fill_is_pinned_and_moves_with_the_seed():
    train, request, _ = varied_arm()
    first = coherent_arm()(train, request).added
    assert digest(first) == COHERENT_DIGEST
    pd.testing.assert_frame_equal(first, coherent_arm()(train, request).added)
    other = coherent_arm()(train, dataclasses.replace(request, seed=1)).added
    assert digest(other) != digest(first)


def test_each_free_value_is_a_training_value():
    train, request, _ = varied_arm()
    added = coherent_arm()(train, request).added
    for name in FIELDS:
        assert set(added[name]) <= set(train[name])


@pytest.mark.parametrize("field", ["age", "sex", "target", "cp"])
def test_the_arm_refuses_to_redraw_the_cell_the_label_or_a_mapped_field(field):
    train, request, _ = varied_arm()
    arm = CoherentArm(coherent_arm().inner, fields=(*FIELDS, field))
    with pytest.raises(EvaluationError, match="may not perturb"):
        arm(train, request)


def test_the_leaf_draws_read_the_seed_and_stream_2(monkeypatch):
    train, request, _ = varied_arm(5)
    real = np.random.default_rng
    seeds: list[Any] = []

    def recorded(seed: Any = None) -> np.random.Generator:
        seeds.append(seed)
        return real(seed)

    monkeypatch.setattr(coherent_module.np.random, "default_rng", recorded)
    coherent_arm()(train, request)
    assert [5, 2] in seeds
    assert [5, 1] not in seeds


def test_coherent_records_of_no_record_is_empty():
    train, request, _ = varied_arm()
    assert coherent_records(train, request, [], fields=FIELDS) == ([], [])


def test_a_free_field_the_draw_fell_back_on_is_no_fallback_once_redrawn():
    train, request, _ = varied_arm()
    keys = cell_keys(train, edges=UCI_FRAME.edges)
    cell = next(iter(request.quota))
    reduced = train[[tuple(key) != cell for key in keys.itertuples(index=False, name=None)]]
    drawn = sample(
        reduced,
        {cell: 4},
        rows=(ANGINA, STENOSIS),
        ontologies={},
        name="uci-test",
        seed=0,
        frame=UCI_FRAME,
    )[cell]
    assert all(set(FIELDS) <= draw.fallback for draw in drawn)
    seeded = dataclasses.replace(request, quota={cell: 4})
    for old, new in zip(drawn, coherent_draws(reduced, seeded, drawn, fields=FIELDS), strict=True):
        assert not set(FIELDS) & new.fallback
        assert new.fallback == old.fallback - set(FIELDS)


def test_coherent_records_refuses_a_field_outside_the_frame_by_name():
    train, request, _ = varied_arm()
    with pytest.raises(ValueError, match="field 'weight' is not a column of frame 'uci'"):
        coherent_records(train, request, [], fields=(*FIELDS, "weight"))


def test_a_score_run_with_the_coherent_arm_logs_and_writes_its_lines(tmp_path, caplog):
    _cohered(tmp_path, caplog)
    folder = tmp_path / "run"
    found = read((folder / "holdout-candidates.jsonl").read_bytes())
    mine = [line for line in found.candidates if line.arm == COHERENT_ARM]
    assert len(mine) == 2 * 28
    assert {line.call for line in mine} == {None}
    assert {line.outcome for line in mine} == {KEPT}
    assert CHANGED in {line.perturb for line in mine} <= {CHANGED, UNCHANGED}
    (configured,) = [m for m in caplog.messages if m.startswith("perturb arm=")]
    assert configured == f"perturb arm=bko-coherent fields={','.join(FIELDS)} stream=2"
    cohered_lines = [m for m in caplog.messages if m.startswith("perturb coherent ")]
    assert len(cohered_lines) == 2
    assert all(
        re.fullmatch(r"perturb coherent holdout=(0|1) arm=bko-coherent rows=28", m)
        for m in cohered_lines
    ), cohered_lines
    for holdout in (0, 1):
        changed = sum(line.perturb == CHANGED for line in mine if line.holdout == holdout)
        (shapes,) = [
            m
            for m in caplog.messages
            if m.startswith(f"perturb shapes holdout={holdout} arm={COHERENT_ARM} ")
        ]
        assert int(shapes.split(" rows_changed=")[1].split(" ")[0]) == changed


def test_a_coherent_fill_traces_to_its_arm_sha(cohered, caplog):
    folder, ctx = cohered
    code, found = _trace(folder, ctx, COHERENT_ARM)
    assert code == 0
    kept = [
        line
        for line in read((folder / "holdout-candidates.jsonl").read_bytes()).candidates
        if (line.holdout, line.arm, line.outcome) == (0, COHERENT_ARM, KEPT)
    ]
    assert len(found) == len(kept) > 0
    (arm_line,) = [
        m for m in caplog.messages if m.startswith(f"trace arm holdout=0 arm={COHERENT_ARM} ")
    ]
    assert arm_line.endswith(" coherent=true matches=true")
    assert CHANGED in {line["perturb"]["result"] for line in found} <= {CHANGED, UNCHANGED}
    for line in found:
        assert (line["perturb"]["result"] == CHANGED) == bool(line["perturb"]["fields"])
        assert set(line["perturb"]["fields"]) <= set(FIELDS)


def test_each_free_field_names_a_leaf_donor_in_the_trace(cohered):
    folder, ctx = cohered
    records = _training(folder, ctx)
    _, direct = _trace(folder, ctx, DIRECT_ARM)
    _, found = _trace(folder, ctx, COHERENT_ARM)
    moved = 0
    for line, drawn in zip(found, direct, strict=True):
        record, sources = line["record"], line["sources"]
        assert set(FIELDS) <= set(sources)
        for name in FIELDS:
            assert records.loc[sources[name], name] == record[name]
        assert not set(FIELDS) & set(line["fallback"])
        others = {name: at for name, at in sources.items() if name not in FIELDS}
        assert others == {k: v for k, v in drawn["sources"].items() if k not in FIELDS}
        moved += sum(sources[name] != drawn["sources"].get(name) for name in FIELDS)
    assert moved > 0


def _altered(real: Any) -> Any:
    """`coherent_draws` with the first draw's first free value moved by one."""

    def altered(*args: Any, **kwargs: Any) -> list[Draw]:
        draws = real(*args, **kwargs)
        first = draws[0]
        record = {**first.record, FIELDS[0]: first.record[FIELDS[0]] + 1}
        return [dataclasses.replace(first, record=record), *draws[1:]]

    return altered


def test_a_change_re_run_that_differs_from_the_lines_exits_3(cohered, caplog, monkeypatch):
    folder, ctx = cohered
    monkeypatch.setattr(trace_perturb, "coherent_draws", _altered(trace_perturb.coherent_draws))
    assert _trace(folder, ctx, COHERENT_ARM) == (3, [])
    assert f"the change re-run gives other values than the lines in: {FIELDS[0]}" in caplog.text
    (arm_line,) = [
        m for m in caplog.messages if m.startswith(f"trace arm holdout=0 arm={COHERENT_ARM} ")
    ]
    assert arm_line.endswith(" coherent=false matches=false")


def test_a_failed_unperturbed_check_skips_the_change_re_run_and_says_so(
    cohered, caplog, monkeypatch
):
    folder, ctx = cohered
    real = trace_perturb.redraw
    reruns: list[int] = []

    def one_short(*args: Any, **kwargs: Any) -> Any:
        found = real(*args, **kwargs)
        return dataclasses.replace(found, drawn=found.drawn[:-1])

    def counted(*args: Any, **kwargs: Any) -> list[Draw]:
        reruns.append(1)
        return []

    monkeypatch.setattr(trace_perturb, "redraw", one_short)
    monkeypatch.setattr(trace_perturb, "coherent_draws", counted)
    assert _trace(folder, ctx, COHERENT_ARM) == (3, [])
    assert reruns == []
    assert "lines in the returned attempt, not one per draw of" in caplog.text
    assert (
        f"trace coherent holdout=0 arm={COHERENT_ARM} rerun=skipped failed=unperturbed"
        in caplog.messages
    )
    (arm_line,) = [
        m for m in caplog.messages if m.startswith(f"trace arm holdout=0 arm={COHERENT_ARM} ")
    ]
    assert arm_line.endswith(" unperturbed=false matches=false")


def test_attribution_classes_a_coherent_free_field_as_a_donor():
    perturb = {"result": "changed", "fields": ["chol"], "drawn": {"chol": 250}}
    line = _direct(COHERENT_ARM, perturb=perturb)
    found = classify(line, mapped=MAPPED)
    assert found["chol"] is Attribution.DONOR
    assert found["cp"] is Attribution.KNOWLEDGE
    assert found["target"] is Attribution.SEED
    assert classify(_direct(PRAM_ARM, perturb=perturb), mapped=MAPPED)["chol"] is Attribution.SEED
    assert arm_kind(_direct(f"{COHERENT_ARM}-replica", perturb=perturb)).value == "perturbation"


def _every_unit(traced: list[dict], groups: tuple[tuple[str, ...], ...]) -> Counter[int]:
    """The donations a fill's trace lines name: each unit read whole from one record."""
    found: Counter[int] = Counter()
    for line in traced:
        sources = line["sources"]
        found.update(
            {
                sources[unit[0]]
                for unit in groups
                if unit[0] in sources and len({sources.get(name) for name in unit}) == 1
            }
        )
    return found


def test_donor_counts_read_the_leaf_donors(cohered):
    folder, ctx = cohered
    code, traced = _trace(folder, ctx, COHERENT_ARM)
    assert code == 0
    named = _named("g2", ctx)
    rows = load(named.dataset, folder.parent / "data", named).field_rows
    groups = donor_attacks.match_groups(rows, frame=UCI_FRAME)
    found = _donors_of(folder, folder.parent / "data", arm=COHERENT_ARM, ctx=ctx)
    assert found == _every_unit(traced, groups)
    assert found != _sourced(traced, groups)


def test_donor_counts_refuse_a_stored_row_its_change_re_run_does_not_give(cohered, monkeypatch):
    folder, ctx = cohered
    monkeypatch.setattr(donor_attacks, "coherent_draws", _altered(donor_attacks.coherent_draws))
    with pytest.raises(IdentityError, match="columns differ from its change re-run"):
        _donors_of(folder, folder.parent / "data", arm=COHERENT_ARM, ctx=ctx)


def test_e17s_runs_are_e9s_with_the_coherent_arm_against_bko_direct():
    for fraction, e9 in (("thin", E9_THIN), ("empty", E9_EMPTY)):
        run = RUNS[f"e17-{fraction}"]
        assert run.experiment == dataclasses.replace(
            e9,
            arms=("none", "real", DIRECT_ARM, COHERENT_ARM),
            candidate=COHERENT_ARM,
            comparators=(DIRECT_ARM, "none"),
        )
        assert run.dataset == RUNS[f"e9-{fraction}"].dataset
        assert REFERENCE[f"e17-{fraction}"] == DIRECT_ARM
