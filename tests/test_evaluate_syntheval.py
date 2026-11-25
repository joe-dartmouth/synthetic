"""`syntheval`: each replica fill of a run's store scored by SynthEval through a runner.

The gate never imports SynthEval: a fake runner stands in for `bench/syntheval/score.py`, whose
real test runs under `make syntheval-test` (ADR `tooling/benchmark-tools-outside-the-lock`).
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import io
import json
import logging
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

import bko.evaluate.syntheval_reading as reading
from bko.baselines.request import Method
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.arguments import parser
from bko.evaluate.data import Dataset, load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, E1
from bko.evaluate.named_run import BAYES_NET, NEW_METHODS
from bko.evaluate.named_runs import RUNS
from bko.evaluate.rows_read import bind, read_rows
from bko.evaluate.run_folder import read_folder
from bko.evaluate.runs_published import P1_ARMS, P2
from bko.evaluate.syntheval_inputs import METRICS, OUT, fills, score_argv
from bko.evaluate.syntheval_reading import COLUMNS, run_syntheval
from bko.generate.messages import SETTINGS
from tests.test_evaluate_membership import _context, _stored
from tests.test_evaluate_rows import BASELINE


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bench" / "syntheval" / "score.py"

CANNED = {"pca_eigval_diff": 0.25, "h_dist": 0.125, "mia_recall": None}
"""What the fake script writes: two numbers and a metric SynthEval could not compute."""


class FakeRunner:
    """Stands in for the script: records each argv, checks its inputs, writes `CANNED`."""

    def __init__(self, *, code: int = 0, stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.inputs: list[dict[str, pd.DataFrame]] = []
        self.code = code
        self.stderr = stderr

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        flags = {name: argv[argv.index(name) + 1] for name in ("--real", "--test", "--synthetic")}
        self.inputs.append({name: pd.read_csv(path) for name, path in flags.items()})
        metrics = json.loads(Path(argv[argv.index("--metrics") + 1]).read_text(encoding="utf-8"))
        assert metrics == {name: dict(value) for name, value in METRICS.items()}
        out = Path(argv[argv.index("--out") + 1])
        if self.code == 0:
            out.write_text(json.dumps(CANNED), encoding="utf-8")
        return subprocess.CompletedProcess(list(argv), self.code, "", self.stderr)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A `bench/syntheval/` whose environment holds an interpreter, which no test runs."""
    found = tmp_path / "bench"
    python = found / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(reading, "bench_dir", lambda: found)
    return found


def _args(tmp_path: Path, folder: Path, store: Path):
    argv = ["syntheval", "--folder", str(folder), "--rows", str(store)]
    return parser(None).parse_args([*argv, "--data-dir", str(tmp_path / "data")])


def test_fills_yields_each_replica_fill_with_its_holdouts_splits(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    data = load(ctx.dataset, tmp_path / "data", ctx)
    parsed, rows = read_folder(folder), read_rows(store)
    bind(rows, parsed, data.frame, run="e1", data=data.digest)
    found = list(fills(parsed, rows, data, BASELINE))
    assert [(fill.holdout, fill.arm) for fill in found] == [
        (0, "smote-replica"),
        (1, "smote-replica"),
    ]
    for fill in found:
        line = parsed.arms[fill.holdout, fill.arm]
        assert f"added={len(fill.synthetic)} " in line
        assert list(fill.real.columns) == list(UCI_FRAME.columns)
        assert list(fill.test.columns) == list(UCI_FRAME.columns)
        assert f"train={len(fill.real)} test={len(fill.test)}" in parsed.splits[fill.holdout]
        assert not set(fill.real.index) & set(fill.test.index)


def test_the_command_runs_the_script_once_per_fill_and_writes_its_metrics(tmp_path, caplog, bench):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    runner = FakeRunner()
    caplog.set_level(logging.INFO)
    text = run_syntheval(_args(tmp_path, folder, store), ctx, runner=runner)
    assert len(runner.calls) == 2
    assert all(
        call[:2] == [str(bench / ".venv/bin/python"), str(bench / "score.py")]
        for call in runner.calls
    )
    table = list(csv.reader(io.StringIO(text)))
    assert tuple(table[0]) == COLUMNS
    assert table[1:] == [
        [str(seed), "smote-replica", name, "" if value is None else repr(value)]
        for seed in (0, 1)
        for name, value in CANNED.items()
    ]
    for inputs in runner.inputs:
        assert list(inputs["--synthetic"].columns) == list(UCI_FRAME.columns)
    lines = [record.getMessage() for record in caplog.records]
    assert (
        sum(line.startswith("syntheval input holdout=0 arm=smote-replica file=") for line in lines)
        == 4
    )
    assert any(
        re.fullmatch(r"syntheval holdout=1 arm=smote-replica rows=\d+ metrics=3", line)
        for line in lines
    )
    assert re.fullmatch(r"syntheval rows=6 sha256=[0-9a-f]{64}", lines[-1])


def test_the_score_argv_names_the_envs_python_and_the_script():
    argv = score_argv(
        python=Path("/b/.venv/bin/python"),
        script=Path("/b/score.py"),
        inputs=Path("/t"),
        target="target",
        categorical=["sex", "cp", "target"],
    )
    assert argv == [
        "/b/.venv/bin/python",
        "/b/score.py",
        "--real",
        "/t/real.csv",
        "--test",
        "/t/test.csv",
        "--synthetic",
        "/t/synthetic.csv",
        "--target",
        "target",
        "--categorical",
        "sex",
        "cp",
        "target",
        "--metrics",
        "/t/metrics.json",
        "--out",
        f"/t/{OUT}",
    ]


def test_a_missing_environment_exits_2_naming_the_make_target(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(reading, "bench_dir", lambda: tmp_path / "bench")
    ctx = _context(tmp_path, experiment=BASELINE)
    caplog.set_level(logging.ERROR)
    argv = ["syntheval", "--folder", str(tmp_path / "run"), "--rows", str(tmp_path / "r.jsonl")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert "make syntheval-env" in caplog.text


def test_a_failing_script_exits_2_with_its_stderr(tmp_path, caplog, bench):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    stderr = "".join(f"line {n}\n" for n in range(30)) + "ValueError: no target\n"
    runner = FakeRunner(code=1, stderr=stderr)
    with pytest.raises(EvaluationError, match="ValueError: no target") as raised:
        run_syntheval(_args(tmp_path, folder, store), ctx, runner=runner)
    assert raised.value.exit_code == 2
    assert "line 10\n" not in str(raised.value)
    assert "line 11\n" in str(raised.value)
    assert len(runner.calls) == 1


@pytest.mark.parametrize("written", ["[]", '{"h_dist": "high"}', '{"h_dist": true}', "{"])
def test_metrics_that_are_not_names_to_numbers_are_refused(tmp_path, caplog, bench, written):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        Path(argv[argv.index("--out") + 1]).write_text(written, encoding="utf-8")
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    with pytest.raises(EvaluationError, match="metrics"):
        run_syntheval(_args(tmp_path, folder, store), ctx, runner=runner)


def test_the_metric_set_is_the_stated_dictionary():
    assert {name: dict(value) for name, value in METRICS.items()} == {
        "pca": {},
        "h_dist": {},
        "corr_diff": {"mixed_corr": True},
        "auroc_diff": {"model": "rf_cls"},
        "cls_acc": {"F1_type": "macro"},
        "eps_risk": {},
        "dcr": {},
        "mia": {"num_eval_iter": 5},
    }


def test_p2_names_the_stated_configuration():
    arms = (
        "none",
        "class-weight",
        "smote",
        "adasyn",
        "ctgan",
        "tvae",
        *NEW_METHODS,
        BAYES_NET,
        "tabddpm",
        "label-bootstrap",
        DIRECT_ARM,
    )
    rowless = {str(Method.NONE), str(Method.CLASS_WEIGHT)}
    stated = dataclasses.replace(
        E1,
        arms=arms,
        candidate=DIRECT_ARM,
        comparators=tuple(arm for arm in arms if arm != DIRECT_ARM),
        replica_arms=tuple(arm for arm in arms if arm not in rowless),
        criteria=(),
    )
    assert stated == P2
    assert P2.arms == P1_ARMS
    assert (P2.holdout_seeds, P2.decontaminate) == (E1.holdout_seeds, E1.decontaminate)
    assert RUNS["p2"].experiment == P2
    assert RUNS["p2"].settings == SETTINGS
    assert RUNS["p2"].dataset is Dataset.UCI


def test_the_score_script_parses_and_imports_no_bko():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    modules |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "syntheval" in modules
    assert not any(module == "bko" or module.startswith("bko.") for module in modules)
