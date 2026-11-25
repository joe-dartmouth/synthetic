"""`syntheval`: each replica fill of a run's store scored by SynthEval through a runner.

The gate never imports SynthEval: a fake runner stands in for `bench/syntheval/score.py`, whose
real test runs under `make syntheval-test` (ADR `tooling/benchmark-tools-outside-the-lock`).
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import io
import json
import logging
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

import bko.evaluate.syntheval_inputs as inputs_module
import bko.evaluate.syntheval_reading as reading
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.arguments import parser
from bko.evaluate.data import Dataset, load
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import E1
from bko.evaluate.named_runs import RUNS
from bko.evaluate.rows_read import bind, frame_of, read_rows
from bko.evaluate.run_folder import fields, read_folder
from bko.evaluate.runs_published import P2
from bko.evaluate.syntheval_inputs import METRICS, OUT, SynthevalFill, fills, score_argv
from bko.evaluate.syntheval_reading import COLUMNS, TIMEOUT_SECONDS, run_script, run_syntheval
from bko.generate.messages import SETTINGS
from tests.test_evaluate_membership import _context, _stored
from tests.test_evaluate_rows import BASELINE


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bench" / "syntheval" / "score.py"

CANNED = {"pca_eigval_diff": 0.25, "h_dist": 0.125, "mia_recall": None}
"""What the fake script writes: two numbers and a metric SynthEval could not compute."""

STATED = {
    "pca": {},
    "h_dist": {},
    "corr_diff": {"mixed_corr": True},
    "auroc_diff": {"model": "rf_cls"},
    "cls_acc": {"F1_type": "macro"},
    "eps_risk": {},
    "dcr": {},
    "mia": {"num_eval_iter": 5},
}
"""The metric set ADR `research/syntheval-reading` states (Decision 3)."""

UCI_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
"""The `--categorical` columns of a UCI reading: the frame's coded predictors and its label."""

SPLIT = re.compile(r"split seed=(\d+) train=(\d+) test=(\d+) fingerprint=\S+")
"""A `split` line's fields."""


def _flag(argv: Sequence[str], name: str) -> str:
    return argv[argv.index(name) + 1]


class FakeRunner:
    """Stands in for the script: records each argv, checks its inputs, writes `written`.

    With `expected`, the i-th call's three tables must equal the i-th fill's.
    """

    def __init__(
        self,
        *,
        code: int = 0,
        stdout: str = "",
        stderr: str = "",
        written: str = json.dumps(CANNED),
        expected: Sequence[SynthevalFill] = (),
    ) -> None:
        self.calls: list[list[str]] = []
        self.inputs: list[dict[str, pd.DataFrame]] = []
        self.shas: dict[tuple[str, str], str] = {}
        self.code, self.stdout, self.stderr, self.written = code, stdout, stderr, written
        self.expected = list(expected)

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        assert _flag(argv, "--target") == "target"
        start = argv.index("--categorical") + 1
        assert list(argv[start : argv.index("--metrics")]) == UCI_CATEGORICAL
        names = ("--real", "--test", "--synthetic", "--metrics")
        for name in names:
            path = Path(_flag(argv, name))
            self.shas[_flag(argv, "--seed"), path.name] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        tables = {
            name: pd.read_csv(_flag(argv, name), float_precision="round_trip") for name in names[:3]
        }
        self.inputs.append(tables)
        if self.expected:
            fill = self.expected[len(self.calls) - 1]
            for name, table in (("--real", fill.real), ("--test", fill.test)):
                pd.testing.assert_frame_equal(tables[name], table.reset_index(drop=True))
            pd.testing.assert_frame_equal(
                tables["--synthetic"], fill.synthetic.reset_index(drop=True)
            )
        metrics = json.loads(Path(_flag(argv, "--metrics")).read_text(encoding="utf-8"))
        assert metrics == STATED
        if self.code == 0:
            Path(_flag(argv, "--out")).write_text(self.written, encoding="utf-8")
        return subprocess.CompletedProcess(list(argv), self.code, self.stdout, self.stderr)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A `bench/syntheval/` whose environment holds an interpreter, which no test runs."""
    found = tmp_path / "bench"
    python = found / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(reading, "bench_dir", lambda: found)
    return found


def _args(tmp_path: Path, folder: Path, store: Path, *extra: str):
    argv = ["syntheval", "--folder", str(folder), "--rows", str(store), *extra]
    return parser(None).parse_args([*argv, "--data-dir", str(tmp_path / "data")])


def _bound(tmp_path: Path, caplog):
    """BASELINE's folder and store, read, bound and with the data they were scored on."""
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    data = load(ctx.dataset, tmp_path / "data", ctx)
    parsed, rows = read_folder(folder), read_rows(store)
    bind(rows, parsed, data.frame, run="e1", data=data.digest)
    return parsed, rows, data


def test_fills_yields_each_replica_fill_with_its_holdouts_splits(tmp_path, caplog):
    parsed, rows, data = _bound(tmp_path, caplog)
    found = list(fills(parsed, rows, data, BASELINE))
    assert [(fill.holdout, fill.arm) for fill in found] == [
        (0, "smote-replica"),
        (1, "smote-replica"),
    ]
    for fill in found:
        line = parsed.arms[fill.holdout, fill.arm]
        assert int(fields(line).group("added")) == len(fill.synthetic)
        assert list(fill.real.columns) == list(UCI_FRAME.columns)
        assert list(fill.test.columns) == list(UCI_FRAME.columns)
        split = SPLIT.fullmatch(parsed.splits[fill.holdout])
        assert split is not None
        assert split.groups()[:3] == (str(fill.holdout), str(len(fill.real)), str(len(fill.test)))
        assert not set(fill.real.index) & set(fill.test.index)


def test_a_decontaminating_run_scores_the_test_split_without_copied_records(
    tmp_path, caplog, monkeypatch
):
    parsed, rows, data = _bound(tmp_path, caplog)
    kept = {fill.holdout: fill.test for fill in fills(parsed, rows, data, BASELINE)}
    columns = list(UCI_FRAME.columns)
    copied = {holdout: test.iloc[:2] for holdout, test in kept.items()}

    def copying(line, *, arm_line, frame):
        stored = frame_of(line, arm_line=arm_line, frame=frame)
        if line.arm != "smote":
            return stored
        return pd.concat([stored, copied[line.holdout]])

    monkeypatch.setattr(inputs_module, "frame_of", copying)
    found = list(fills(parsed, rows, data, dataclasses.replace(BASELINE, decontaminate=True)))
    assert [fill.holdout for fill in found] == [0, 1]
    for fill in found:
        gone = set(copied[fill.holdout][columns].itertuples(index=False))
        assert not gone & set(fill.test[columns].itertuples(index=False))
        whole = list(kept[fill.holdout][columns].itertuples(index=False))
        assert len(fill.test) == sum(record not in gone for record in whole)
        assert len(fill.test) < len(kept[fill.holdout])


def test_a_replica_arm_with_no_stored_fill_is_refused_naming_it(tmp_path, caplog):
    parsed, rows, data = _bound(tmp_path, caplog)
    missing = (1, "smote-replica")
    rows = dataclasses.replace(rows, lines={k: v for k, v in rows.lines.items() if k != missing})
    parsed = dataclasses.replace(
        parsed, arms={k: v for k, v in parsed.arms.items() if k != missing}
    )
    with pytest.raises(
        IdentityError, match="holdout 1's smote-replica has no stored fill"
    ) as raised:
        list(fills(parsed, rows, data, BASELINE))
    assert raised.value.exit_code == 3


def test_a_holdout_the_folder_never_split_is_refused_naming_it(tmp_path, caplog):
    parsed, rows, data = _bound(tmp_path, caplog)
    wider = dataclasses.replace(BASELINE, holdout_seeds=(0, 7))
    with pytest.raises(IdentityError, match="no split line for holdout 7"):
        list(fills(parsed, rows, data, wider))


def test_the_command_runs_the_script_once_per_fill_and_writes_its_metrics(tmp_path, caplog, bench):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    data = load(ctx.dataset, tmp_path / "data", ctx)
    parsed, rows = read_folder(folder), read_rows(store)
    expected = list(fills(parsed, rows, data, BASELINE))
    runner = FakeRunner(expected=expected)
    caplog.set_level(logging.INFO)
    caplog.clear()
    text = run_syntheval(_args(tmp_path, folder, store), ctx, runner=runner)
    assert len(runner.calls) == 2
    assert all(
        call[:2] == [str(bench / ".venv/bin/python"), str(bench / "score.py")]
        for call in runner.calls
    )
    assert [_flag(call, "--seed") for call in runner.calls] == ["0", "1"]
    table = list(csv.reader(io.StringIO(text)))
    assert tuple(table[0]) == COLUMNS
    assert table[1:] == [
        [str(seed), "smote-replica", name, "" if value is None else repr(value)]
        for seed in (0, 1)
        for name, value in CANNED.items()
    ]
    lines = [record.getMessage() for record in caplog.records]
    logged = {}
    for line in lines:
        found = re.fullmatch(
            r"syntheval input holdout=(\d+) arm=smote-replica file=(\S+) sha256=([0-9a-f]{64})",
            line,
        )
        if found:
            logged[found[1], found[2]] = found[3]
    assert logged == runner.shas
    assert len(logged) == 8
    first_input = next(i for i, line in enumerate(lines) if line.startswith("syntheval input"))
    assert lines[first_input - 1] == "syntheval start holdout=0 arm=smote-replica"
    assert "syntheval start holdout=1 arm=smote-replica" in lines
    assert any(
        re.fullmatch(r"syntheval holdout=1 arm=smote-replica rows=\d+ metrics=3", line)
        for line in lines
    )
    assert re.fullmatch(r"syntheval rows=6 sha256=[0-9a-f]{64}", lines[-1])


def test_the_bench_dir_flag_names_the_environment_and_script(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(reading, "bench_dir", lambda: tmp_path / "absent")
    other = tmp_path / "other"
    python = other / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    runner = FakeRunner()
    run_syntheval(_args(tmp_path, folder, store, "--bench-dir", str(other)), ctx, runner=runner)
    assert [call[:2] for call in runner.calls] == [[str(python), str(other / "score.py")]] * 2


def test_the_score_argv_names_the_envs_python_and_the_script():
    argv = score_argv(
        python=Path("/b/.venv/bin/python"),
        script=Path("/b/score.py"),
        inputs=Path("/t"),
        target="target",
        categorical=["sex", "cp", "target"],
        seed=7,
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
        "--seed",
        "7",
        "--out",
        f"/t/{OUT}",
    ]


def _script_flags() -> list[str]:
    """Each flag `score.py`'s parser adds, read from its source."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    return [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


def test_the_score_argv_passes_exactly_the_scripts_flags():
    argv = score_argv(
        python=Path("p"), script=Path("s"), inputs=Path("i"), target="t", categorical=["c"], seed=0
    )
    flags = [item for item in argv if item.startswith("--")]
    assert sorted(flags) == sorted(_script_flags())
    assert len(set(flags)) == len(flags)


def test_a_missing_environment_exits_2_naming_the_make_target(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(reading, "bench_dir", lambda: tmp_path / "bench")
    ctx = _context(tmp_path, experiment=BASELINE)
    caplog.set_level(logging.ERROR)
    argv = ["syntheval", "--folder", str(tmp_path / "run"), "--rows", str(tmp_path / "r.jsonl")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert "make syntheval-env" in caplog.text


def test_a_failing_script_exits_2_with_its_stdout_and_stderr(tmp_path, caplog, bench):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    stderr = "".join(f"line {n}\n" for n in range(30)) + "ValueError: no target\n"
    stdout = "".join(f"out {n}\n" for n in range(30)) + "auroc_diff failed\n"
    runner = FakeRunner(code=1, stdout=stdout, stderr=stderr)
    with pytest.raises(EvaluationError, match="ValueError: no target") as raised:
        run_syntheval(_args(tmp_path, folder, store), ctx, runner=runner)
    assert raised.value.exit_code == 2
    message = str(raised.value)
    assert "line 10\n" not in message
    assert "line 11\n" in message
    assert "auroc_diff failed" in message
    assert "out 10\n" not in message
    assert "out 11\n" in message
    assert len(runner.calls) == 1


@pytest.mark.parametrize("written", ["[]", '{"h_dist": "high"}', '{"h_dist": true}', "{"])
def test_metrics_that_are_not_names_to_numbers_are_refused(tmp_path, caplog, bench, written):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    runner = FakeRunner(written=written)
    with pytest.raises(EvaluationError, match="metrics"):
        run_syntheval(_args(tmp_path, folder, store), ctx, runner=runner)


def test_non_finite_metrics_are_written_blank(tmp_path, caplog, bench):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)
    written = '{"a": NaN, "b": Infinity, "c": -Infinity, "d": 0.5}'
    text = run_syntheval(_args(tmp_path, folder, store), ctx, runner=FakeRunner(written=written))
    table = list(csv.reader(io.StringIO(text)))
    assert table[1:5] == [["0", "smote-replica", name, value] for name, value in
        (("a", ""), ("b", ""), ("c", ""), ("d", "0.5"))]  # fmt: skip


def test_a_script_that_exits_0_without_its_json_is_refused(tmp_path, caplog, bench):
    folder, store = _stored(tmp_path, caplog, experiment=BASELINE)
    ctx = _context(tmp_path, experiment=BASELINE)

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    with pytest.raises(EvaluationError, match="no readable metrics"):
        run_syntheval(_args(tmp_path, folder, store), ctx, runner=runner)


def test_the_script_gets_no_credential_and_a_timeout(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("HOME", "/home/x")
    seen = {}

    def fake(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(reading.subprocess, "run", fake)
    run_script(["python"])
    assert seen["env"] == {
        "PATH": "/bin",
        "HOME": "/home/x",
        "TZ": "UTC",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    assert seen["timeout"] == TIMEOUT_SECONDS


def test_a_script_that_runs_too_long_is_stopped(monkeypatch):
    def fake(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(reading.subprocess, "run", fake)
    with pytest.raises(EvaluationError, match="longer than") as raised:
        run_script(["python"])
    assert raised.value.exit_code == 2


def test_the_metric_set_is_the_stated_dictionary():
    assert {name: dict(value) for name, value in METRICS.items()} == STATED


def test_p2_names_the_stated_configuration():
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
    assert P2.arms == arms
    assert P2.candidate == "bko-direct"
    assert P2.comparators == arms[:-1]
    assert P2.replica_arms == arms[2:]
    assert P2.criteria == ()
    assert P2.decontaminate is True
    assert P2.holdout_seeds == tuple(range(100))
    rest = dataclasses.replace(
        P2,
        arms=E1.arms,
        candidate=E1.candidate,
        comparators=E1.comparators,
        replica_arms=E1.replica_arms,
        criteria=E1.criteria,
    )
    assert rest == E1
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
