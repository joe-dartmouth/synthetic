"""`bench/syntheval/score.py` under the real SynthEval, run by `make syntheval-test`.

Not part of the gate, which never imports SynthEval: this runs the script with the interpreter of
`bench/syntheval/.venv` (`make syntheval-env`) on a tiny fixture fill under the metric set
`bko.evaluate.syntheval_inputs.METRICS`, through the argv `score_argv` builds (ADR
`tooling/benchmark-tools-outside-the-lock`).
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from bko.evaluate.syntheval_inputs import METRICS, OUT, score_argv


BENCH = Path(__file__).resolve().parents[1]
PYTHON = BENCH / ".venv" / "bin" / "python"
SCRIPT = BENCH / "score.py"
CATEGORICAL = ("sex", "cp", "fbs", "target")

RETURNED = {
    "pca": ("pca_eigval_diff", "pca_eigvec_ang"),
    "h_dist": ("avg_h_dist",),
    "corr_diff": ("corr_mat_diff",),
    "auroc_diff": ("auroc",),
    "cls_acc": ("avg_F1_diff", "avg_F1_diff_hout"),
    "eps_risk": ("eps_identif_risk", "priv_loss_eps"),
    "dcr": ("median_DCR",),
    "mia": ("mia_precision", "mia_recall"),
}
"""The names SynthEval 1.7.3 returns for each metric of the set."""

VERSION = "syntheval==1.7.3"
"""The pin `RETURNED` was read under."""


def test_the_returned_names_cover_the_metric_set():
    assert set(RETURNED) == set(METRICS)


def test_the_returned_names_are_read_under_the_pinned_version():
    requirements = (BENCH / "requirements.txt").read_text(encoding="utf-8")
    pins = [line.split()[0] for line in requirements.splitlines() if "==" in line]
    assert VERSION in pins


def records(rows: int, seed: int) -> pd.DataFrame:
    """`rows` UCI-shaped records drawn at `seed`, both labels present."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(29, 78, rows),
            "sex": rng.integers(0, 2, rows),
            "cp": rng.integers(1, 5, rows),
            "trestbps": rng.integers(94, 200, rows),
            "chol": rng.integers(126, 565, rows),
            "fbs": rng.integers(0, 2, rows),
            "oldpeak": np.round(rng.uniform(0, 6.2, rows), 1),
            "target": np.arange(rows) % 2,
        }
    )


TINY = (40, 10, 40)
"""The fixture fill's real, test and synthetic rows."""

LARGER = (200, 120, 200)
"""Rows enough that the membership-inference model splits, so its draws move its scores."""


def run_script(
    tmp_path: Path, metrics: dict, *, seed: int = 0, sizes: tuple[int, int, int] = TINY
) -> subprocess.CompletedProcess[str]:
    """The script run on the fixture fill under `metrics`, writing `OUT` in `tmp_path`."""
    assert PYTHON.is_file(), "no bench/syntheval/.venv; run make syntheval-env"
    for name, rows, drawn in zip(("real", "test", "synthetic"), sizes, (1, 2, 3), strict=True):
        records(rows, drawn).to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    argv = score_argv(
        python=PYTHON,
        script=SCRIPT,
        inputs=tmp_path,
        target="target",
        categorical=CATEGORICAL,
        seed=seed,
    )
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, timeout=600
    )


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    """`code` run under the environment's interpreter with the script importable as `score`."""
    assert PYTHON.is_file(), "no bench/syntheval/.venv; run make syntheval-env"
    prelude = f"import sys; sys.path.insert(0, {str(BENCH)!r}); import score\n"
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [str(PYTHON), "-c", prelude + code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def every_metric() -> dict[str, dict[str, object]]:
    return {name: dict(value) for name, value in METRICS.items()}


def test_a_tiny_fixture_fill_scores_every_metric(tmp_path):
    done = run_script(tmp_path, every_metric())
    assert done.returncode == 0, done.stderr[-4000:]
    found = json.loads((tmp_path / OUT).read_text(encoding="utf-8"))
    assert set(found) == {name for names in RETURNED.values() for name in names}
    assert all(value is not None and math.isfinite(value) for value in found.values()), found
    assert (tmp_path / "SE_analysis_config.json").is_file()
    assert not (BENCH.parents[1] / "SE_analysis_config.json").exists()


def test_one_seed_writes_one_json(tmp_path):
    texts = []
    for name, seed in (("first", 5), ("second", 5), ("other", 6)):
        (tmp_path / name).mkdir()
        done = run_script(tmp_path / name, {"mia": {"num_eval_iter": 5}}, seed=seed, sizes=LARGER)
        assert done.returncode == 0, done.stderr[-4000:]
        texts.append((tmp_path / name / OUT).read_text(encoding="utf-8"))
    assert texts[0] == texts[1]
    assert texts[0] != texts[2]


def test_a_metric_that_fails_exits_1_naming_only_it(tmp_path):
    done = run_script(tmp_path, {"h_dist": {}, "auroc_diff": {"model": "no_such_model"}})
    assert done.returncode == 1
    assert done.stderr.splitlines()[-1] == "SynthEval returned no result for ['auroc_diff']"
    assert not (tmp_path / OUT).exists()


def test_a_non_finite_value_is_written_as_null():
    done = run_python("print(score.encoded({'b': float('nan'), 'a': float('inf'), 'c': 0.5}))")
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout == '{"a": null, "b": null, "c": 0.5}\n'


def test_a_name_returned_twice_is_refused():
    done = run_python(
        "try:\n"
        "    score.collected(['a', 'b', 'a'], [1.0, 2.0, 3.0])\n"
        "except ValueError as exc:\n"
        "    print(exc)\n"
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout == "SynthEval returned the metric 'a' twice\n"


def test_categorical_columns_read_as_text(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("cp,age\n1,50\n4,61\n", encoding="utf-8")
    done = run_python(
        f"frame = score.read(__import__('pathlib').Path({str(path)!r}), ['cp'])\n"
        "print(list(frame['cp']), str(frame['age'].dtype))\n"
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout == "['1', '4'] int64\n"
