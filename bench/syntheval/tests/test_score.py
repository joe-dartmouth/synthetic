"""`bench/syntheval/score.py` under the real SynthEval, run by `make syntheval-test`.

Not part of the gate, which never imports SynthEval: this runs the script with the interpreter of
`bench/syntheval/.venv` (`make syntheval-env`) on a tiny fixture fill under the metric set
`bko.evaluate.syntheval_inputs.METRICS` (ADR `tooling/benchmark-tools-outside-the-lock`).
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from bko.evaluate.syntheval_inputs import METRICS


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


def test_the_returned_names_cover_the_metric_set():
    assert set(RETURNED) == set(METRICS)


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


def run_script(tmp_path: Path, metrics: dict) -> subprocess.CompletedProcess[str]:
    """The script run on the fixture fill under `metrics`, writing `out.json` in `tmp_path`."""
    assert PYTHON.is_file(), "no bench/syntheval/.venv; run make syntheval-env"
    paths = {name: tmp_path / f"{name}.csv" for name in ("real", "test", "synthetic")}
    for path, (rows, seed) in zip(paths.values(), ((40, 1), (10, 2), (40, 3)), strict=True):
        records(rows, seed).to_csv(path, index=False)
    written = tmp_path / "metrics.json"
    written.write_text(json.dumps(metrics), encoding="utf-8")
    argv = [str(PYTHON), str(SCRIPT)]
    argv += ["--real", str(paths["real"]), "--test", str(paths["test"])]
    argv += ["--synthetic", str(paths["synthetic"]), "--target", "target"]
    argv += ["--categorical", *CATEGORICAL, "--metrics", str(written)]
    argv += ["--out", str(tmp_path / "out.json")]
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, timeout=600
    )


def test_a_tiny_fixture_fill_scores_every_metric(tmp_path):
    done = run_script(tmp_path, {k: dict(v) for k, v in METRICS.items()})
    assert done.returncode == 0, done.stderr[-4000:]
    found = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert set(found) == {name for names in RETURNED.values() for name in names}
    assert all(value is not None and math.isfinite(value) for value in found.values()), found
    assert (tmp_path / "SE_analysis_config.json").is_file()
    assert not (BENCH.parents[1] / "SE_analysis_config.json").exists()


def test_a_metric_that_fails_exits_1_naming_it(tmp_path):
    done = run_script(tmp_path, {"h_dist": {}, "auroc_diff": {"model": "no_such_model"}})
    assert done.returncode == 1
    assert "auroc_diff" in done.stderr
    assert not (tmp_path / "out.json").exists()
