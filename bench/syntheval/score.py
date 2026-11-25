"""Score one synthetic table under SynthEval and write each metric's value as one JSON object.

`python bench/syntheval/score.py --real R.csv --test T.csv --synthetic S.csv --target NAME
--categorical NAME... --metrics M.json --out O.json` reads the three tables, runs
`SynthEval(real, test, cat_cols=..., verbose=False, console="off", enable_plots=False)
.evaluate(synthetic, analysis_target=NAME, **metrics)`, where `metrics` is the keyword dictionary
read from M.json, and writes the metric name to value mapping SynthEval returns to O.json, a
non-finite value as null. SynthEval reports a metric that raises on stdout and carries on, so the
script exits 1, naming it on stderr, when a metric of M.json returned no result. It runs in O.json's
directory, where SynthEval writes its analysis configuration. It runs in `bench/syntheval/.venv`
(`make syntheval-env`) and imports no `bko`; `bko.evaluate` runs it as a subprocess (ADR
`tooling/benchmark-tools-outside-the-lock`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from syntheval import SynthEval


def parser() -> argparse.ArgumentParser:
    """The script's arguments."""
    found = argparse.ArgumentParser(description=__doc__)
    found.add_argument("--real", type=Path, required=True, help="the real (training) table")
    found.add_argument("--test", type=Path, required=True, help="the holdout (test) table")
    found.add_argument("--synthetic", type=Path, required=True, help="the synthetic table")
    found.add_argument("--target", required=True, help="the analysis target column")
    found.add_argument("--categorical", nargs="+", required=True, help="the categorical columns")
    found.add_argument("--metrics", type=Path, required=True, help="the metric keyword JSON")
    found.add_argument("--out", type=Path, required=True, help="the JSON file to write")
    return found


def score(
    real: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target: str,
    categorical: Sequence[str],
    metrics: dict[str, dict[str, object]],
) -> dict[str, float]:
    """Each metric SynthEval returns for `synthetic`, by name.

    Raises:
        ValueError: If a metric of `metrics` returned no result, or SynthEval returns a value's
            name twice.
    """
    evaluator = SynthEval(
        real, test, cat_cols=list(categorical), verbose=False, console="off", enable_plots=False
    )
    table = evaluator.evaluate(synthetic, analysis_target=target, **metrics)
    failed = sorted(set(metrics) - set(evaluator._raw_results))
    if failed:
        msg = f"SynthEval returned no result for {failed}"
        raise ValueError(msg)
    found: dict[str, float] = {}
    for name, value in zip(table["metric"], table["val"], strict=True):
        if name in found:
            msg = f"SynthEval returned the metric {name!r} twice"
            raise ValueError(msg)
        found[str(name)] = float(value)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """Score the tables the arguments name and write the JSON; return the exit code."""
    args = parser().parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    tables = [pd.read_csv(path) for path in (args.real, args.test, args.synthetic)]
    out = args.out.resolve()
    os.chdir(out.parent)
    try:
        found = score(*tables, target=args.target, categorical=args.categorical, metrics=metrics)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    text = json.dumps(
        {name: value if math.isfinite(value) else None for name, value in found.items()},
        sort_keys=True,
    )
    out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
