"""Score one synthetic table under SynthEval and write each metric's value as one JSON object.

`python bench/syntheval/score.py --real R.csv --test T.csv --synthetic S.csv --target NAME
--categorical NAME... --metrics M.json --seed N --out O.json` reads the three tables, each
categorical column as text, seeds Python's and NumPy's global random state with N, runs
`SynthEval(real, test, cat_cols=..., verbose=False, console="off", enable_plots=False)
.evaluate(synthetic, analysis_target=NAME, **metrics)`, where `metrics` is the keyword dictionary
read from M.json, and writes the metric name to value mapping SynthEval returns to O.json, a
non-finite value as null. SynthEval draws its membership-inference split and its models' seeds
from NumPy's global state, so one seed gives one O.json. SynthEval reports a metric that raises on
stdout and carries on, so the script exits 1, naming it on stderr, when a metric of M.json
returned no result. It runs in O.json's directory, where SynthEval writes its analysis
configuration. It runs in `bench/syntheval/.venv` (`make syntheval-env`) and imports no `bko`;
`bko.evaluate` runs it as a subprocess (ADR `tooling/benchmark-tools-outside-the-lock`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
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
    found.add_argument("--seed", type=int, required=True, help="the global random state's seed")
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
    # SynthEval 1.7.3 records each metric that returned under the private `_raw_results`, keyed
    # by the metric's name; a metric that raised is absent. Its public table names no metric key.
    failed = sorted(set(metrics) - set(evaluator._raw_results))
    if failed:
        msg = f"SynthEval returned no result for {failed}"
        raise ValueError(msg)
    return collected(table["metric"], table["val"])


def collected(names: Iterable[object], values: Iterable[object]) -> dict[str, float]:
    """Each value of `values` by its name in `names`, as a float.

    Raises:
        ValueError: If a name comes twice.
    """
    found: dict[str, float] = {}
    for name, value in zip(names, values, strict=True):
        if str(name) in found:
            msg = f"SynthEval returned the metric {name!r} twice"
            raise ValueError(msg)
        found[str(name)] = float(value)
    return found


def encoded(found: dict[str, float]) -> str:
    """`found` as one JSON object with sorted keys, a non-finite value as null."""
    return json.dumps(
        {name: value if math.isfinite(value) else None for name, value in found.items()},
        sort_keys=True,
    )


def read(path: Path, categorical: Sequence[str]) -> pd.DataFrame:
    """The CSV at `path`, each of `categorical` read as text, so every table reads alike."""
    return pd.read_csv(path, dtype=dict.fromkeys(categorical, str))


def main(argv: Sequence[str] | None = None) -> int:
    """Score the tables the arguments name and write the JSON; return the exit code."""
    args = parser().parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    tables = [read(path, args.categorical) for path in (args.real, args.test, args.synthetic)]
    out = args.out.resolve()
    os.chdir(out.parent)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        found = score(*tables, target=args.target, categorical=args.categorical, metrics=metrics)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    out.write_text(encoded(found) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
