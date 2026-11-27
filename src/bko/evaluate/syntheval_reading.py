"""`python -m bko.evaluate syntheval RUN --folder FOLDER --rows PATH`: SynthEval's metrics per fill.

`run_syntheval` binds the `rows` store at PATH to FOLDER, and for each replica fill
(`bko.evaluate.syntheval_inputs.fills`) writes the fill's inputs into a temporary directory, logs
each file's sha256, runs `score.py` under the `.venv` of `--bench-dir` (default: this checkout's
`bench/syntheval/`) through an injectable runner, seeded with the fill's holdout seed, and reads
back the metrics it writes. It returns one CSV row per holdout, arm
and metric, and makes no API call. ADR `research/syntheval-reading` records the reading; ADR
`tooling/benchmark-tools-outside-the-lock` the environment, which `make syntheval-env` builds.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import io
import json
import logging
import math
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from bko.datasets.errors import UnavailableError
from bko.datasets.fetch import default_data_dir
from bko.datasets.synthea_run import child_environment
from bko.evaluate.context import Context
from bko.evaluate.data import load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.named_runs import DEFAULT_RUN
from bko.evaluate.provenance import log_versions
from bko.evaluate.rows_read import bind, read_rows
from bko.evaluate.run_folder import read_folder
from bko.evaluate.scoring import data_dir
from bko.evaluate.shared import logged_params
from bko.evaluate.syntheval_inputs import OUT, SynthevalFill, export, fills, score_argv


logger = logging.getLogger("bko.evaluate")

type Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
"""What runs the script: its argv in, the finished process, with its stderr text, out."""

COLUMNS: Final = ("holdout", "arm", "metric", "value")
"""The CSV's columns: one row per holdout, replica fill and metric."""

TAIL: Final = 20
"""The most lines of each of the script's stdout and stderr a failure names."""

TIMEOUT_SECONDS: Final = 3600.0
"""The longest one fill's script may run before the reading stops."""

THREADS: Final[dict[str, str]] = {"OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
"""The thread caps the script gets, beside `child_environment`'s variables; no other is passed."""


def bench_dir() -> Path:
    """`bench/syntheval/` at the root of the checkout this package runs from.

    Raises:
        UnavailableError: As `default_data_dir`, outside a checkout.
    """
    return default_data_dir().parent / "bench" / "syntheval"


def run_script(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run `argv` with no stdin, its stdout and stderr captured as text.

    The script gets `PATH`, `HOME`, the locale variables and `THREADS` only, so no credential
    reaches it.

    Raises:
        EvaluationError: If it runs longer than `TIMEOUT_SECONDS`.
    """
    try:
        return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            list(argv),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            env=child_environment() | THREADS,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"the SynthEval script ran longer than {TIMEOUT_SECONDS:g} s"
        raise EvaluationError(msg) from exc


def run_syntheval(
    args: argparse.Namespace, context: Context, *, runner: Runner = run_script
) -> str:
    """The `syntheval` command over `context`, the context of the run `args.run`; the CSV.

    It logs the versions, `syntheval store=%s sha256=%s`, per fill
    `syntheval start holdout=%d arm=%s`, then, under `Experiment.partial_replica`, for a replica
    that filled partially `replica-short holdout=%d arm=%s rows=%d` from its `arm` line's `short`
    params, as `compare` logs it, per input file
    `syntheval input holdout=%d arm=%s file=%s sha256=%s`, then
    `syntheval holdout=%d arm=%s rows=%d metrics=%d`, and last `syntheval rows=%d sha256=%s`,
    the CSV's rows and sha256. The script runs from `args.bench_dir`, else `bench_dir()`.

    Raises:
        UnavailableError: If the bench directory's `.venv` holds no interpreter, naming
            `make syntheval-env`, before any file is read.
        IdentityError: As `bind`, `fills` and `_log_short`.
        EvaluationError: As `run_script`, or if the script exits non-zero, with the last `TAIL`
            lines of its stdout and its stderr, or writes no JSON object of metric name to number
            or null.
    """
    bench = bench_dir() if args.bench_dir is None else args.bench_dir
    python, script = bench / ".venv" / "bin" / "python", bench / "score.py"
    if not python.is_file():
        msg = f"no SynthEval environment at {python}; run make syntheval-env"
        raise UnavailableError(msg)
    log_versions()
    folder, rows = read_folder(args.folder), read_rows(args.rows)
    logger.info("syntheval store=%s sha256=%s", args.rows, rows.sha256)
    data = load(context.dataset, data_dir(args), context)
    frame = data.frame
    bind(rows, folder, frame, run=args.run, data=data.digest)
    argv = functools.partial(
        score_argv,
        python=python,
        script=script,
        target=frame.label,
        categorical=[*frame.categorical, frame.label],
    )
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(COLUMNS)
    written = 0
    for fill in fills(folder, rows, data, context.experiment):
        logger.info("syntheval start holdout=%d arm=%s", fill.holdout, fill.arm)
        if context.experiment.partial_replica:
            _log_short(fill, folder.arms[fill.holdout, fill.arm])
        metrics = _scored(fill, runner, argv)
        for metric, value in metrics.items():
            writer.writerow((fill.holdout, fill.arm, metric, "" if value is None else repr(value)))
            written += 1
        logger.info(
            "syntheval holdout=%d arm=%s rows=%d metrics=%d",
            fill.holdout,
            fill.arm,
            len(fill.synthetic),
            len(metrics),
        )
    text = out.getvalue()
    logger.info("syntheval rows=%d sha256=%s", written, hashlib.sha256(text.encode()).hexdigest())
    return text


def _log_short(fill: SynthevalFill, line: str) -> None:
    """Log `replica-short holdout=%d arm=%s rows=%d` when the fill's `arm` line names a `short`.

    The line is `compare`'s, the rows the sum of the `short` params; the fill is scored as any.

    Raises:
        IdentityError: As `bko.evaluate.shared.logged_params`.
    """
    short = logged_params(line).get("short")
    if isinstance(short, Mapping) and short:
        missing = sum(cast("Mapping[str, int]", short).values())
        logger.info("replica-short holdout=%d arm=%s rows=%d", fill.holdout, fill.arm, missing)


def _scored(
    fill: SynthevalFill, runner: Runner, argv: Callable[..., list[str]]
) -> dict[str, float | None]:
    """`fill`'s metrics: its inputs exported to a temporary directory, the script run on them.

    `argv` is `score_argv` with every keyword but `inputs` and `seed` bound; the seed is the
    fill's holdout seed. Each input file logs its `syntheval input` line.

    Raises:
        EvaluationError: As `_metrics`.
    """
    with tempfile.TemporaryDirectory(prefix="syntheval-") as name:
        inputs = Path(name)
        for file, sha in export(fill, inputs).items():
            logger.info(
                "syntheval input holdout=%d arm=%s file=%s sha256=%s",
                fill.holdout,
                fill.arm,
                file,
                sha,
            )
        done = runner(argv(inputs=inputs, seed=fill.holdout))
        return _metrics(done, inputs / OUT, where=f"{fill.arm}/{fill.holdout}")


def _metrics(
    done: subprocess.CompletedProcess[str], path: Path, *, where: str
) -> dict[str, float | None]:
    """The metrics the script wrote to `path`, once it exited 0; a non-finite number as None.

    Raises:
        EvaluationError: If it exited non-zero, or `path` holds no JSON object of metric name to
            number or null.
    """
    if done.returncode != 0:
        msg = f"the SynthEval script exited {done.returncode} on {where}"
        msg += f"\nstdout:\n{_tail(done.stdout)}\nstderr:\n{_tail(done.stderr)}"
        raise EvaluationError(msg)
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"the SynthEval script wrote no readable metrics on {where}: {exc}"
        raise EvaluationError(msg) from exc
    if not isinstance(found, dict) or not all(_value(value) for value in found.values()):
        msg = f"the SynthEval script's metrics on {where} are not names to numbers: {found!r}"
        raise EvaluationError(msg)
    return {
        str(name): None if value is None or not math.isfinite(value) else float(value)
        for name, value in found.items()
    }


def _value(value: object) -> bool:
    """Whether `value` is a number or null (a JSON `true` is not a number)."""
    return value is None or (isinstance(value, int | float) and not isinstance(value, bool))


def _tail(text: str | None) -> str:
    """The last `TAIL` lines of `text`."""
    return "\n".join((text or "").splitlines()[-TAIL:])


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `syntheval` subcommand to `commands`."""
    reading = commands.add_parser(
        "syntheval", help="score each replica fill of a run's store under SynthEval's metrics"
    )
    reading.add_argument("run", nargs="?", default=DEFAULT_RUN, help="the named run; default: e1")
    reading.add_argument("--folder", type=Path, required=True, help="the run's folder under runs/")
    reading.add_argument(
        "--rows", type=Path, required=True, help="the folder's store, which `rows` wrote"
    )
    reading.add_argument(
        "--data-dir", type=Path, default=None, help="the data root; default: the repository's"
    )
    reading.add_argument(
        "--bench-dir",
        type=Path,
        default=None,
        help="the directory of score.py and its .venv; default: this checkout's bench/syntheval",
    )
