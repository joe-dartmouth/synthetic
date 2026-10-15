"""Command line for the baselines: `python -m bko.baselines <method>`.

Each subcommand fetches UCI Heart Disease into the data root if it is absent, splits it, builds one
method's training set from the training split and writes it to stdout as CSV: `record_id`
(negative for an added row), the predictors, `target` and `weight`. The split, the quota, the row
counts, the output's sha256 and the method's parameters go to stderr through logging.
`docs/DEVELOPMENT.md` (Baselines) is the usage home.

Exit codes: 0 done; 1 an unexpected error, with its traceback; 2 the command could not run; 3 the
data or a method's output contradicts its contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

import pandas as pd

from bko.baselines.errors import BaselineError
from bko.baselines.methods import METHODS, Runner, augment
from bko.baselines.records import FRAME_COLUMNS, Baseline
from bko.baselines.request import BASELINE_SEED, SEED_LIMIT, Method, Request, Strata, plan
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.datasets.split import split
from bko.datasets.uci import load_uci


logger = logging.getLogger("bko.baselines")

QUIET_LOGGERS: Final = ("sdv", "SingleTableSynthesizer")
"""SDV's loggers, held at WARNING so their INFO records stay off stderr."""


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        methods: Each method's runner; the command offers exactly these.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    methods: Mapping[Method, Runner] = METHODS


def main(
    argv: Sequence[str] | None = None,
    *,
    context: Context | None = None,
    out: TextIO | None = None,
) -> int:
    """Run one method and return the exit code.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when omitted.
        context: The outside world; the real one when omitted.
        out: Where the training set goes; stdout when omitted.

    Returns:
        0 done, 2 the command could not run, 3 the data or the output contradicts its contract.
        Any other exception propagates, so the process exits 1 with its traceback.
    """
    context = context or Context()
    args = _parser(context.methods).parse_args(argv)
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        _run(args, data_dir, context, out or sys.stdout)
    except (BaselineError, DatasetError) as exc:
        # The message is the diagnosis; a traceback would read as a crash.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    return 0


def configure_logging() -> None:
    """Send INFO and higher to stderr, keeping stdout for the CSV, and quiet SDV's loggers."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def render(train: pd.DataFrame, baseline: Baseline) -> str:
    """The training set a method hands a classifier, as CSV: the real rows, then the added ones."""
    real = train.loc[:, list(FRAME_COLUMNS)].assign(weight=baseline.weights)
    added = baseline.added.assign(weight=1.0)
    return pd.concat([real, added]).to_csv(lineterminator="\n")


def _run(args: argparse.Namespace, data_dir: Path, context: Context, out: TextIO) -> None:
    method = Method(args.method)
    ensure(context.uci, data_dir, open_url=context.open_url)
    parts = split(load_uci(data_dir, source=context.uci))
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        parts.seed,
        len(parts.train),
        len(parts.test),
        parts.fingerprint,
    )
    train = parts.train
    request = (
        plan(train, args.rows, strata=Strata(args.strata), seed=args.seed)
        if method.adds_rows
        else Request()
    )
    for stratum, rows in request.quota.items():
        logger.info("quota stratum=%s rows=%d", "/".join(map(str, stratum)), rows)
    baseline = augment(train, method, request, methods=context.methods)
    text = render(train, baseline)
    request_fields = f" seed={request.seed} strata={request.strata}" if method.adds_rows else ""
    logger.info(
        "method=%s%s real=%d synthetic=%d total=%d sha256=%s",
        method,
        request_fields,
        len(train),
        len(baseline.added),
        len(train) + len(baseline.added),
        hashlib.sha256(text.encode()).hexdigest(),
    )
    logger.info("params %s", json.dumps(baseline.params, sort_keys=True, default=str))
    out.write(text)


def _rows(text: str) -> int:
    rows = int(text)
    if rows < 1:
        msg = f"{rows} is not a positive row count"
        raise argparse.ArgumentTypeError(msg)
    return rows


def _seed(text: str) -> int:
    seed = int(text)
    if not 0 <= seed < SEED_LIMIT:
        msg = f"seed {seed} is outside [0, {SEED_LIMIT})"
        raise argparse.ArgumentTypeError(msg)
    return seed


def _parser(methods: Mapping[Method, Runner]) -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    adding = argparse.ArgumentParser(add_help=False)
    adding.add_argument("--rows", type=_rows, required=True, help="rows to add, water-filled")
    adding.add_argument(
        "--strata",
        choices=[strata.value for strata in Strata],
        default=Strata.CELL.value,
        help="what the rows are spread over; default: cell",
    )
    adding.add_argument("--seed", type=_seed, default=BASELINE_SEED, help="default: %(default)s")
    parser = argparse.ArgumentParser(prog="python -m bko.baselines", description=__doc__)
    commands = parser.add_subparsers(dest="method", required=True)
    offered = [method for method in Method if method in methods]
    for method in offered:
        parents = [common, adding] if method.adds_rows else [common]
        commands.add_parser(method.value, parents=parents, help=f"the {method} training set")
    return parser


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
