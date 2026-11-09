"""Command line for the evaluation data: `python -m bko.datasets <subcommand>`.

`cells` fetches UCI Heart Disease into the data root if it is absent, verifies it, splits it and
writes the age-sex subgroup-label cell counts of the training split to stdout as CSV, the same
bytes on every run over the same archive. The snapshot identity, the split and the underrepresented
cells go to stderr through logging. `docs/DEVELOPMENT.md` (Data) is the usage home.

`synthea` generates the pinned Synthea population into the data root if it is absent, removes the
default ablation's patients, and writes each cell's count before and after to stdout as CSV.

Exit codes: 0 done; 1 an unexpected error, with its traceback; 2 the command could not run; 3 the
data contradicts its pin or schema.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from bko.core.logs import configure
from bko.datasets.ablation import DEFAULT_ABLATION, AblationSpec, ablate
from bko.datasets.cells import CELL_COLUMNS, LABEL, UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, open_https
from bko.datasets.sources import SYNTHEA_JAR, UCI_HEART_DISEASE, Source
from bko.datasets.split import uci_split
from bko.datasets.synthea import load_population
from bko.datasets.synthea_run import (
    DEFAULT_RUN,
    Runner,
    SyntheaRun,
    default_java,
    ensure_population,
    run_java,
)
from bko.datasets.uci import UCI_FRAME


logger = logging.getLogger("bko.datasets")


@dataclass(frozen=True, slots=True)
class Context:
    """What the commands reach outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
        synthea_jar: The pinned Synthea jar.
        resolve_java: Returns the Java to run Synthea with and its version line.
        runner: Runs Synthea's argv in a directory and returns its exit status.
        run: The Synthea population to generate.
        ablation: The cases `synthea` removes.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE
    synthea_jar: Source = SYNTHEA_JAR
    resolve_java: Callable[[], tuple[Path, str]] = default_java
    runner: Runner = run_java
    run: SyntheaRun = DEFAULT_RUN
    ablation: AblationSpec = DEFAULT_ABLATION


def main(
    argv: Sequence[str] | None = None,
    *,
    context: Context | None = None,
    out: TextIO | None = None,
) -> int:
    """Run one subcommand and return its exit code.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when omitted.
        context: The outside world; the real one when omitted.
        out: Where the report goes; stdout when omitted.

    Returns:
        0 done, 2 the command could not run, 3 the data contradicts its pin or schema.
    """
    args = _parser().parse_args(argv)
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        command = {"cells": _cells, "synthea": _synthea}[args.command]
        command(data_dir, context or Context(), out or sys.stdout)
    except DatasetError as exc:
        # The message is the diagnosis; a traceback would read as a crash in the loader.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    return 0


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr, keeping stdout for the report."""
    configure()


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    parser = argparse.ArgumentParser(prog="python -m bko.datasets", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "cells", parents=[common], help="write the training split's cell counts to stdout as CSV"
    )
    commands.add_parser(
        "synthea",
        parents=[common],
        help="ablate the default Synthea cell; write counts before and after",
    )
    return parser


def _cells(data_dir: Path, context: Context, out: TextIO) -> None:
    parts = uci_split(data_dir, source=context.uci, open_url=context.open_url)
    counts = cell_counts(parts.train, edges=UCI_FRAME.edges)
    under = counts.loc[counts["underrepresented"]]
    cells = zip(*(under[column] for column in (*CELL_COLUMNS, "count")), strict=True)
    logger.info(
        "%d of %d cells hold fewer than %d training records: %s",
        len(under),
        len(counts),
        UNDERREPRESENTED_BELOW,
        "; ".join(f"{band} {sex} {LABEL}={label} count={n}" for band, sex, label, n in cells)
        or "none",
    )
    counts.to_csv(out, index=False, lineterminator="\n")


def _synthea(data_dir: Path, context: Context, out: TextIO) -> None:
    directory = ensure_population(
        context.run,
        data_dir,
        jar_source=context.synthea_jar,
        open_url=context.open_url,
        resolve_java=context.resolve_java,
        runner=context.runner,
    )
    population = load_population(directory, snapshot_date=context.run.snapshot_date)
    result = ablate(population, context.ablation)
    logger.info(
        "population %s; removed %d patients with %s, age band %s, sex %s",
        directory.name,
        len(result.removed),
        sorted(context.ablation.codes),
        context.ablation.age_band,
        context.ablation.sex,
    )
    report = result.before[list(CELL_COLUMNS)].assign(
        before=result.before["count"], after=result.after["count"]
    )
    report.to_csv(out, index=False, lineterminator="\n")


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
