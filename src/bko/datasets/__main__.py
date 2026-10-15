"""Command line for the evaluation data: `python -m bko.datasets <subcommand>`.

`cells` fetches UCI Heart Disease into the data root if it is absent, verifies it, splits it and
writes the age-sex subgroup-label cell counts of the training split to stdout as CSV, the same
bytes on every run over the same archive. The snapshot identity, the split and the underrepresented
cells go to stderr through logging. `docs/DEVELOPMENT.md` (Data) is the usage home.

Exit codes: 0 done; 1 the data contradicts its pin or schema; 2 the command could not run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from bko.datasets.cells import CELL_COLUMNS, UNDERREPRESENTED_BELOW, cell_counts
from bko.datasets.errors import DatasetError, UnavailableError
from bko.datasets.fetch import UrlOpener, ensure, open_https, read_record
from bko.datasets.sources import UCI_HEART_DISEASE, Source
from bko.datasets.split import split
from bko.datasets.uci import LABEL, load_uci


logger = logging.getLogger("bko.datasets")


@dataclass(frozen=True, slots=True)
class Context:
    """What the commands reach outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        uci: The pinned UCI archive.
    """

    open_url: UrlOpener = open_https
    uci: Source = UCI_HEART_DISEASE


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
        0 done, 1 the data contradicts its pin or schema, 2 the command could not run.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        data_dir = _data_dir(args.data_dir)
        _cells(data_dir, context or Context(), out or sys.stdout)
    except DatasetError as exc:
        # The message is the diagnosis; a traceback would read as a crash in the loader.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2
    return 0


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
    return parser


_THIS_FILE: Final = Path(__file__)


def default_data_dir(anchor: Path = _THIS_FILE) -> Path:
    """Return `data/` at the root of the checkout this module runs from.

    The editable install runs this file from `src/bko/datasets/`, three levels below the root,
    whose `.gitignore` ignores `/data/`. Elsewhere, as in an installed wheel, there is no default.

    Raises:
        UnavailableError: If `anchor` does not sit in a checkout's `src/bko/datasets/`.
    """
    root = anchor.resolve().parents[3]
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "bko").is_dir():
        msg = f"no repository root above {anchor}; pass --data-dir"
        raise UnavailableError(msg)
    return root / "data"


def _data_dir(given: Path | None) -> Path:
    return given.resolve() if given is not None else default_data_dir()


def _cells(data_dir: Path, context: Context, out: TextIO) -> None:
    ensure(context.uci, data_dir, open_url=context.open_url)
    record = read_record(context.uci, data_dir)
    logger.info(
        "snapshot %s sha256=%s retrieved_at=%s",
        context.uci.name,
        context.uci.sha256,
        record["retrieved_at"],
    )
    parts = split(load_uci(data_dir, source=context.uci))
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        parts.seed,
        len(parts.train),
        len(parts.test),
        parts.fingerprint,
    )
    counts = cell_counts(parts.train)
    under = counts.loc[counts["underrepresented"]]
    cells = zip(*(under[column] for column in (*CELL_COLUMNS, "count")), strict=True)
    logger.info(
        "%d of %d cells hold fewer than %d training records: %s",
        len(under),
        len(counts),
        UNDERREPRESENTED_BELOW,
        "; ".join(f"{band} {sex} {LABEL}={label} count={n}" for band, sex, label, n in cells),
    )
    counts.to_csv(out, index=False, lineterminator="\n")


if __name__ == "__main__":
    sys.exit(main())
