"""Command line for the ontology stage: `python -m bko.ontology check`.

`check` fetches the pinned MONDO and HPO releases into the data root if they are absent, verifies
and parses them, and writes one CSV row per concept row of `bko.ontology.mapping` to stdout: `ok`,
or how the release disagrees. Each release's identity goes to stderr through logging.
`docs/DEVELOPMENT.md` (Ontologies) is the usage home.

Exit codes: 0 every row agrees; 1 an unexpected error, with its traceback; 2 the command could not
run; 3 a release contradicts its pin or is not OBO the loader reads, or a row disagrees with its
release (the report is written first).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import Source
from bko.ontology.check import check
from bko.ontology.mapping import CONCEPT_ROWS, RELEASES
from bko.ontology.obo import load_ontology
from bko.ontology.schema import ConceptRow
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.ontology")


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        releases: The pinned release for each concept prefix.
        rows: The concept rows to check.
    """

    open_url: UrlOpener = open_https
    releases: Mapping[str, Source] = RELEASES
    rows: tuple[ConceptRow, ...] = CONCEPT_ROWS


def main(
    argv: Sequence[str] | None = None,
    *,
    context: Context | None = None,
    out: TextIO | None = None,
) -> int:
    """Run the subcommand and return its exit code.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when omitted.
        context: The outside world; the real one when omitted.
        out: Where the report goes; stdout when omitted.

    Returns:
        0 every row agrees, 2 the command could not run, 3 a contract failure or a disagreeing row.
    """
    args = _parser().parse_args(argv)
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        context = Context() if context is None else context
        return _check(data_dir, context, sys.stdout if out is None else out)
    except DatasetError as exc:
        # The message is the diagnosis; a traceback would read as a crash in the loader.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2


def configure_logging() -> None:
    """Send INFO and higher log records to stderr, keeping stdout for the report."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bko.ontology", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("check", help="check the mapping table against the releases")
    command.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the data root; default: data/ at the repository root",
    )
    return parser


def _check(data_dir: Path, context: Context, out: TextIO) -> int:
    ontologies: dict[str, Ontology] = {}
    for prefix, source in sorted(context.releases.items()):
        ensure(source, data_dir, open_url=context.open_url)
        ontology = load_ontology(source, data_dir)
        logger.info(
            "release %s data_version=%s snapshot=%s terms=%d obsolete=%d",
            source.name,
            ontology.version,
            source.sha256[:16],
            len(ontology.terms),
            sum(term.obsolete for term in ontology.terms.values()),
        )
        ontologies[prefix] = ontology
    verdicts = check(context.rows, ontologies)
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("table", "subject", "concept", "label", "status"))
    writer.writerows((v.table, v.subject, v.concept, v.label, v.problem or "ok") for v in verdicts)
    disagree = sum(v.problem is not None for v in verdicts)
    logger.info("checked %d rows: %d disagree", len(verdicts), disagree)
    return 3 if disagree else 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
