"""Command line for the ontology stage: `python -m bko.ontology check` and `reproduce`.

`check` fetches the pinned MONDO and HPO releases into the data root if they are absent, verifies
and parses them, and writes one CSV row per concept row of `bko.ontology.mapping` to stdout: `ok`,
or how the release disagrees. `reproduce` does the same for the pinned MONDO and DO releases,
rebuilds Santos et al.'s fusion example with `bko.ontology.santos`, and writes one CSV row per
quantity of the example. The parser's version, each release's identity and a warning per
disagreeing row go to stderr through logging. `docs/DEVELOPMENT.md` (Ontologies, Fusion
reproduction) is the usage home.

Exit codes: 0 every `check` row agrees, or `reproduce` wrote its report, whatever its statuses; 1
an unexpected error, with its traceback, including a fusion the converter should never have
built; 2 the command could not run; 3 a release contradicts its pin or is not OBO the loader
reads (stdout is empty); 4 a `check` row disagrees with its release (the report is written first).
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
from bko.datasets.fetch import UrlOpener, default_data_dir, open_https
from bko.datasets.sources import Source
from bko.ontology.check import check
from bko.ontology.mapping import CONCEPT_ROWS, RELEASES
from bko.ontology.releases import load_releases
from bko.ontology.santos import PAPER_RELEASES, reproduce
from bko.ontology.schema import ConceptRow


logger = logging.getLogger("bko.ontology")


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        releases: The pinned release for each concept prefix.
        rows: The concept rows to check.
        paper_releases: The pinned release for each prefix `reproduce` reads.
    """

    open_url: UrlOpener = open_https
    releases: Mapping[str, Source] = RELEASES
    rows: tuple[ConceptRow, ...] = CONCEPT_ROWS
    paper_releases: Mapping[str, Source] = PAPER_RELEASES


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
        0 done (see the module docstring), 2 the command could not run, 3 a contract failure, 4 a
        disagreeing check row.
    """
    args = _parser().parse_args(argv)
    try:
        data_dir = args.data_dir.resolve() if args.data_dir is not None else default_data_dir()
        context = Context() if context is None else context
        out = sys.stdout if out is None else out
        if args.command == "check":
            return _check(data_dir, context, out)
        return _reproduce(data_dir, context, out)
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
    for name, purpose in (
        ("check", "check the mapping table against the releases"),
        ("reproduce", "rebuild Santos et al.'s MONDO/DO fusion example and compare it"),
    ):
        command = commands.add_parser(name, help=purpose)
        command.add_argument(
            "--data-dir",
            type=Path,
            default=None,
            help="the data root; default: data/ at the repository root",
        )
    return parser


def _check(data_dir: Path, context: Context, out: TextIO) -> int:
    verdicts = check(
        context.rows, load_releases(context.releases, data_dir, open_url=context.open_url)
    )
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("table", "subject", "concept", "label", "status"))
    writer.writerows((v.table, v.subject, v.concept, v.label, v.problem or "ok") for v in verdicts)
    for verdict in verdicts:
        if verdict.problem is not None:
            logger.warning(
                "row %s %s disagrees: %s", verdict.table, verdict.concept, verdict.problem
            )
    disagree = sum(v.problem is not None for v in verdicts)
    logger.info("checked %d rows: %d disagree", len(verdicts), disagree)
    return 4 if disagree else 0


def _reproduce(data_dir: Path, context: Context, out: TextIO) -> int:
    ontologies = load_releases(context.paper_releases, data_dir, open_url=context.open_url)
    findings = reproduce(ontologies["MONDO"], ontologies["DOID"])
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("kind", "item", "paper", "measured", "status"))
    writer.writerows((f.kind, f.item, f.paper, f.measured, f.status) for f in findings)
    differ = [f for f in findings if f.status == "differs"]
    for finding in differ:
        logger.warning(
            "finding %s %s differs: paper %r, measured %r",
            finding.kind,
            finding.item,
            finding.paper,
            finding.measured,
        )
    logger.info("reproduced %d findings: %d differ", len(findings), len(differ))
    return 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
