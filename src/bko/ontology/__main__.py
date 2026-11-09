"""Command line for the ontology stage: `python -m bko.ontology check`, `reproduce` and `elk`.

`check` fetches the pinned MONDO and HPO releases into the data root if they are absent, verifies
and parses them, and writes one CSV row per checked row of `bko.ontology.mapping` to stdout: `ok`,
or how the release disagrees. `reproduce` does the same for the pinned MONDO and DO releases,
rebuilds Santos et al.'s fusion example with `bko.ontology.santos`, and writes one CSV row per
quantity of the example. `elk` fetches the pinned OWL releases and ROBOT, classifies each release
with ELK (`bko.ontology.elk`), which needs a JDK, and writes one CSV row per ancestor pair,
within the mapped concepts and their ancestors, that the classification or the loaded OBO release
holds and the other lacks; it finds Java before it fetches anything. The parser's version, each
release's identity, a warning per disagreeing row, and `elk`'s tool, heap, timeout, timings,
classified files and per-release counts go to stderr through logging.
`docs/DEVELOPMENT.md` (Ontologies, Fusion reproduction) is the usage home.

Exit codes: 0 every `check` row agrees, or `reproduce` or `elk` wrote its report, whatever its
statuses; 1 an unexpected error, with its traceback, including a fusion the converter should never
have built; 2 the command could not run, including ROBOT failing or timing out; 3 a release
contradicts its pin, is not OBO the loader reads, or is OWL that names an import (stdout is
empty); 4 a `check` row disagrees with its release (the report is written first).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from bko.core.logs import configure
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import ROBOT_JAR, Source
from bko.datasets.synthea_run import default_java
from bko.ontology import elk
from bko.ontology.check import check
from bko.ontology.mapping import CHECKED_ROWS, MAPPED, RELEASES
from bko.ontology.releases import load_releases
from bko.ontology.santos import PAPER_RELEASES, reproduce
from bko.ontology.schema import CheckedRow
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.ontology")


@dataclass(frozen=True, slots=True)
class Context:
    """What the command reaches outside the process; tests replace each field.

    Attributes:
        open_url: The HTTPS opener.
        releases: The pinned release for each concept prefix.
        rows: The rows to check: concept rows and excluded codes.
        paper_releases: The pinned release for each prefix `reproduce` reads.
        owl_releases: The pinned OWL release `elk` classifies for each prefix.
        robot: The pinned ROBOT jar `elk` runs.
        resolve_java: Returns the Java `elk` runs ROBOT with and its version line.
        robot_runner: Runs ROBOT's argv for `elk`.
        concepts: The mapped concepts whose hierarchy `elk` compares.
    """

    open_url: UrlOpener = open_https
    releases: Mapping[str, Source] = RELEASES
    rows: tuple[CheckedRow, ...] = CHECKED_ROWS
    paper_releases: Mapping[str, Source] = PAPER_RELEASES
    owl_releases: Mapping[str, Source] = elk.OWL_RELEASES
    robot: Source = ROBOT_JAR
    resolve_java: Callable[[], tuple[Path, str]] = default_java
    robot_runner: elk.Runner = elk.run_robot
    concepts: frozenset[str] = MAPPED


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
        if args.command == "elk":
            return _elk(data_dir, context, out)
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
    configure()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bko.ontology", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, purpose in (
        ("check", "check the mapping table against the releases"),
        ("reproduce", "rebuild Santos et al.'s MONDO/DO fusion example and compare it"),
        ("elk", "classify the OWL releases with ELK and compare their hierarchies (needs a JDK)"),
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
                "row %s %s disagrees: %s",
                verdict.table,
                verdict.concept or verdict.subject,
                verdict.problem,
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


def _elk(data_dir: Path, context: Context, out: TextIO) -> int:
    java, version = context.resolve_java()
    loaded = load_releases(context.releases, data_dir, open_url=context.open_url)
    owls = {
        prefix: ensure(source, data_dir, open_url=context.open_url)
        for prefix, source in sorted(context.owl_releases.items())
    }
    for path in owls.values():
        elk.refuse_imports(path)
    jar = ensure(context.robot, data_dir, open_url=context.open_url)
    logger.info(
        "elk tool robot_url=%s robot_sha256=%s java=%s xmx_gb=%d timeout_s=%d",
        context.robot.url,
        context.robot.sha256,
        version,
        elk.ELK_MEMORY_GB,
        elk.ELK_TIMEOUT_SECONDS,
    )
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(elk.COLUMNS)
    differing = 0
    for prefix, owl in owls.items():
        release = loaded[prefix].name
        comparison = _classify_and_compare(
            prefix, owl, loaded[prefix], data_dir, context, java=java, jar=jar
        )
        writer.writerows(elk.rows(release, comparison))
        differing += comparison.differing
    report = text.getvalue()
    digest = hashlib.sha256(report.encode()).hexdigest()
    logger.info("elk differing ancestor_pairs=%d sha256=%s", differing, digest)
    out.write(report)
    return 0


def _classify_and_compare(
    prefix: str,
    owl: Path,
    ontology: Ontology,
    data_dir: Path,
    context: Context,
    *,
    java: Path,
    jar: Path,
) -> elk.Comparison:
    """Classify one prefix's OWL release, compare it with `ontology`, and log both steps."""
    release = ontology.name
    source = context.owl_releases[prefix]
    classified = data_dir / "elk" / f"{source.name}-{source.sha256[:16]}.obo"
    logger.info(
        "elk classify release=%s owl_sha256=%s xmx_gb=%d timeout_s=%d",
        release,
        source.sha256,
        elk.ELK_MEMORY_GB,
        elk.ELK_TIMEOUT_SECONDS,
    )
    seconds = elk.classify(owl, classified, java=java, jar=jar, runner=context.robot_runner)
    body = classified.read_bytes()
    logger.info(
        "elk classified release=%s seconds=%.1f path=%s bytes=%d sha256=%s",
        release,
        seconds,
        classified,
        len(body),
        hashlib.sha256(body).hexdigest(),
    )
    mine, theirs = elk.loaded_hierarchy(ontology), elk.read_hierarchy(classified)
    comparison = elk.compare_hierarchies(
        mine,
        theirs,
        {concept for concept in context.concepts if concept.startswith(f"{prefix}:")},
    )
    logger.info(
        "elk compare release=%s loaded_terms=%d elk_classes=%d concepts=%d "
        "absent_loaded=%d absent_elk=%d only_loaded=%d only_elk=%d "
        "minimal_loaded=%d minimal_elk=%d",
        release,
        len(mine),
        len(theirs),
        comparison.concepts,
        comparison.absent_loaded,
        comparison.absent_elk,
        len(comparison.only_loaded),
        len(comparison.only_elk),
        comparison.minimal_loaded,
        comparison.minimal_elk,
    )
    return comparison


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
