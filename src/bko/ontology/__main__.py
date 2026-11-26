"""The ontology stage's command line: `check`, `reproduce`, `elk`, `codes` and `occupations`.

`check` fetches the pinned release of each prefix its rows name into the data root if it is absent,
from `RELEASES` or, for a prefix those lack, a catalog entry's release map (OccO for ACS), converts
an OWL-only one to OBO, verifies and parses them, and writes one CSV row per checked row of
`bko.ontology.mapping` to stdout: `ok`, or how the release disagrees. `reproduce` does the same for
the pinned MONDO and DO releases, rebuilds Santos et al.'s fusion example with
`bko.ontology.santos`, and writes one CSV row per quantity of the example. `elk` fetches the pinned
OWL releases and ROBOT, classifies each release with ELK (`bko.ontology.elk`), which needs a JDK,
and writes one CSV row per ancestor pair, within the mapped concepts and their ancestors, that the
classification or the loaded OBO release holds and the other lacks; it finds Java before it fetches
anything. The parser's version, each release's identity, a warning per disagreeing row, and `elk`'s
tool, heap, timeout, timings, classified files and per-release counts go to stderr through logging.
`codes` fetches the pinned MONDO release and Diabetes 130 archive if they are absent, maps every
diagnosis code the `diabetes-codes` cohort holds with `bko.ontology.codes.generate`, writes the
module text of `bko.ontology.icd9_codes` to stdout, and logs a `codes unmapped` line per reason and
a closing `codes rows=` line; stdout stays empty until the text is complete. `occupations --part
low` or `--part high` fetches the pinned OccO release and ACS archive if they are absent, maps each
kept `OCCP` code through its `SOCP` code with `bko.ontology.occupations.generate`, writes the module
text of that part, `bko.ontology.occupation_codes_low` or `_high`, to stdout, and logs an
`occupations unmapped parts=all` line per reason, counted over both parts, and a closing
`occupations part=` line with the codes read and the classes holding a SOC id, stdout likewise
empty until then. `docs/DEVELOPMENT.md` (Ontologies, Fusion reproduction) is the usage home.

Exit codes: 0 every `check` row agrees, or `reproduce`, `elk`, `codes` or `occupations` wrote its
report,
whatever its statuses; 1 an unexpected error, with its traceback, including a fusion the
converter should never have built; 2 the command could not run, including ROBOT failing or timing
out; 3 a release or archive contradicts its pin, is not OBO the loader reads, or is OWL that names
an import (stdout is empty); 4 a `check` row disagrees with its release (the report is written
first).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, TextIO

from bko.core.logs import configure
from bko.datasets.acs import occupation_pairs, read_kept
from bko.datasets.diabetes import DIAGNOSES, load_diabetes
from bko.datasets.diagnoses import code_text
from bko.datasets.errors import DatasetError
from bko.datasets.fetch import UrlOpener, default_data_dir, ensure, open_https
from bko.datasets.sources import ACS_MA_2024, DIABETES_130, OCCO, ROBOT_JAR, Source
from bko.datasets.synthea_run import default_java
from bko.ontology import elk
from bko.ontology.catalog import CATALOG, Catalogued
from bko.ontology.check import check, prefixes
from bko.ontology.codes import generate, render
from bko.ontology.convert import converted
from bko.ontology.mapping import CHECKED_ROWS, MAPPED, RELEASES
from bko.ontology.occupations import PARTS, Part, part, soc_classes
from bko.ontology.occupations import generate as generate_occupations
from bko.ontology.occupations import render as render_occupations
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
        diabetes: The pinned Diabetes 130 archive whose diagnosis codes `codes` maps.
        catalog: The catalogued datasets, whose release maps `check` reads for a prefix
            `releases` lacks.
        occo: The pinned Occupation Ontology release `occupations` maps into.
        acs: The pinned ACS archive whose occupation codes `occupations` maps.
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
    diabetes: Source = DIABETES_130
    catalog: Mapping[str, Catalogued] = CATALOG
    occo: Source = OCCO
    acs: Source = ACS_MA_2024


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
        if args.command == "occupations":
            return _occupations(data_dir, context, out, which=args.part)
        return _COMMANDS[args.command](data_dir, context, out)
    except DatasetError as exc:
        # The message is the diagnosis; a traceback would read as a crash in the loader.
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return exc.exit_code
    except OSError as exc:
        logger.error("%s", exc)  # ruff: ignore[error-instead-of-exception]
        return 2


def configure_logging() -> None:
    """Send INFO and higher, and every warning, to stderr, keeping stdout for the report."""
    configure()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bko.ontology", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, purpose in (
        ("check", "check the mapping table against the releases"),
        ("reproduce", "rebuild Santos et al.'s MONDO/DO fusion example and compare it"),
        ("elk", "classify the OWL releases with ELK and compare their hierarchies (needs a JDK)"),
        ("codes", "generate bko.ontology.icd9_codes from MONDO and the Diabetes 130 archive"),
        ("occupations", "generate one part of ACS's occupation rows from OccO and the archive"),
    ):
        command = commands.add_parser(name, help=purpose)
        command.add_argument(
            "--data-dir",
            type=Path,
            default=None,
            help="the data root; default: data/ at the repository root",
        )
        if name == "occupations":
            command.add_argument(
                "--part",
                choices=PARTS,
                required=True,
                help="low: SOC major groups below 31; high: 31 and up",
            )
    return parser


def check_releases(context: Context) -> dict[str, Source]:
    """The release of each prefix `context.rows` name, from its releases or else its catalog.

    A prefix `context.releases` pins takes that release; any other takes the one every catalog
    entry that pins it agrees on, and a prefix none pins is left out, so its rows report it.

    Raises:
        ValueError: If two catalog entries pin a prefix `context.releases` lacks to different
            releases.
    """
    chosen: dict[str, Source] = {}
    for prefix in sorted(prefixes(context.rows)):
        if prefix in context.releases:
            chosen[prefix] = context.releases[prefix]
            continue
        pinned = {
            entry.releases[prefix]
            for entry in context.catalog.values()
            if entry.releases is not None and prefix in entry.releases
        }
        if len(pinned) > 1:
            names = ", ".join(sorted(source.name for source in pinned))
            msg = f"catalog entries pin prefix {prefix} to different releases: {names}"
            raise ValueError(msg)
        chosen.update((prefix, source) for source in pinned)
    return chosen


def _check(data_dir: Path, context: Context, out: TextIO) -> int:
    convert = partial(
        converted,
        resolve_java=context.resolve_java,
        robot=context.robot,
        runner=context.robot_runner,
    )
    releases = load_releases(
        check_releases(context), data_dir, open_url=context.open_url, convert=convert
    )
    verdicts = check(context.rows, releases)
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


def _codes(data_dir: Path, context: Context, out: TextIO) -> int:
    releases = {"MONDO": context.releases["MONDO"]}
    mondo = load_releases(releases, data_dir, open_url=context.open_url)["MONDO"]
    source = context.diabetes
    ensure(source, data_dir, open_url=context.open_url)
    records = load_diabetes(data_dir, indicators={}, diagnoses=True, source=source)
    held = {int(number) for name in DIAGNOSES for number in records[name].unique()}
    generated = generate(mondo, map(code_text, sorted(held)))
    text = render(generated.rows)
    for reason, count in generated.unmapped.items():
        logger.info("codes unmapped reason=%s codes=%d", json.dumps(reason), count)
    digest = hashlib.sha256(text.encode()).hexdigest()
    logger.info("codes rows=%d sha256=%s release=%s", len(generated.rows), digest, mondo.version)
    out.write(text)
    return 0


def _occupations(data_dir: Path, context: Context, out: TextIO, *, which: Part) -> int:
    convert = partial(
        converted,
        resolve_java=context.resolve_java,
        robot=context.robot,
        runner=context.robot_runner,
    )
    occo = load_releases(
        {"OCCO": context.occo}, data_dir, open_url=context.open_url, convert=convert
    )["OCCO"]
    ensure(context.acs, data_dir, open_url=context.open_url)
    rows, _ = read_kept(data_dir, context.acs)
    pairs = occupation_pairs(rows)
    generated, unmapped = generate_occupations(pairs, occo)
    text = render_occupations([row for row in generated if part(row) == which], which)
    for reason, count in unmapped.items():
        logger.info("occupations unmapped parts=all reason=%s codes=%d", json.dumps(reason), count)
    logger.info(
        "occupations part=%s rows=%d of=%d codes=%d holders=%d sha256=%s release=%s",
        which,
        sum(part(row) == which for row in generated),
        len(generated),
        len(pairs),
        len(soc_classes(occo)),
        hashlib.sha256(text.encode()).hexdigest(),
        occo.version,
    )
    out.write(text)
    return 0


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


_COMMANDS: Final[Mapping[str, Callable[[Path, Context, TextIO], int]]] = {
    "check": _check,
    "reproduce": _reproduce,
    "elk": _elk,
    "codes": _codes,
}


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
