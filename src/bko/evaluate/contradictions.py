"""`python -m bko.evaluate contradictions`: count each data contradiction in every frame's cells.

`run` loads UCI Heart Disease, Synthea and Diabetes 130 through `bko.evaluate.data.load`, takes
each frame's training split at holdout seed `SEED` with no removal, loads once the pinned releases
any frame's field rows name, and `census` writes every finding `bko.gaps.contradictions.survey`
makes as CSV. ADR `research/data-contradiction-handling` names the kinds and the basis. It makes
no API call. ADR `research/data-contradiction-handling` records why it runs from `bko.evaluate`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.datasets.split import split
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, digest, load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.provenance import log_versions
from bko.gaps.contradictions import KINDS, Finding, survey
from bko.ontology.releases import load_releases
from bko.ontology.schema import Clause, Codes, FieldMapping
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.evaluate")

SEED: Final = 0
"""The holdout seed whose training split the census reads."""

COLUMNS: Final = (
    "kind",
    "frame",
    "cell",
    "concept",
    "parent",
    "records",
    "shown",
    "with_parent",
    "without_parent",
    "child_share",
    "parent_share",
    "field",
    "codes",
)
HEADER: Final = ",".join(COLUMNS)


@dataclass(frozen=True, slots=True, eq=False)
class Framed:
    """One frame's training split and the knowledge the census reads it against.

    Attributes:
        name: The frame's name, which the CSV's `frame` column and the log lines carry.
        train: The training records.
        rows: The field mappings.
        ontologies: The loaded releases the rows name, by concept prefix.
        frame: The records' frame.
    """

    name: str
    train: pd.DataFrame
    rows: tuple[FieldMapping, ...]
    ontologies: Mapping[str, Ontology]
    frame: FrameSpec


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the `contradictions` subcommand to `commands`."""
    counting = commands.add_parser(
        "contradictions", help="count each data contradiction in every frame's training cells"
    )
    counting.add_argument(
        "--data-dir", type=Path, default=None, help="the data root; default: the repository's"
    )


def frames(datasets: Sequence[Dataset], data_dir: Path, context: Context) -> list[Framed]:
    """Each dataset's training split at holdout seed `SEED`, its field rows and their releases.

    Every release any frame's rows name is loaded once, and each frame gets those its rows name.

    Raises:
        EvaluationError: If a row's concept prefix has no pinned release, naming the frame.
        DatasetError: As `load` and `bko.ontology.releases.load_releases`.
    """
    loaded = [load(dataset, data_dir, context) for dataset in datasets]
    named: list[list[str]] = []
    for data in loaded:
        prefixes = sorted({row.concept.partition(":")[0] for row in data.field_rows})
        unpinned = [prefix for prefix in prefixes if prefix not in context.releases]
        if unpinned:
            msg = f"the {data.frame.name} frame's rows name no pinned release for the prefix "
            raise EvaluationError(msg + ", ".join(unpinned))
        named.append(prefixes)
    every = sorted({prefix for prefixes in named for prefix in prefixes})
    sources = {prefix: context.releases[prefix] for prefix in every}
    ontologies = load_releases(sources, data_dir, open_url=context.open_url)
    return [
        Framed(
            data.frame.name,
            split(data.records, seed=SEED, edges=data.frame.edges).train,
            data.field_rows,
            {prefix: ontologies[prefix] for prefix in prefixes},
            data.frame,
        )
        for data, prefixes in zip(loaded, named, strict=True)
    ]


def run(data_dir: Path, context: Context) -> str:
    """The census of every dataset's frame, in `Dataset` order, as CSV.

    Logs the `versions` line (`bko.evaluate.provenance.log_versions`) and the data root first.
    """
    log_versions()
    logger.info("census data_dir=%s", data_dir)
    return census(frames(tuple(Dataset), data_dir, context))


def census(framed: Sequence[Framed]) -> str:
    """Every finding of each frame in `framed`, as CSV with `HEADER`, and its log lines.

    Logs, per frame and in frame order: one `census data` line naming its training records'
    count and digest; one `census rows` line naming its field rows' count, their digest
    (`rows_digest`) and its `is_a` pair count; one `census cells` line naming the grid's cell
    count, how many hold a record, and the concepts that follow the label, which `all-show`
    leaves out (`-` for none); one `census pair` line per `is_a` pair with its counts over the
    whole split (`bko.gaps.contradictions.PairTotal`); and a `census kind` line per kind with its
    count, 0 included. Then one `census rows` line with the CSV's data rows and its sha256.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(COLUMNS)
    rows = 0
    for each in framed:
        logger.info(
            "census data frame=%s records=%d digest=%s basis=train seed=%d",
            each.name,
            len(each.train),
            digest(each.train),
            SEED,
        )
        found = survey(each.train, each.rows, each.ontologies, frame=each.frame, name=each.name)
        logger.info(
            "census rows frame=%s rows=%d digest=%s pairs=%d",
            each.name,
            len(each.rows),
            rows_digest(each.rows),
            len(found.pairs),
        )
        logger.info(
            "census cells frame=%s cells=%d examined=%d following=%s",
            each.name,
            found.cells,
            found.examined,
            ",".join(sorted(found.following)) or "-",
        )
        for pair in found.pairs:
            logger.info(
                "census pair frame=%s concept=%s parent=%s shown=%d without=%d cells=%d",
                each.name,
                pair.concept,
                pair.parent,
                pair.shown,
                pair.without,
                pair.cells,
            )
        for kind in KINDS:
            logger.info(
                "census kind=%s frame=%s count=%d",
                kind,
                each.name,
                sum(finding.kind == kind for finding in found.findings),
            )
        writer.writerows(_row(finding, each.name) for finding in found.findings)
        rows += len(found.findings)
    text = out.getvalue()
    logger.info("census rows=%d sha256=%s", rows, hashlib.sha256(text.encode()).hexdigest())
    return text


def rows_digest(rows: Sequence[FieldMapping]) -> str:
    """The sha256 of `rows` as JSON, in their order, each clause's codes sorted."""
    document = [
        {
            "when": [[type(c).__name__, c.field, _read(c)] for c in row.when],
            "concept": row.concept,
            "label": row.label,
            "source": row.source,
            "limits": row.limits,
        }
        for row in rows
    ]
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _read(clause: Clause) -> list[int] | float:
    """What `clause` reads: its codes in ascending order, or its threshold."""
    return sorted(clause.codes) if isinstance(clause, Codes) else clause.value


def _row(finding: Finding, frame: str) -> tuple[str, ...]:
    """`finding` in `frame` as one CSV row, in `COLUMNS` order, an absent value empty."""
    return (
        finding.kind,
        frame,
        finding.cell,
        finding.concept,
        finding.parent,
        _whole(finding.records),
        _whole(finding.shown),
        _whole(finding.with_parent),
        _whole(finding.without_parent),
        _share(finding.child_share),
        _share(finding.parent_share),
        finding.field,
        " ".join(str(code) for code in finding.codes),
    )


def _whole(value: int | None) -> str:
    """A count as text, or empty where the kind has none."""
    return "" if value is None else str(value)


def _share(value: float | None) -> str:
    """A share as text to six decimals, or empty where the kind has none."""
    return "" if value is None else f"{value:.6f}"
