"""`python -m bko.evaluate contradictions`: count each data contradiction in every frame's cells.

`run` loads UCI Heart Disease, Synthea and Diabetes 130 through `bko.evaluate.data.load`, takes
each frame's training split at holdout seed `SEED` with no removal, loads the pinned releases its
field rows name, and `census` writes every finding `bko.gaps.contradictions.contradictions` makes
as CSV. ADR `research/data-contradiction-handling` names the kinds and the basis. It makes no API
call. The frames live in `bko.evaluate`, so the command does too
(`cleanup:evaluate/frames-below-evaluate`).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
from bko.gaps.contradictions import KINDS, Finding, contradictions
from bko.ontology.releases import load_releases
from bko.ontology.schema import FieldMapping
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


def framed(dataset: Dataset, data_dir: Path, context: Context) -> Framed:
    """`dataset`'s training split at holdout seed `SEED`, its field rows and their releases.

    Raises:
        EvaluationError: If a row's concept prefix has no pinned release.
        DatasetError: As `load` and `bko.ontology.releases.load_releases`.
    """
    data = load(dataset, data_dir, context)
    train = split(data.records, seed=SEED, edges=data.frame.edges).train
    prefixes = sorted({row.concept.partition(":")[0] for row in data.field_rows})
    unpinned = [prefix for prefix in prefixes if prefix not in context.releases]
    if unpinned:
        msg = f"no pinned release for the prefix {', '.join(unpinned)}"
        raise EvaluationError(msg)
    sources = {prefix: context.releases[prefix] for prefix in prefixes}
    ontologies = load_releases(sources, data_dir, open_url=context.open_url)
    return Framed(data.frame.name, train, data.field_rows, ontologies, data.frame)


def run(data_dir: Path, context: Context) -> str:
    """The census of every dataset's frame, in `Dataset` order, as CSV."""
    return census([framed(dataset, data_dir, context) for dataset in Dataset])


def census(frames: Sequence[Framed]) -> str:
    """Every finding of each frame in `frames`, as CSV with `HEADER`, and its log lines.

    Logs, per frame, one `census data` line naming its training records' count and digest, and a
    `census kind` line per kind with its count, 0 included; then one `census rows` line with the
    CSV's data rows and its sha256.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(COLUMNS)
    rows = 0
    for each in frames:
        logger.info(
            "census data frame=%s records=%d digest=%s basis=train seed=%d",
            each.name,
            len(each.train),
            digest(each.train),
            SEED,
        )
        found = contradictions(
            each.train, each.rows, each.ontologies, frame=each.frame, name=each.name
        )
        for kind in KINDS:
            logger.info(
                "census kind=%s frame=%s count=%d",
                kind,
                each.name,
                sum(finding.kind == kind for finding in found),
            )
        writer.writerows(_row(finding, each.name) for finding in found)
        rows += len(found)
    text = out.getvalue()
    logger.info("census rows=%d sha256=%s", rows, hashlib.sha256(text.encode()).hexdigest())
    return text


def _row(finding: Finding, frame: str) -> tuple[str, ...]:
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
    return "" if value is None else str(value)


def _share(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"
