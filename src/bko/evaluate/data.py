"""The data a named run scores: one `Data` value, chosen by `NamedRun.dataset`, every stage reads.

`score` loads it once (`load`) and hands it to the harness, the BKO arms, the rehearsal and the
candidates file, so each reads the same records, frame, field mappings and digest, and the run's
logs name the frame. ADR `research/uci-evaluation-frame` records why the run, not a flag, chooses
it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import pandas as pd

from bko.datasets.fetch import UrlOpener, ensure
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import Source
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.ontology.schema import FieldMapping


class Dataset(StrEnum):
    """The datasets a named run can score."""

    UCI = "uci"


@dataclass(frozen=True, slots=True, eq=False)
class Data:
    """A run's records and what every stage reads about them.

    Attributes:
        records: The complete records, indexed by `record_id`, in `frame`'s columns.
        frame: Their frame.
        field_rows: The field mappings the BKO arms observe cells by.
        name: The dataset's source name; each cell's ontology fragment takes it.
        digest: The sha256 of `records` as `digest` renders them, which the candidates file's
            header names.
    """

    records: pd.DataFrame
    frame: FrameSpec
    field_rows: tuple[FieldMapping, ...]
    name: str
    digest: str


class Reach(Protocol):
    """What `load` reads of a command's context: the evaluate and app contexts both have it."""

    @property
    def open_url(self) -> UrlOpener:
        """The HTTPS opener."""
        ...

    @property
    def uci(self) -> Source:
        """The pinned UCI archive."""
        ...

    @property
    def field_rows(self) -> tuple[FieldMapping, ...]:
        """The field mappings."""
        ...


def digest(records: pd.DataFrame) -> str:
    """The sha256 of `records` as CSV with newline line ends (`to_csv(lineterminator=...)`)."""
    return hashlib.sha256(records.to_csv(lineterminator="\n").encode()).hexdigest()


def load(dataset: Dataset, data_dir: Path, context: Reach) -> Data:
    """Load `dataset` from `data_dir`, fetching UCI Heart Disease if the data root lacks it.

    Raises:
        DatasetError: As `bko.datasets.fetch.ensure` and `bko.datasets.uci.load_uci`.
        OSError: If the data root cannot be written.
    """
    match dataset:
        case Dataset.UCI:
            ensure(context.uci, data_dir, open_url=context.open_url)
            records = load_uci(data_dir, source=context.uci)
            return Data(records, UCI_FRAME, context.field_rows, context.uci.name, digest(records))
