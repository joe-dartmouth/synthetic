"""The data a named run scores: one `Data` value, chosen by `NamedRun.dataset`, every stage reads.

`score` loads it once (`load`) and hands it to the harness, the BKO arms, the rehearsal and the
candidates file, so each reads the same records, frame, field mappings and digest, and the run's
logs name the frame. ADR `research/uci-evaluation-frame` records why the run, not a flag, chooses
it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, assert_never

import pandas as pd

from bko.datasets.fetch import UrlOpener, ensure
from bko.datasets.frames import FrameSpec
from bko.datasets.sources import Source
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.ontology.schema import FieldMapping


logger = logging.getLogger("bko.evaluate")


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
        digest: The sha256 of `records` as `digest` renders them, computed from them when the
            value is made, which the candidates file's header names.
    """

    records: pd.DataFrame
    frame: FrameSpec
    field_rows: tuple[FieldMapping, ...]
    name: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute `digest` from `records`, so the two never disagree."""
        object.__setattr__(self, "digest", digest(self.records))


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
    """The sha256 of `records` as CSV with newline line ends (`to_csv(lineterminator=...)`).

    `Data.digest` and each arm's `arm` log line (`bko.evaluate.run`) hash a frame this way.
    """
    return hashlib.sha256(records.to_csv(lineterminator="\n").encode()).hexdigest()


def load(dataset: Dataset, data_dir: Path, context: Reach) -> Data:
    """Load `dataset` from `data_dir`, fetching UCI Heart Disease if the data root lacks it.

    Logs one `data` line naming the dataset, its frame, its record count and its digest.

    Raises:
        DatasetError: As `bko.datasets.fetch.ensure` and `bko.datasets.uci.load_uci`.
        OSError: If the data root cannot be written.
    """
    match dataset:
        case Dataset.UCI:
            ensure(context.uci, data_dir, open_url=context.open_url)
            records = load_uci(data_dir, source=context.uci)
            data = Data(records, UCI_FRAME, context.field_rows, context.uci.name)
        case _:
            assert_never(dataset)
    logger.info(
        "data dataset=%s frame=%s records=%d digest=%s",
        dataset,
        data.frame.name,
        len(data.records),
        data.digest,
    )
    return data
