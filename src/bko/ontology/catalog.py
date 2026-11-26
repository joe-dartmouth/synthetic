"""The dataset catalog: each frame added after Diabetes 130, with what loads it and its field rows.

A catalogued dataset is read the same way by every command: `Catalogued.read` ensures each pinned
source, runs the entry's loader and builds the entry's frame from the records, then checks the
records against it. `bko.evaluate.data` reaches the catalog through its context's `catalog`, so
a test can swap an entry's pins (`dataclasses.replace`), and `bko.gaps` lists each entry's gaps
as its own command. ADR `research/ontology-mapping-design` records why frames join here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

import pandas as pd

from bko.datasets.acs import ACS_NAME, load_acs
from bko.datasets.acs_frame import acs_frame
from bko.datasets.fetch import UrlOpener, ensure
from bko.datasets.frames import FrameSpec, check_records
from bko.datasets.nhamcs import NHAMCS_NAME, load_nhamcs
from bko.datasets.nhamcs_frame import nhamcs_frame
from bko.datasets.nhanes import NHANES_NAME, load_nhanes
from bko.datasets.nhanes_frame import nhanes_frame
from bko.datasets.sources import (
    ACS_MA_2024,
    HPO,
    NHAMCS_ED_2022,
    NHANES,
    OCCO,
    SUPPORT2,
    Agreement,
    Source,
)
from bko.datasets.support2 import SUPPORT2_NAME, load_support2
from bko.datasets.support2_frame import support2_frame
from bko.ontology.acs_rows import ACS_FIELD_ROWS
from bko.ontology.nhamcs_rows import NHAMCS_FIELD_ROWS
from bko.ontology.nhanes_rows import NHANES_FIELD_ROWS
from bko.ontology.schema import FieldMapping
from bko.ontology.support2_rows import SUPPORT2_FIELD_ROWS


class Loader(Protocol):
    """Reads a dataset's records from the verified snapshots of its sources."""

    def __call__(self, data_dir: Path, *, sources: Sequence[Source]) -> pd.DataFrame:
        """Return the records, indexed by `record_id`, every column int64."""
        ...


@dataclass(frozen=True, slots=True)
class Catalogued:
    """One catalogued dataset.

    Attributes:
        name: The dataset's name: its `Dataset` value, its `gaps` command and its cells' name.
        sources: Its pinned downloads, in the order its loader reads them.
        load: Reads its records from the verified snapshots of `sources`.
        frame: Builds its frame from its records.
        rows: Its field rows.
        releases: The pinned ontology release for each concept prefix its rows name; None reads
            the command's own releases (`resolved_releases`).
    """

    name: str
    sources: tuple[Source, ...]
    load: Loader
    frame: Callable[[pd.DataFrame], FrameSpec]
    rows: tuple[FieldMapping, ...]
    releases: Mapping[str, Source] | None = None

    def __post_init__(self) -> None:
        """Refuse an entry with no source, or sources held under different agreements.

        Raises:
            ValueError: If `sources` is empty or its agreements differ.
        """
        if not self.sources:
            msg = f"catalog entry {self.name} names no source"
            raise ValueError(msg)
        if len({source.agreement for source in self.sources}) != 1:
            msg = f"catalog entry {self.name}: its sources are held under different agreements"
            raise ValueError(msg)

    def resolved_releases(self, default: Mapping[str, Source]) -> Mapping[str, Source]:
        """The entry's own `releases`, or `default`, a command's own, when it pins none."""
        return default if self.releases is None else self.releases

    @property
    def agreement(self) -> Agreement | None:
        """The agreement every source is held under; None for open data."""
        return self.sources[0].agreement

    def read(self, data_dir: Path, *, open_url: UrlOpener) -> tuple[pd.DataFrame, FrameSpec]:
        """Ensure each source, load the records, build their frame and check them against it.

        Raises:
            DatasetError: As `bko.datasets.fetch.ensure`, the loader and
                `bko.datasets.frames.check_records`.
            OSError: If the data root cannot be written.
        """
        for source in self.sources:
            ensure(source, data_dir, open_url=open_url)
        records = self.load(data_dir, sources=self.sources)
        spec = self.frame(records)
        check_records(records, spec)
        return records, spec


CATALOG: Final[Mapping[str, Catalogued]] = MappingProxyType(
    {
        SUPPORT2_NAME: Catalogued(
            name=SUPPORT2_NAME,
            sources=(SUPPORT2,),
            load=load_support2,
            frame=support2_frame,
            rows=SUPPORT2_FIELD_ROWS,
        ),
        ACS_NAME: Catalogued(
            name=ACS_NAME,
            sources=(ACS_MA_2024,),
            load=load_acs,
            frame=acs_frame,
            rows=ACS_FIELD_ROWS,
            releases=MappingProxyType({"HP": HPO, "OCCO": OCCO}),
        ),
        NHANES_NAME: Catalogued(
            name=NHANES_NAME,
            sources=NHANES,
            load=load_nhanes,
            frame=nhanes_frame,
            rows=NHANES_FIELD_ROWS,
        ),
        NHAMCS_NAME: Catalogued(
            name=NHAMCS_NAME,
            sources=(NHAMCS_ED_2022,),
            load=load_nhamcs,
            frame=nhamcs_frame,
            rows=NHAMCS_FIELD_ROWS,
        ),
    }
)
"""Each catalogued dataset by name, in the order the frames were added."""
