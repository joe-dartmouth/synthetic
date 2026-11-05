"""The data a named run scores: one `Data` value, chosen by `NamedRun.dataset`, every stage reads.

`score` loads it once (`load`) and hands it to the harness, the BKO arms, the rehearsal and the
candidates file, so each reads the same records, frame, field mappings and digest, and the run's
logs name the frame. ADR `research/uci-evaluation-frame` records why the run, not a flag, chooses
it. The Synthea frame, its indicators and its field rows are defined here, from
`bko.ontology.mapping.CODE_MAPPINGS` (ADR `research/synthea-downstream-design`); the Diabetes 130
frame's are in `bko.evaluate.diabetes`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, assert_never

import pandas as pd

from bko.datasets.cells import LABEL, LABEL_CODES, SEXES
from bko.datasets.diabetes import load_diabetes
from bko.datasets.fetch import UrlOpener, ensure
from bko.datasets.frames import FrameSpec, RangeConstraint
from bko.datasets.sources import Agreement, Source
from bko.datasets.synthea import ISCHEMIC_HEART_DISEASE, frame_records, load_population
from bko.datasets.synthea_run import SyntheaRun, check_population, population_dir
from bko.datasets.uci import UCI_FRAME, load_uci
from bko.evaluate.diabetes import (
    DIABETES_FIELD_ROWS,
    DIABETES_FRAME,
    DIABETES_INDICATORS,
    DIABETES_NAME,
    check_records,
)
from bko.ontology.mapping import CODE_MAPPINGS
from bko.ontology.schema import CodeMapping, Codes, FieldMapping


logger = logging.getLogger("bko.evaluate")


class Dataset(StrEnum):
    """The datasets a named run can score."""

    UCI = "uci"
    SYNTHEA = "synthea"
    DIABETES = DIABETES_NAME


SYNTHEA_NAME: Final = "synthea"
"""The Synthea frame's name and its dataset's source name, as E2's fragments take it."""

SYNTHEA_ADULTS_FROM: Final = 40
"""The youngest age, in whole years on the snapshot date, a Synthea record has."""

INDICATOR_ROWS: Final[tuple[CodeMapping, ...]] = tuple(
    row for row in CODE_MAPPINGS if row.snomed != ISCHEMIC_HEART_DISEASE
)
"""The code rows that become the Synthea frame's indicators: every one but the label's."""


def indicator(row: CodeMapping) -> str:
    """The indicator column of `row`: its label, lowercased, each run of other characters `_`."""
    return re.sub(r"[^a-z0-9]+", "_", row.label.lower()).strip("_")


def indicators(rows: Sequence[CodeMapping]) -> Mapping[str, str]:
    """Each row's indicator column and its condition code, in `rows`' order, frozen.

    Raises:
        ValueError: If two rows name one column, or a row names `age`, `sex` or the label.
    """
    found: dict[str, str] = {}
    for row in rows:
        name = indicator(row)
        if name in found:
            msg = f"indicator {name!r} is named twice"
            raise ValueError(msg)
        if name in {"age", "sex", LABEL}:
            msg = f"indicator {name!r} is one of the frame's own columns"
            raise ValueError(msg)
        found[name] = row.snomed
    return MappingProxyType(found)


SYNTHEA_INDICATORS: Final[Mapping[str, str]] = indicators(INDICATOR_ROWS)
"""Each indicator column's condition code, in `CODE_MAPPINGS` order; checked when the module is
imported."""

_PRESENCE: Final[Mapping[int, str]] = MappingProxyType({0: "absent", 1: "present"})

SYNTHEA_FRAME: Final = FrameSpec(
    name=SYNTHEA_NAME,
    features=("age", "sex", *SYNTHEA_INDICATORS),
    numeric=("age",),
    fractional=(),
    fraction_digits=0,
    categorical=("sex", *SYNTHEA_INDICATORS),
    codebook={"sex": SEXES, **dict.fromkeys(SYNTHEA_INDICATORS, _PRESENCE)},
    descriptions={
        "age": "age in years on the snapshot date",
        "sex": "sex",
        **{indicator(row): f"{row.label} recorded by the snapshot date" for row in INDICATOR_ROWS},
        LABEL: "ischemic heart disease recorded by the snapshot date",
    },
    ranges=(
        RangeConstraint(
            field="age",
            unit="a",
            low=SYNTHEA_ADULTS_FROM,
            high=120,
            source="research/synthea-downstream-design",
            limits="the frame keeps patients aged 40 and over; 120 is UCI's team-specified top",
        ),
    ),
    label_codes=tuple(sorted(LABEL_CODES)),
    label_meanings={0: "no ischemic heart disease", 1: "ischemic heart disease"},
    edges=(65,),
    study="ischemic heart disease prediction",
    cohort="patient aged 40 or over in a generated population",
)
"""E3's frame over the pinned Synthea population (ADR `research/synthea-downstream-design`)."""


def _field_row(row: CodeMapping) -> FieldMapping:
    column = LABEL if row.snomed == ISCHEMIC_HEART_DISEASE else indicator(row)
    return FieldMapping(
        when=(Codes(column, frozenset({1})),),
        concept=row.concept,
        label=row.label,
        source=row.source,
        limits="the code recorded by the snapshot date; a patient without it may still hold it",
    )


SYNTHEA_FIELD_ROWS: Final[tuple[FieldMapping, ...]] = tuple(_field_row(r) for r in CODE_MAPPINGS)
"""The Synthea frame's field rows, one per `CODE_MAPPINGS` row: each indicator's
`Codes(column, {1})`, and the label's `Codes(target, {1})` for ischemic heart disease (ADR
`research/ontology-mapping-design`)."""


@dataclass(frozen=True, slots=True, eq=False)
class Data:
    """A run's records and what every stage reads about them.

    Attributes:
        records: The complete records, indexed by `record_id`, in `frame`'s columns.
        frame: Their frame.
        field_rows: The field mappings the BKO arms observe cells by.
        name: The dataset's source name; each cell's ontology fragment takes it.
        agreement: The data use agreement the records are held under; None for open data. Every
            Messages arm refuses records whose agreement grants no permission to send them.
        digest: The sha256 of `records` as `digest` renders them, computed from them when the
            value is made, which the candidates file's header names.
    """

    records: pd.DataFrame
    frame: FrameSpec
    field_rows: tuple[FieldMapping, ...]
    name: str
    agreement: Agreement | None
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

    @property
    def synthea(self) -> SyntheaRun:
        """The pinned Synthea population's run."""
        ...

    @property
    def synthea_jar(self) -> Source:
        """The jar the pinned Synthea population was generated with."""
        ...

    @property
    def diabetes(self) -> Source:
        """The pinned Diabetes 130 archive."""
        ...


def digest(records: pd.DataFrame) -> str:
    """The sha256 of `records` as CSV with newline line ends (`to_csv(lineterminator=...)`).

    `Data.digest` and each arm's `arm` log line (`bko.evaluate.run`) hash a frame this way.
    """
    return digested(records)[1]


def digested(records: pd.DataFrame) -> tuple[str, str]:
    """The CSV text `digest` hashes, and its sha256, from one rendering of `records`."""
    text = records.to_csv(lineterminator="\n")
    return text, hashlib.sha256(text.encode()).hexdigest()


def field_rows(dataset: Dataset, context: Reach) -> tuple[FieldMapping, ...]:
    """The field rows `load` gives `dataset`'s data, read with no data loaded.

    UCI's are the context's; Synthea's are `SYNTHEA_FIELD_ROWS` and Diabetes 130's
    `DIABETES_FIELD_ROWS`, whatever the context's.
    """
    match dataset:
        case Dataset.UCI:
            return context.field_rows
        case Dataset.SYNTHEA:
            return SYNTHEA_FIELD_ROWS
        case Dataset.DIABETES:
            return DIABETES_FIELD_ROWS
        case _:
            assert_never(dataset)


def load(dataset: Dataset, data_dir: Path, context: Reach) -> Data:
    """Load `dataset` from `data_dir`, fetching a pinned archive the data root lacks.

    UCI's records are read from UCI Heart Disease. Synthea's are the pinned population's, checked
    against its `run.json` with no Java and never generated (`make synthea` generates it), kept
    from `SYNTHEA_ADULTS_FROM`. Diabetes 130's are `bko.datasets.diabetes.load_diabetes`'s over
    `DIABETES_INDICATORS`, checked against `DIABETES_FRAME` (`check_records`). Each takes
    `field_rows(dataset, context)`, and a fetched source's data carries the source's agreement.
    Logs one `data` line naming the dataset, its frame, its record count, its digest (the
    records' sha256) and its agreement: `none` for open data, else the agreement's name quoted.

    Raises:
        DatasetError: As `bko.datasets.fetch.ensure`, `bko.datasets.uci.load_uci` and
            `bko.datasets.diabetes.load_diabetes`, as `check_records`, or as
            `bko.datasets.synthea_run.check_population` and
            `bko.datasets.synthea.load_population`.
        OSError: If the data root cannot be written.
    """
    rows = field_rows(dataset, context)
    match dataset:
        case Dataset.UCI:
            ensure(context.uci, data_dir, open_url=context.open_url)
            records = load_uci(data_dir, source=context.uci)
            data = Data(records, UCI_FRAME, rows, context.uci.name, context.uci.agreement)
        case Dataset.SYNTHEA:
            run = context.synthea
            directory = population_dir(run, data_dir, context.synthea_jar)
            check_population(directory, run, context.synthea_jar)
            population = load_population(directory, snapshot_date=run.snapshot_date)
            records = frame_records(
                population,
                label=ISCHEMIC_HEART_DISEASE,
                indicators=SYNTHEA_INDICATORS,
                adults_from=SYNTHEA_ADULTS_FROM,
            )
            data = Data(records, SYNTHEA_FRAME, rows, SYNTHEA_NAME, None)
        case Dataset.DIABETES:
            source = context.diabetes
            ensure(source, data_dir, open_url=context.open_url)
            records = load_diabetes(data_dir, indicators=DIABETES_INDICATORS, source=source)
            check_records(records)
            data = Data(records, DIABETES_FRAME, rows, source.name, source.agreement)
        case _:
            assert_never(dataset)
    agreement = data.agreement
    logger.info(
        "data dataset=%s frame=%s records=%d digest=%s agreement=%s",
        dataset,
        data.frame.name,
        len(data.records),
        data.digest,
        "none" if agreement is None else json.dumps(agreement.name),
    )
    return data
